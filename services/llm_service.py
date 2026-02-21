"""
LLM interaction service with ordered provider fallback.
LangChain is used for provider orchestration where possible.
"""
import asyncio
import logging
import re
import time
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover - optional dependency
    ChatOllama = None

from core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for text/code generation across multiple providers."""

    _KNOWN_PROVIDERS = ("hf_api", "resp_api", "ollama", "hf_qwen_api", "local")

    def __init__(self):
        self.backend = (settings.LLM_BACKEND or "api").strip().lower()

        # Hugging Face Router
        self.api_url = settings.HF_INFERENCE_URL or "https://router.huggingface.co/v1"
        self.model_id = settings.HF_MODEL_ID or "ibm-granite/granite-4.0-h-small"
        self.hf_qwen_model_id = settings.HF_QWEN_MODEL_ID or "Qwen/Qwen2.5-Coder-7B-Instruct"
        self.hf_headers = {"Authorization": f"Bearer {settings.HF_API_KEY}"}

        # Generic REST response API
        self.resp_api_url = (settings.RESP_API_URL or "").strip()
        self.resp_api_key = (settings.RESP_API_KEY or "").strip()
        self.resp_api_model = (settings.RESP_API_MODEL or "").strip()

        # Ollama local
        self.ollama_base_url = (settings.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
        self.ollama_model = settings.OLLAMA_MODEL or "llama3.2"

        # Optional local transformers inference
        self.local_model_ref = settings.LOCAL_MODEL_PATH or settings.LOCAL_MODEL_ID or self.model_id
        self.local_files_only = settings.LOCAL_MODEL_FILES_ONLY
        self._local_tokenizer = None
        self._local_model = None

        self.max_tokens = settings.MAX_TOKENS or 2048
        self.temperature = settings.TEMPERATURE or 0.3
        self.request_timeout = max(5, int(settings.LLM_REQUEST_TIMEOUT_SECONDS or 45))
        self.provider_chain = self._build_provider_chain()
        self.http = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=50)
        self.http.mount("http://", adapter)
        self.http.mount("https://", adapter)

        logger.info(
            "LLM service initialized backend=%s chain=%s primary_model=%s ollama_model=%s qwen_model=%s resp_api_configured=%s langchain_ollama=%s",
            self.backend,
            ",".join(self.provider_chain),
            self.model_id,
            self.ollama_model,
            self.hf_qwen_model_id,
            bool(self.resp_api_url),
            bool(ChatOllama),
        )

    @staticmethod
    def _clip_text(text: str, limit: int = 240) -> str:
        normalized = " ".join((text or "").split())
        return normalized if len(normalized) <= limit else f"{normalized[:limit]}..."

    def _resolve_http_timeout(self, request_timeout: Optional[float] = None):
        read_timeout = float(request_timeout) if request_timeout is not None else float(self.request_timeout)
        read_timeout = max(5.0, read_timeout)
        connect_timeout = min(3.0, read_timeout)
        return (connect_timeout, read_timeout)

    def _resolve_langchain_timeout(self, request_timeout: Optional[float] = None) -> float:
        timeout = float(request_timeout) if request_timeout is not None else float(self.request_timeout)
        return max(5.0, timeout)

    def _normalize_provider(self, value: str) -> Optional[str]:
        aliases = {
            "api": "hf_api",
            "hf": "hf_api",
            "hf_api": "hf_api",
            "hf_router": "hf_api",
            "router": "hf_api",
            "resp": "resp_api",
            "rest": "resp_api",
            "resp_api": "resp_api",
            "response_api": "resp_api",
            "ollama": "ollama",
            "local_ollama": "ollama",
            "qwen": "hf_qwen_api",
            "qwen_api": "hf_qwen_api",
            "hf_qwen_api": "hf_qwen_api",
            "local": "local",
        }
        candidate = (value or "").strip().lower()
        normalized = aliases.get(candidate, candidate)
        return normalized if normalized in self._KNOWN_PROVIDERS else None

    def _parse_provider_chain(self, raw_chain: str) -> List[str]:
        providers: List[str] = []
        for item in (raw_chain or "").split(","):
            provider = self._normalize_provider(item)
            if provider and provider not in providers:
                providers.append(provider)
        return providers

    def _build_provider_chain(self) -> List[str]:
        env_chain = self._parse_provider_chain(settings.LLM_PROVIDER_CHAIN)
        if env_chain:
            return env_chain
        if self.backend == "ollama":
            return ["ollama", "hf_api", "resp_api", "hf_qwen_api"]
        if self.backend == "local":
            return ["local", "ollama", "hf_api", "resp_api", "hf_qwen_api"]
        return ["hf_api", "resp_api", "ollama", "hf_qwen_api"]

    def _provider_configured(self, provider: str) -> bool:
        if provider in ("hf_api", "hf_qwen_api"):
            return bool((settings.HF_API_KEY or "").strip())
        if provider == "resp_api":
            return bool(self.resp_api_url)
        if provider == "ollama":
            return bool(self.ollama_base_url and self.ollama_model)
        if provider == "local":
            return bool(self.local_model_ref)
        return False

    def _resolve_effective_chain(
        self,
        preferred_chain: Optional[List[str]] = None,
        force_provider: Optional[str] = None,
    ) -> List[str]:
        base_chain = self.provider_chain
        if preferred_chain:
            parsed = []
            for item in preferred_chain:
                provider = self._normalize_provider(item)
                if provider and provider not in parsed:
                    parsed.append(provider)
            if parsed:
                base_chain = parsed

        if not force_provider:
            return base_chain

        forced = self._normalize_provider(force_provider)
        if not forced:
            return base_chain
        return [forced] + [item for item in base_chain if item != forced]

    def get_routing_info(self) -> Dict:
        chain = self.provider_chain
        return {
            "backend": self.backend,
            "provider_chain": chain,
            "configured_providers": {provider: self._provider_configured(provider) for provider in chain},
            "primary_model": self.model_id,
            "qwen_model": self.hf_qwen_model_id,
            "ollama_model": self.ollama_model,
            "resp_api_url_configured": bool(self.resp_api_url),
            "langchain": {
                "message_orchestration": True,
                "ollama_available": bool(ChatOllama),
            },
        }

    def _build_langchain_messages(self, prompt: str, system_message: Optional[str] = None) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt or ""))
        return messages

    @staticmethod
    def _extract_text_from_langchain_content(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    elif item.get("type") == "text" and isinstance(item.get("content"), str):
                        parts.append(item["content"])
            return "\n".join(part for part in parts if part).strip()
        return str(content or "")

    def _messages_to_openai_payload(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        payload: List[Dict[str, str]] = []
        for message in messages:
            role = "system" if isinstance(message, SystemMessage) else "user"
            payload.append(
                {
                    "role": role,
                    "content": self._extract_text_from_langchain_content(message.content),
                }
            )
        return payload

    def _extract_text_from_result(self, result: Dict) -> str:
        if not isinstance(result, dict):
            return str(result or "")
        if isinstance(result.get("generated_text"), str):
            return result["generated_text"]
        if isinstance(result.get("response"), str):
            return result["response"]
        message = result.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                if isinstance(first.get("text"), str):
                    return first["text"]
                msg = first.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"]
        return ""

    def _extract_tokens_from_result(self, result: Dict, generated_text: str) -> int:
        if isinstance(result, dict):
            direct_tokens = result.get("tokens_used")
            if isinstance(direct_tokens, (int, float)):
                return int(direct_tokens)
            usage = result.get("usage")
            if isinstance(usage, dict):
                total_tokens = usage.get("total_tokens")
                if isinstance(total_tokens, (int, float)):
                    return int(total_tokens)
                completion_tokens = usage.get("completion_tokens")
                prompt_tokens = usage.get("prompt_tokens")
                if isinstance(completion_tokens, (int, float)) and isinstance(prompt_tokens, (int, float)):
                    return int(completion_tokens) + int(prompt_tokens)
            eval_count = result.get("eval_count")
            prompt_eval_count = result.get("prompt_eval_count")
            if isinstance(eval_count, (int, float)) and isinstance(prompt_eval_count, (int, float)):
                return int(eval_count) + int(prompt_eval_count)
            if isinstance(eval_count, (int, float)):
                return int(eval_count)
        return max(1, int(len((generated_text or "").split()) * 1.3))

    def _extract_tokens_from_langchain_response(self, response, generated_text: str) -> int:
        usage_metadata = getattr(response, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            total_tokens = usage_metadata.get("total_tokens")
            if isinstance(total_tokens, (int, float)):
                return int(total_tokens)
            input_tokens = usage_metadata.get("input_tokens")
            output_tokens = usage_metadata.get("output_tokens")
            if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
                return int(input_tokens) + int(output_tokens)
        response_metadata = getattr(response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
            if isinstance(token_usage, dict):
                total_tokens = token_usage.get("total_tokens")
                if isinstance(total_tokens, (int, float)):
                    return int(total_tokens)
            eval_count = response_metadata.get("eval_count")
            prompt_eval_count = response_metadata.get("prompt_eval_count")
            if isinstance(eval_count, (int, float)) and isinstance(prompt_eval_count, (int, float)):
                return int(eval_count) + int(prompt_eval_count)
            if isinstance(eval_count, (int, float)):
                return int(eval_count)
        return max(1, int(len((generated_text or "").split()) * 1.3))

    def _ensure_local_model(self):
        if self._local_model is not None and self._local_tokenizer is not None:
            return
        logger.info("Loading local LLM model model_ref=%s", self.local_model_ref)
        try:
            self._local_tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_ref,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
            self._local_model = AutoModelForCausalLM.from_pretrained(
                self.local_model_ref,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
                torch_dtype=torch.float32,
            )
            self._local_model.eval()
            logger.info("Local LLM model loaded model_ref=%s", self.local_model_ref)
        except Exception as e:
            if self.local_files_only:
                raise RuntimeError(
                    "Local model files were not found. Set LOCAL_MODEL_PATH to your downloaded model "
                    "or set LOCAL_MODEL_FILES_ONLY=false to allow initial download."
                ) from e
            raise

    def _build_local_prompt(self, prompt: str, system_message: Optional[str] = None) -> str:
        user_prompt = prompt or ""
        if system_message:
            return (
                "You are a helpful coding assistant.\n"
                f"System instruction:\n{system_message}\n\n"
                f"User request:\n{user_prompt}\n\n"
                "Assistant:"
            )
        return user_prompt

    def _generate_local(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict:
        self._ensure_local_model()
        prompt_text = self._build_local_prompt(prompt, system_message)
        temp = self.temperature if temperature is None else temperature
        new_tokens = max_tokens or self.max_tokens
        inputs = self._local_tokenizer(prompt_text, return_tensors="pt")
        pad_token_id = self._local_tokenizer.eos_token_id or self._local_tokenizer.pad_token_id or 0
        with torch.no_grad():
            output = self._local_model.generate(
                **inputs,
                max_new_tokens=new_tokens,
                do_sample=temp > 0,
                temperature=max(temp, 1e-5),
                top_p=0.95,
                pad_token_id=pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output[0][input_len:]
        generated_text = self._local_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        tokens_used = int(generated_ids.shape[0]) if hasattr(generated_ids, "shape") else int(len(generated_ids))
        return {
            "generated_text": generated_text,
            "tokens_used": tokens_used,
            "model": self.local_model_ref,
            "provider": "local",
            "success": True,
        }

    def _ollama_chat_url(self) -> str:
        return f"{self.ollama_base_url}/api/chat"

    def _ollama_generate_url(self) -> str:
        return f"{self.ollama_base_url}/api/generate"

    def _build_ollama_prompt(self, prompt: str, system_message: Optional[str] = None) -> str:
        if not system_message:
            return prompt or ""
        return (
            f"System instruction:\n{system_message}\n\n"
            f"User request:\n{prompt or ''}\n\n"
            "Assistant:"
        )

    def _generate_ollama_langchain(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ) -> Dict:
        if ChatOllama is None:
            raise RuntimeError("langchain-ollama is not installed")
        temp = self.temperature if temperature is None else temperature
        num_predict = max_tokens or self.max_tokens
        timeout_seconds = self._resolve_langchain_timeout(request_timeout)
        model = ChatOllama(
            model=self.ollama_model,
            base_url=self.ollama_base_url,
            temperature=temp,
            num_predict=num_predict,
            sync_client_kwargs={"timeout": timeout_seconds},
        )
        response = model.invoke(self._build_langchain_messages(prompt, system_message))
        generated_text = self._extract_text_from_langchain_content(getattr(response, "content", ""))
        tokens_used = self._extract_tokens_from_langchain_response(response, generated_text)
        return {
            "generated_text": generated_text or "",
            "tokens_used": int(tokens_used),
            "model": self.ollama_model,
            "provider": "ollama",
            "success": True,
        }

    def _generate_ollama_http(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ) -> Dict:
        temp = self.temperature if temperature is None else temperature
        num_predict = max_tokens or self.max_tokens
        messages = self._messages_to_openai_payload(self._build_langchain_messages(prompt, system_message))
        response = self.http.post(
            self._ollama_chat_url(),
            json={
                "model": self.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temp, "num_predict": num_predict},
            },
            timeout=self._resolve_http_timeout(request_timeout),
        )
        if response.status_code == 404:
            response = self.http.post(
                self._ollama_generate_url(),
                json={
                    "model": self.ollama_model,
                    "prompt": self._build_ollama_prompt(prompt, system_message),
                    "stream": False,
                    "options": {"temperature": temp, "num_predict": num_predict},
                },
                timeout=self._resolve_http_timeout(request_timeout),
            )
        response.raise_for_status()
        result = response.json()
        generated_text = self._extract_text_from_result(result)
        tokens_used = self._extract_tokens_from_result(result, generated_text)
        return {
            "generated_text": generated_text or "",
            "tokens_used": int(tokens_used),
            "model": self.ollama_model,
            "provider": "ollama",
            "success": True,
        }

    def _generate_ollama(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ) -> Dict:
        try:
            return self._generate_ollama_langchain(
                prompt=prompt,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                request_timeout=request_timeout,
            )
        except Exception as e:
            logger.warning("LangChain Ollama call failed; falling back to direct HTTP: %s", e)
            return self._generate_ollama_http(
                prompt=prompt,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                request_timeout=request_timeout,
            )

    def _extract_model_id_from_url(self, url: str) -> Optional[str]:
        match = re.search(r"/models/(.+)$", url)
        return match.group(1) if match else None

    def _router_v1_chat_url(self) -> str:
        return f"{self.api_url.rstrip('/')}/chat/completions"

    def _parse_error_message(self, response: requests.Response) -> str:
        try:
            body = response.json()
            if isinstance(body, dict):
                return str(body.get("error") or body.get("message") or body)
            return str(body)
        except Exception:
            return response.text or "Unknown error"

    def _parse_error_code(self, response: requests.Response) -> Optional[str]:
        try:
            body = response.json()
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict):
                    return err.get("code")
        except Exception:
            return None
        return None

    def _run_hf_router_http(self, payload: Dict) -> Dict:
        timeout_value = payload.get("_request_timeout")
        request_payload = {k: v for k, v in payload.items() if k != "_request_timeout"}
        response = self.http.post(
            self._router_v1_chat_url(),
            headers=self.hf_headers,
            json=request_payload,
            timeout=self._resolve_http_timeout(timeout_value),
        )
        model_id = request_payload.get("model", "unknown")
        if response.status_code == 404 and "/hf-inference/models/" in self.api_url:
            legacy_model = self._extract_model_id_from_url(self.api_url) or model_id
            logger.warning("HF legacy path unavailable. Retrying router v1 model=%s", legacy_model)
            self.api_url = "https://router.huggingface.co/v1"
            request_payload["model"] = legacy_model
            response = self.http.post(
                self._router_v1_chat_url(),
                headers=self.hf_headers,
                json=request_payload,
                timeout=self._resolve_http_timeout(timeout_value),
            )
            model_id = legacy_model
        if response.status_code in (401, 403):
            err = self._parse_error_message(response)
            raise Exception(
                f"Hugging Face authentication failed ({response.status_code}): {err}. "
                "Verify HF_API_KEY and model access permissions."
            )
        if response.status_code == 400:
            err = self._parse_error_message(response)
            error_code = self._parse_error_code(response)
            raise Exception(f"Hugging Face request rejected (400, code={error_code}): {err}")
        if response.status_code == 404:
            err = self._parse_error_message(response)
            raise Exception(f"Hugging Face model unavailable ({model_id}): {err}")
        response.raise_for_status()
        result = response.json()
        generated_text = self._extract_text_from_result(result)
        tokens_used = self._extract_tokens_from_result(result, generated_text)
        return {
            "generated_text": generated_text,
            "tokens_used": int(tokens_used),
            "resolved_model": model_id,
        }

    def _generate_hf_router(
        self,
        prompt: str,
        model_id: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        provider_name: str = "hf_api",
        request_timeout: Optional[float] = None,
    ) -> Dict:
        payload = {
            "model": model_id,
            "messages": self._messages_to_openai_payload(self._build_langchain_messages(prompt, system_message)),
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": 0.95,
            "_request_timeout": request_timeout,
        }
        response_data = RunnableLambda(self._run_hf_router_http).invoke(payload)
        return {
            "generated_text": response_data.get("generated_text", ""),
            "tokens_used": int(response_data.get("tokens_used", 0)),
            "model": response_data.get("resolved_model", model_id),
            "provider": provider_name,
            "success": True,
        }

    def _resp_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.resp_api_key:
            headers["Authorization"] = f"Bearer {self.resp_api_key}"
        return headers

    def _run_resp_api_http(self, payload: Dict) -> Dict:
        timeout_value = payload.get("_request_timeout")
        request_payload = {k: v for k, v in payload.items() if k != "_request_timeout"}
        response = self.http.post(
            self.resp_api_url,
            headers=self._resp_headers(),
            json=request_payload,
            timeout=self._resolve_http_timeout(timeout_value),
        )
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {"generated_text": response.text}

    def _generate_resp_api(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ) -> Dict:
        if not self.resp_api_url:
            raise Exception("RESP_API_URL is not configured.")
        payload: Dict = {
            "prompt": prompt,
            "system_message": system_message,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "_request_timeout": request_timeout,
        }
        if self.resp_api_model:
            payload["model"] = self.resp_api_model
        result = RunnableLambda(self._run_resp_api_http).invoke(payload)
        generated_text = self._extract_text_from_result(result)
        tokens_used = self._extract_tokens_from_result(result, generated_text)
        resolved_model = (
            result.get("model")
            if isinstance(result, dict) and isinstance(result.get("model"), str)
            else self.resp_api_model or "resp_api_default"
        )
        return {
            "generated_text": generated_text,
            "tokens_used": int(tokens_used),
            "model": resolved_model,
            "provider": "resp_api",
            "success": True,
        }

    def _attempt_provider(
        self,
        provider: str,
        prompt: str,
        system_message: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        request_timeout: Optional[float] = None,
    ) -> Dict:
        if provider == "hf_api":
            return self._generate_hf_router(
                prompt=prompt,
                model_id=self.model_id,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                provider_name=provider,
                request_timeout=request_timeout,
            )
        if provider == "hf_qwen_api":
            return self._generate_hf_router(
                prompt=prompt,
                model_id=self.hf_qwen_model_id,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                provider_name=provider,
                request_timeout=request_timeout,
            )
        if provider == "resp_api":
            return self._generate_resp_api(
                prompt=prompt,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                request_timeout=request_timeout,
            )
        if provider == "ollama":
            return self._generate_ollama(
                prompt=prompt,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                request_timeout=request_timeout,
            )
        if provider == "local":
            return self._generate_local(
                prompt=prompt,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        raise Exception(f"Unknown provider '{provider}'")

    async def generate_code(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        preferred_chain: Optional[List[str]] = None,
        force_provider: Optional[str] = None,
    ) -> Dict:
        """
        Generate text/code using ordered provider fallback.

        Default order is configurable with `LLM_PROVIDER_CHAIN`.
        """
        effective_chain = self._resolve_effective_chain(
            preferred_chain=preferred_chain,
            force_provider=force_provider,
        )
        prompt_preview = self._clip_text(prompt)
        errors: List[str] = []
        call_start = time.perf_counter()

        logger.info(
            "LLM request started chain=%s max_tokens=%s temperature=%s prompt_preview=%s",
            ",".join(effective_chain),
            max_tokens or self.max_tokens,
            self.temperature if temperature is None else temperature,
            prompt_preview,
        )
        max_total_seconds = float(self.request_timeout) + 5.0

        for index, provider in enumerate(effective_chain, start=1):
            elapsed_seconds = time.perf_counter() - call_start
            remaining_seconds = max_total_seconds - elapsed_seconds
            if remaining_seconds <= 1:
                errors.append(f"global timeout reached at {elapsed_seconds:.1f}s before provider {provider}")
                logger.warning(
                    "LLM request budget exceeded chain=%s elapsed_seconds=%.1f max_total_seconds=%.1f",
                    ",".join(effective_chain),
                    elapsed_seconds,
                    max_total_seconds,
                )
                break

            if not self._provider_configured(provider):
                errors.append(f"{provider}: not configured")
                logger.warning(
                    "LLM provider skipped provider=%s attempt=%s/%s reason=not_configured",
                    provider,
                    index,
                    len(effective_chain),
                )
                continue

            attempt_start = time.perf_counter()
            try:
                logger.info(
                    "LLM provider attempt started provider=%s attempt=%s/%s",
                    provider,
                    index,
                    len(effective_chain),
                )
                result = await asyncio.to_thread(
                    self._attempt_provider,
                    provider,
                    prompt,
                    system_message,
                    max_tokens,
                    temperature,
                    remaining_seconds,
                )
                attempt_latency_ms = int((time.perf_counter() - attempt_start) * 1000)
                total_latency_ms = int((time.perf_counter() - call_start) * 1000)

                result["attempt"] = index
                result["fallback_used"] = index > 1
                result["latency_ms"] = attempt_latency_ms
                result["total_latency_ms"] = total_latency_ms

                logger.info(
                    "LLM provider attempt succeeded provider=%s model=%s attempt=%s/%s tokens_used=%s latency_ms=%s response_preview=%s",
                    result.get("provider", provider),
                    result.get("model", "unknown"),
                    index,
                    len(effective_chain),
                    result.get("tokens_used", 0),
                    attempt_latency_ms,
                    self._clip_text(result.get("generated_text", "")),
                )
                return result
            except requests.exceptions.Timeout:
                attempt_latency_ms = int((time.perf_counter() - attempt_start) * 1000)
                errors.append(f"{provider}: timed out after {attempt_latency_ms}ms")
                logger.warning(
                    "LLM provider timeout provider=%s attempt=%s/%s latency_ms=%s",
                    provider,
                    index,
                    len(effective_chain),
                    attempt_latency_ms,
                )
            except requests.exceptions.ConnectionError:
                attempt_latency_ms = int((time.perf_counter() - attempt_start) * 1000)
                errors.append(f"{provider}: connection error after {attempt_latency_ms}ms")
                logger.warning(
                    "LLM provider connection error provider=%s attempt=%s/%s latency_ms=%s",
                    provider,
                    index,
                    len(effective_chain),
                    attempt_latency_ms,
                )
            except requests.exceptions.HTTPError as e:
                attempt_latency_ms = int((time.perf_counter() - attempt_start) * 1000)
                status_code = e.response.status_code if e.response is not None else "unknown"
                errors.append(f"{provider}: http {status_code} ({e})")
                logger.warning(
                    "LLM provider http error provider=%s attempt=%s/%s status_code=%s latency_ms=%s error=%s",
                    provider,
                    index,
                    len(effective_chain),
                    status_code,
                    attempt_latency_ms,
                    e,
                )
            except Exception as e:
                attempt_latency_ms = int((time.perf_counter() - attempt_start) * 1000)
                errors.append(f"{provider}: {e}")
                logger.warning(
                    "LLM provider failed provider=%s attempt=%s/%s latency_ms=%s error=%s",
                    provider,
                    index,
                    len(effective_chain),
                    attempt_latency_ms,
                    e,
                )

        logger.error(
            "LLM request failed all providers chain=%s errors=%s",
            ",".join(effective_chain),
            " | ".join(errors) if errors else "none",
        )
        raise Exception(
            "All configured LLM providers failed. "
            f"Tried chain={','.join(effective_chain)}. "
            f"Errors={' | '.join(errors) if errors else 'unknown'}"
        )

    async def batch_generate(
        self,
        prompts: List[str],
        system_message: Optional[str] = None,
    ) -> List[Dict]:
        """Generate code for multiple prompts."""
        results = []
        for prompt in prompts:
            result = await self.generate_code(prompt, system_message)
            results.append(result)
        return results


# Singleton instance
llm_service = LLMService()
