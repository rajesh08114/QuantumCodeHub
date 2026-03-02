import pytest

from api.routers import completion, explanation, transpilation


@pytest.mark.asyncio
async def test_non_quantum_explain_uses_direct_llm(client, monkeypatch):
    async def fake_generate_code(*args, **kwargs):
        return {"generated_text": "This code adds two numbers.", "tokens_used": 12}

    monkeypatch.setattr(explanation.llm_service, "generate_code", fake_generate_code)

    response = await client.post(
        "/api/explain/code",
        json={
            "code": "def add(a, b):\n    return a + b",
            "framework": "qiskit",
            "detail_level": "intermediate",
            "include_math": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["gate_breakdown"].startswith("N/A")
    assert body["runtime_validation"]["status"] == "skipped_non_quantum_domain"


@pytest.mark.asyncio
async def test_non_quantum_completion_uses_direct_llm(client, monkeypatch):
    async def fake_generate_code(*args, **kwargs):
        return {
            "generated_text": "console.log(value) - Log value to console",
            "tokens_used": 9,
        }

    monkeypatch.setattr(completion.llm_service, "generate_code", fake_generate_code)

    response = await client.post(
        "/api/complete/suggest",
        json={
            "code_prefix": "const value = 10\n",
            "framework": "qiskit",
            "cursor_line": 1,
            "cursor_column": 15,
            "max_suggestions": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["runtime_validation"]["status"] == "skipped_non_quantum_domain"
    assert len(body["suggestions"]) >= 1


@pytest.mark.asyncio
async def test_jsx_tsx_conversion_direct_path(client, monkeypatch):
    async def fake_generate_code(*args, **kwargs):
        return {
            "generated_text": "const App: React.FC = () => <div>Hello</div>;",
            "tokens_used": 20,
        }

    monkeypatch.setattr(transpilation.llm_service, "generate_code", fake_generate_code)

    response = await client.post(
        "/api/transpile/convert",
        json={
            "source_code": "const App = () => <div>Hello</div>;",
            "source_framework": "jsx",
            "target_framework": "tsx",
            "preserve_comments": True,
            "optimize": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["source_framework"] == "jsx"
    assert body["target_framework"] == "tsx"
    assert "React.FC" in body["transpiled_code"]


@pytest.mark.asyncio
async def test_non_quantum_code_generation_accepts_generic_framework(client, monkeypatch):
    from api.routers import code_generation

    async def fake_generate_code(*args, **kwargs):
        return {
            "generated_text": "const Improved = () => <section>Improved</section>;",
            "tokens_used": 18,
            "provider": "test",
            "model": "test-model",
            "attempt": 1,
            "fallback_used": False,
        }

    monkeypatch.setattr(code_generation.llm_service, "generate_code", fake_generate_code)

    response = await client.post(
        "/api/code/generate",
        json={
            "prompt": "Improve this JSX component for readability and keep behavior unchanged.",
            "framework": "generic",
            "include_explanation": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["framework"] == "generic"
    assert body["validation_passed"] is True
    assert body["metadata"]["runtime_validation"]["status"] == "skipped_non_quantum_domain"
