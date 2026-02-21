def _normalize_text_block(text: str, fallback: str = "No additional documentation context provided.") -> str:
    value = (text or "").strip()
    return value if value else fallback


def _optional_block(title: str, text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    return f"\n{title}:\n{value}\n"


class CodeGenerationPrompts:
    """Prompt templates for code generation."""

    @staticmethod
    def get_system_message(framework: str) -> str:
        """Get framework-specific system message."""
        system_messages = {
            "qiskit": """You are a senior quantum software engineer specializing in IBM Qiskit.
Produce accurate, runnable Python code with modern Qiskit APIs.
Never output placeholders or pseudo-code.""",
            "pennylane": """You are a senior quantum software engineer specializing in PennyLane.
Produce accurate, runnable Python code with valid device + QNode usage.
Never output placeholders or pseudo-code.""",
            "cirq": """You are a senior quantum software engineer specializing in Cirq.
Produce accurate, runnable Python code with modern Cirq idioms.
Never output placeholders or pseudo-code.""",
            "torchquantum": """You are a senior quantum ML engineer specializing in TorchQuantum.
Produce accurate, runnable hybrid quantum-classical PyTorch code.
Never output placeholders or pseudo-code.""",
        }
        return system_messages.get(framework, system_messages["qiskit"])

    @staticmethod
    def build_generation_prompt(
        user_query: str,
        framework: str,
        rag_context: str,
        num_qubits: int = None,
        conversation_context: str = "",
        compatibility_context: str = "",
    ) -> str:
        """Build complete prompt for code generation."""
        qubit_line = f"Requested qubits: {num_qubits}" if num_qubits else "Requested qubits: not specified"
        docs = _normalize_text_block(rag_context)
        memory_block = _optional_block("Conversation memory", conversation_context)
        compatibility_block = _optional_block("Runtime compatibility context", compatibility_context)

        return f"""Task: Generate production-ready {framework} quantum code.

User request:
{user_query}

{qubit_line}
{memory_block}
{compatibility_block}

Documentation context:
{docs}

Requirements:
1. Return complete runnable Python code with all imports.
2. Preserve the user intent exactly; do not change problem scope.
3. Use framework-idiomatic APIs and current syntax.
4. Add concise comments only where they improve readability.
5. Include measurement/output steps when relevant.
6. Avoid markdown outside requested output format.
7. Avoid deprecated APIs and keep package usage version-compatible.

Output format (strict):
```python
# complete {framework} solution
```
Explanation:
- 2-4 concise bullet points describing approach and key quantum operations.
Runtime Recommendations:
- Python: <supported version>
- Packages: <framework + critical dependencies with versions>
"""


class TranspilationPrompts:
    """Prompt templates for code transpilation."""

    @staticmethod
    def build_transpilation_prompt(
        source_code: str,
        source_framework: str,
        target_framework: str,
        rag_context: str,
        compatibility_context: str = "",
    ) -> str:
        """Build prompt for framework-to-framework transpilation."""
        docs = _normalize_text_block(rag_context)
        compatibility_block = _optional_block("Runtime compatibility context", compatibility_context)
        return f"""Task: Transpile quantum code from {source_framework} to {target_framework}.

Source code:
```python
{source_code}
```

Target framework: {target_framework}

Documentation context:
{docs}
{compatibility_block}

Requirements:
1. Preserve algorithmic behavior and gate logic.
2. Use idiomatic {target_framework} constructs and imports.
3. Keep code runnable as-is.
4. If an operation has no exact equivalent, use the closest correct pattern and explain.
5. Avoid deprecated APIs for the target version range.

Output format (strict):
```python
# transpiled {target_framework} code
```
Notable Changes:
- <change 1>
- <change 2>
"""


class ExplanationPrompts:
    """Prompt templates for code explanation."""

    @staticmethod
    def build_explanation_prompt(
        code: str,
        framework: str,
        detail_level: str = "intermediate",
        rag_context: str = "",
        conversation_context: str = "",
        compatibility_context: str = "",
    ) -> str:
        """Build prompt for code explanation."""
        detail_instructions = {
            "beginner": "Use simple language and avoid heavy math notation unless essential.",
            "intermediate": "Use precise terminology with moderate technical depth.",
            "advanced": "Include mathematical reasoning and deeper algorithmic detail.",
        }
        detail_instruction = detail_instructions.get(detail_level, detail_instructions["intermediate"])
        docs = _normalize_text_block(rag_context)

        memory_block = _optional_block("Conversation memory", conversation_context)
        compatibility_block = _optional_block("Runtime compatibility context", compatibility_context)

        return f"""Task: Explain this {framework} quantum code.

Code:
```python
{code}
```

Explanation level: {detail_level}
Style instruction: {detail_instruction}
{memory_block}
{compatibility_block}

Documentation context:
{docs}

Output headings (strict, keep exact labels):
Overall Purpose:
Gate-by-Gate Breakdown:
Quantum Concepts:
Mathematical Representation:
Practical Applications:
Runtime Recommendations:
"""


class ErrorFixingPrompts:
    """Prompt templates for error detection and fixing."""

    @staticmethod
    def build_error_fixing_prompt(
        code: str,
        framework: str,
        error_message: str = None,
        rag_context: str = "",
        conversation_context: str = "",
        compatibility_context: str = "",
    ) -> str:
        """Build prompt for fixing erroneous code."""
        error_section = "No explicit traceback provided."
        if error_message:
            error_section = error_message.strip()
        docs = _normalize_text_block(rag_context)

        memory_block = _optional_block("Conversation memory", conversation_context)
        compatibility_block = _optional_block("Runtime compatibility context", compatibility_context)

        return f"""Task: Fix broken {framework} quantum code and return a runnable solution.

Buggy code:
```python
{code}
```

Error message / traceback:
{error_section}
{memory_block}
{compatibility_block}

Documentation context:
{docs}

Requirements:
1. Identify concrete defects in the provided code.
2. Preserve the intended logic unless objectively incorrect.
3. Return corrected executable code with required imports.
4. Keep fixes minimal and justified.
5. Avoid deprecated APIs and use version-compatible replacements.

Output format (strict):
Identified Issues:
- <issue 1>
- <issue 2>

Corrected Code:
```python
# fixed code
```

Explanation of Fixes:
- <fix 1>
- <fix 2>
Runtime Recommendations:
- Python: <supported version>
- Packages: <framework + critical dependencies with versions>
"""


class CompletionPrompts:
    """Prompt templates for auto-completion."""

    @staticmethod
    def build_completion_prompt(
        code_prefix: str,
        framework: str,
        cursor_context: dict,
        rag_context: str,
        max_suggestions: int = 5,
        compatibility_context: str = "",
    ) -> str:
        """Build prompt for code completion suggestions."""
        docs = _normalize_text_block(rag_context)
        suggestion_count = max(1, min(max_suggestions, 10))
        compatibility_block = _optional_block("Runtime compatibility context", compatibility_context)

        return f"""Task: Suggest likely next code completions for {framework}.

Current code prefix:
```python
{code_prefix}
```
[CURSOR]

Context:
- scope: {cursor_context.get('scope', 'global')}
- last_statement: {cursor_context.get('last_statement', 'N/A')}
- variables: {cursor_context.get('variables', [])}
- imports: {cursor_context.get('imports', [])}

Documentation context:
{docs}
{compatibility_block}

Output format (strict):
Return exactly {suggestion_count} lines if possible.
Each line: <completion code> - <short description>
Do not add extra commentary or version notes.
"""


class ChatbotPrompts:
    """Prompt templates for general chatbot responses."""

    @staticmethod
    def build_general_prompt(
        framework: str,
        user_question: str,
        rag_context: str,
        conversation_context: str = "",
        cross_session_summary: str = "",
        compatibility_context: str = "",
        math_focus: bool = False,
    ) -> str:
        docs = _normalize_text_block(rag_context)
        memory_block = ""
        if (conversation_context or "").strip():
            memory_block += f"\nCurrent session memory:\n{conversation_context}\n"
        if (cross_session_summary or "").strip():
            memory_block += f"\nOther session summaries:\n{cross_session_summary}\n"
        compatibility_block = _optional_block("Runtime compatibility context", compatibility_context)
        math_instruction = (
            "Include equations/state-vectors/matrix notation where useful, then explain them intuitively."
            if math_focus
            else "Use mathematical notation only when it materially improves understanding."
        )

        return f"""You are an expert quantum programming assistant and senior software engineer.
Answer clearly, correctly, and in a structured way grounded in provided documentation.
When relevant, explain quantum state evolution, core equations, and intuition.
If code is requested, return clean runnable Python.
If uncertain, state assumptions explicitly.

Framework preference: {framework}
User question:
{user_question}
{memory_block}
{compatibility_block}

Documentation context:
{docs}

Response style requirements:
1. Start with a direct answer.
2. Add intuition/analogy for complex ideas.
3. {math_instruction}
4. If code is needed, include a single runnable code block.
5. Mention compatible runtime/package versions only when code or dependency details are involved.
"""
