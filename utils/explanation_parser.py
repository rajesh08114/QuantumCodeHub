"""
Utilities for parsing structured explanation text from LLM markdown output.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

_SECTION_ALIASES = {
    "overview": ("overall purpose", "purpose", "overview"),
    "gate_breakdown": (
        "gate-by-gate breakdown",
        "gate by gate breakdown",
        "gate breakdown",
        "breakdown",
    ),
    "quantum_concepts": ("quantum concepts", "concepts", "quantum principles", "principles"),
    "mathematics": (
        "mathematical representation",
        "mathematics",
        "mathematical details",
        "math",
    ),
    "applications": ("practical applications", "applications", "use cases"),
    "visualization": ("visualization", "circuit visualization", "circuit diagram"),
}

_ALIAS_TO_SECTION = {
    alias: section for section, aliases in _SECTION_ALIASES.items() for alias in aliases
}


def _is_setext_underline(line: Optional[str]) -> bool:
    return bool(line and re.match(r"^\s*[=-]{3,}\s*$", line))


def _normalize_heading(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text)
    text = text.strip("*_` ")
    text = text.rstrip(":").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _match_section(heading_text: str) -> Optional[str]:
    normalized = _normalize_heading(heading_text)
    return _ALIAS_TO_SECTION.get(normalized)


def _parse_heading(line: str, next_line: Optional[str]) -> tuple[Optional[str], Optional[str], bool]:
    """
    Parse a markdown heading-like line.
    Returns: (section_key, inline_content, consume_next_line)
    """
    match = re.match(r"^\s{0,3}#{1,6}\s*(.+?)\s*#*\s*$", line)
    if match:
        section = _match_section(match.group(1))
        if section:
            return section, None, _is_setext_underline(next_line)

    match = re.match(r"^\s*\*\*\s*(.+?)\s*\*\*\s*$", line)
    if match:
        section = _match_section(match.group(1))
        if section:
            return section, None, _is_setext_underline(next_line)

    match = re.match(r"^\s*([A-Za-z][A-Za-z0-9\s\-/]+?)\s*:\s*(.*)\s*$", line)
    if match:
        section = _match_section(match.group(1))
        if section:
            inline = (match.group(2) or "").strip() or None
            return section, inline, False

    if _is_setext_underline(next_line):
        section = _match_section(line)
        if section:
            return section, None, True

    return None, None, False


def parse_structured_sections(text: str) -> Dict[str, Optional[str]]:
    """
    Extract section bodies from markdown-formatted explanation text.
    """
    sections = {key: [] for key in _SECTION_ALIASES}
    lines = (text or "").splitlines()

    current_section: Optional[str] = None
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        next_line = lines[idx + 1] if idx + 1 < len(lines) else None
        section, inline_content, consume_next = _parse_heading(line, next_line)

        if section:
            current_section = section
            if inline_content:
                sections[current_section].append(inline_content)
            idx += 2 if consume_next else 1
            continue

        if current_section is not None:
            sections[current_section].append(line)

        idx += 1

    result: Dict[str, Optional[str]] = {}
    for key, collected_lines in sections.items():
        content = "\n".join(collected_lines).strip()
        result[key] = content or None

    return result


def parse_explanation(text: str) -> Dict[str, str]:
    """
    Parse required explanation fields with robust markdown heading handling.
    """
    parsed = parse_structured_sections(text)
    return {
        "overview": parsed.get("overview") or "N/A",
        "gate_breakdown": parsed.get("gate_breakdown") or "N/A",
        "quantum_concepts": parsed.get("quantum_concepts") or "N/A",
        "applications": parsed.get("applications") or "N/A",
    }


def extract_mathematics(text: str) -> Optional[str]:
    """
    Extract a dedicated mathematics section if present.
    """
    parsed = parse_structured_sections(text)
    if parsed.get("mathematics"):
        return parsed["mathematics"]

    match = re.search(
        r"(?ims)^\s*Mathematically\s*[:,\-]?\s*(.+?)(?=\n\s*\n|\Z)",
        text or "",
    )
    return match.group(1).strip() if match else None
