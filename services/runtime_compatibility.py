"""
Runtime compatibility utilities for client-aware generation and retrieval.
"""
import hashlib
import re
from typing import Dict, Optional, Tuple

from core.config import settings
from schemas.common import ClientContext


def _normalize_client_type(value: str) -> str:
    candidate = (value or "").strip().lower()
    aliases = {
        "web": "website",
        "website": "website",
        "site": "website",
        "vscode": "vscode_extension",
        "vscode_extension": "vscode_extension",
        "extension": "vscode_extension",
        "api": "api",
    }
    return aliases.get(candidate, "api")


def _major_from_version(value: str) -> Optional[int]:
    if not value:
        return None
    match = re.search(r"(\d+)", value)
    if not match:
        return None
    return int(match.group(1))


def _major_bounds(spec: str) -> Optional[Tuple[int, int]]:
    """
    Parse lightweight bounds from a version spec like ">=1.2,<2.0".
    Returns (min_major_inclusive, max_major_exclusive) when possible.
    """
    if not spec:
        return None

    lower = re.search(r">=\s*(\d+)", spec)
    upper = re.search(r"<\s*(\d+)", spec)
    if not lower or not upper:
        return None
    return int(lower.group(1)), int(upper.group(1))


def _runtime_matrix() -> Dict[str, Dict]:
    return {
        "qiskit": {
            "python": settings.SUPPORTED_PYTHON_VERSION,
            "packages": {
                "qiskit": settings.SUPPORTED_QISKIT_VERSION,
                "qiskit-aer": settings.SUPPORTED_QISKIT_AER_VERSION,
            },
        },
        "pennylane": {
            "python": settings.SUPPORTED_PYTHON_VERSION,
            "packages": {
                "pennylane": settings.SUPPORTED_PENNYLANE_VERSION,
            },
        },
        "cirq": {
            "python": settings.SUPPORTED_PYTHON_VERSION,
            "packages": {
                "cirq": settings.SUPPORTED_CIRQ_VERSION,
            },
        },
        "torchquantum": {
            "python": settings.SUPPORTED_PYTHON_VERSION,
            "packages": {
                "torchquantum": settings.SUPPORTED_TORCHQUANTUM_VERSION,
                "torch": settings.SUPPORTED_TORCH_VERSION,
            },
        },
    }


def build_runtime_bundle(framework: str, client_context: Optional[ClientContext]) -> Dict:
    safe_framework = (framework or "").strip().lower()
    ctx = client_context or ClientContext()
    client_type = _normalize_client_type(ctx.client_type)

    runtime = _runtime_matrix().get(
        safe_framework,
        {
            "python": settings.SUPPORTED_PYTHON_VERSION,
            "packages": {},
        },
    )

    installed_packages = {k.lower(): v for k, v in (ctx.installed_packages or {}).items() if v}
    if ctx.framework_version and safe_framework and safe_framework not in installed_packages:
        installed_packages[safe_framework] = ctx.framework_version

    recommended_packages = runtime.get("packages", {})
    conflicts = []
    for package, version in installed_packages.items():
        recommended_spec = recommended_packages.get(package)
        if not recommended_spec:
            continue
        bounds = _major_bounds(recommended_spec)
        major = _major_from_version(version)
        if bounds and major is not None:
            if major < bounds[0] or major >= bounds[1]:
                conflicts.append(
                    f"{package}={version} is outside supported range {recommended_spec}"
                )

    lines = [
        f"Client type: {client_type}",
        f"Preferred supported Python runtime: {runtime.get('python', settings.SUPPORTED_PYTHON_VERSION)}",
    ]

    if ctx.python_version:
        lines.append(f"Client reported Python version: {ctx.python_version}")
    if ctx.client_version:
        lines.append(f"Client version: {ctx.client_version}")
    if ctx.extension_installed is not None:
        lines.append(f"VS Code extension installed: {ctx.extension_installed}")
    if ctx.extension_version:
        lines.append(f"VS Code extension version: {ctx.extension_version}")

    if installed_packages:
        lines.append("Detected installed packages:")
        for package, version in sorted(installed_packages.items()):
            lines.append(f"- {package}: {version}")

    if recommended_packages:
        lines.append("Supported package versions to target:")
        for package, version_spec in sorted(recommended_packages.items()):
            lines.append(f"- {package}: {version_spec}")

    if conflicts:
        lines.append("Potential version conflicts detected:")
        for conflict in conflicts:
            lines.append(f"- {conflict}")

    if client_type == "vscode_extension":
        lines.append(
            "Generate code compatible with installed versions and avoid deprecated APIs for those versions."
        )
    else:
        lines.append(
            "Always provide runtime recommendations with Python and package versions for reliable execution."
        )

    compatibility_context = "\n".join(lines)

    rag_query_suffix = (
        f"Compatibility target for {safe_framework}: "
        f"python={runtime.get('python', settings.SUPPORTED_PYTHON_VERSION)}; "
        f"packages={recommended_packages}; "
        f"installed={installed_packages}; "
        f"client_type={client_type}"
    )

    cache_material = "|".join(
        [
            safe_framework,
            client_type,
            ctx.python_version or "",
            ctx.client_version or "",
            ctx.extension_version or "",
            ",".join(f"{k}={v}" for k, v in sorted(installed_packages.items())),
            ",".join(f"{k}:{v}" for k, v in sorted(recommended_packages.items())),
        ]
    )

    return {
        "client_type": client_type,
        "compatibility_context": compatibility_context,
        "rag_query_suffix": rag_query_suffix,
        "runtime_recommendations": {
            "python": runtime.get("python", settings.SUPPORTED_PYTHON_VERSION),
            "packages": recommended_packages,
        },
        "version_conflicts": conflicts,
        "client_context": {
            "client_type": client_type,
            "client_version": ctx.client_version,
            "extension_installed": ctx.extension_installed,
            "extension_version": ctx.extension_version,
            "python_version": ctx.python_version,
            "framework_version": ctx.framework_version,
            "installed_packages": installed_packages,
        },
        "cache_fingerprint": hashlib.sha1(cache_material.encode("utf-8")).hexdigest(),
    }
