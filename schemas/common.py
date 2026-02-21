"""
Shared request schemas.
"""
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ClientContext(BaseModel):
    """
    Client metadata used for version-aware retrieval and generation.
    """

    client_type: str = Field(
        "website",
        description="Origin of request: website, vscode_extension, or api",
    )
    client_version: Optional[str] = Field(
        None,
        description="Version of web app or custom client",
    )
    extension_installed: Optional[bool] = Field(
        None,
        description="Whether VS Code extension is installed/enabled",
    )
    extension_version: Optional[str] = Field(
        None,
        description="VS Code extension version",
    )
    python_version: Optional[str] = Field(
        None,
        description="Runtime Python version available on client side",
    )
    framework_version: Optional[str] = Field(
        None,
        description="Installed version for selected framework, if known",
    )
    installed_packages: Dict[str, str] = Field(
        default_factory=dict,
        description="Detected package versions, e.g. {'qiskit': '1.2.4'}",
    )
