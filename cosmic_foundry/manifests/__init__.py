"""Platform manifest and specification infrastructure.

Provides the shared machinery that application repositories use to define,
validate, fetch, and track observational validation products and simulation
specifications.

Install the optional dependencies required by this module:
    pip install cosmic-foundry[observational]
"""

from cosmic_foundry.manifests.adapter import ValidationAdapter
from cosmic_foundry.manifests.bibliography import generate_bibliography
from cosmic_foundry.manifests.http_client import BOT_UA, STANDARD_UA, HTTPClient
from cosmic_foundry.manifests.provenance import Hash, Provenance
from cosmic_foundry.manifests.record import Record
from cosmic_foundry.manifests.validate import (
    load_schema,
    validate_manifest,
    validate_manifest_file,
)

__all__ = [
    "BOT_UA",
    "Record",
    "STANDARD_UA",
    "HTTPClient",
    "ValidationAdapter",
    "Hash",
    "Provenance",
    "generate_bibliography",
    "load_schema",
    "validate_manifest",
    "validate_manifest_file",
]
