import json
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore

_SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"


def load_schema(name: str) -> dict[str, Any]:
    """Load a base schema by name (without the .schema.json suffix)."""
    path = _SCHEMAS_DIR / f"{name}.schema.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Base schema not found: {name} (looked in {_SCHEMAS_DIR})"
        )
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def validate_manifest(manifest: dict[str, Any], schema_name: str) -> None:
    """Validate a manifest dict against a named base schema.

    Raises jsonschema.ValidationError if the manifest is invalid.
    schema_name: one of 'catalog', 'validation-set', 'artifact-provenance'.
    """
    schema = load_schema(schema_name)
    jsonschema.validate(instance=manifest, schema=schema)


def validate_manifest_file(path: Path, schema_name: str) -> None:
    """Load a YAML manifest file and validate it against a named base schema."""
    import yaml  # type: ignore

    with open(path) as f:
        manifest = yaml.safe_load(f)
    validate_manifest(manifest, schema_name)
