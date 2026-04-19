import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore

from cosmic_foundry.manifests.record import Record


@dataclass(frozen=True)
class Hash(Record):
    algorithm: str
    value: str

    def as_dict(self) -> dict[str, str]:
        return {"algorithm": self.algorithm, "value": self.value}


@dataclass(frozen=True)
class Provenance(Record):
    """Immutable record of how one validation artifact was produced.

    Matches the structure of artifact-provenance.schema.json so that
    write_sidecar() produces a schema-valid document.
    """

    validation_set_id: str
    built_at: datetime
    adapter_script: str
    adapter_version: str
    upstream_catalog: str
    upstream_release: str
    upstream_retrieved_at: datetime
    artifact_path: str
    artifact_hash: Hash
    artifact_row_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "validation_set_id": self.validation_set_id,
            "built_at": self.built_at.isoformat(),
            "adapter": {
                "script": self.adapter_script,
                "version": self.adapter_version,
            },
            "upstream": {
                "catalog": self.upstream_catalog,
                "release": self.upstream_release,
                "retrieved_at": self.upstream_retrieved_at.isoformat(),
            },
            "artifact": {
                "path": self.artifact_path,
                "content_hash": self.artifact_hash.as_dict(),
                "row_count": self.artifact_row_count,
            },
        }

    def write_sidecar(self, path: Path) -> None:
        """Write provenance as a YAML sidecar next to the artifact.

        Sink:
            domain — (self: Provenance, path: Path)
            effect — YAML file written at path containing the full
                     provenance record; schema-valid against
                     artifact-provenance.schema.json
        """
        with open(path, "w") as f:
            yaml.safe_dump(self.as_dict(), f, sort_keys=False)

    @staticmethod
    def now() -> datetime:
        return datetime.now(tz=UTC)

    @staticmethod
    def sha256(path: Path) -> Hash:
        """Compute SHA-256 of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return Hash(algorithm="sha256", value=h.hexdigest())
