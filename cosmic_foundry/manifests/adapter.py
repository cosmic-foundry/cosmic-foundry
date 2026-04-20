from abc import abstractmethod
from pathlib import Path

from cosmic_foundry.manifests.provenance import Provenance
from cosmic_foundry.theory.function import Function


class ValidationAdapter(Function[Path, Provenance]):
    """Abstract base for application-repo observational adapters.

    Function:
        domain   — artifact_dir: Path — directory where the normalized
                   artifact and provenance sidecar will be written
        codomain — Provenance — record of how the artifact was produced
        operator — (artifact_dir) → Provenance; fetches upstream data,
                   normalizes it, writes the artifact, and returns provenance

    Each application repo subclasses ValidationAdapter for each upstream
    catalog it ingests from, providing catalog_id, validation_set_id, and
    a concrete execute().
    """

    catalog_id: str
    validation_set_id: str

    @abstractmethod
    def __call__(self, artifact_dir: Path) -> Provenance:
        """Produce the normalized artifact and return its provenance."""
