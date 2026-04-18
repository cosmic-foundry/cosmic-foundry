from abc import abstractmethod
from pathlib import Path

from cosmic_foundry.kernels import Map
from cosmic_foundry.manifests.provenance import Provenance


class ValidationAdapter(Map):
    """Abstract base for application-repo observational adapters.

    Map:
        domain   — artifact_dir: Path — directory where the normalized
                   artifact and provenance sidecar will be written
        codomain — Provenance — record of how the artifact was produced
        operator — execute(artifact_dir) → Provenance; fetches upstream
                   data, normalizes it, writes the artifact, and returns
                   its provenance

    Each application repo subclasses ValidationAdapter for each upstream
    catalog it ingests from, providing catalog_id, validation_set_id, and
    a concrete execute().
    """

    catalog_id: str
    validation_set_id: str

    @abstractmethod
    def execute(self, artifact_dir: Path) -> Provenance:
        """Produce the normalized artifact and return its provenance.

        The adapter fetches upstream data if needed, applies selection and
        normalization, writes the artifact to artifact_dir, writes a
        provenance sidecar alongside it, and returns the Provenance record.
        """
