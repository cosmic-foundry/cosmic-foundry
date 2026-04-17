from pathlib import Path
from typing import Protocol, runtime_checkable

from cosmic_foundry.manifests.provenance import Provenance


@runtime_checkable
class ValidationAdapter(Protocol):
    """Interface that every application-repo observational adapter must implement.

    An adapter is responsible for fetching upstream data, normalizing it to a
    reproducible artifact, and recording the provenance of that transformation.
    Application repos (stellar-foundry, cosmological-foundry, etc.) implement
    this protocol for each upstream catalog they ingest from.
    """

    catalog_id: str
    validation_set_id: str

    def build(self, artifact_dir: Path) -> Provenance:
        """Produce the normalized artifact and return its provenance.

        The adapter fetches upstream data if needed, applies selection and
        normalization, writes the artifact to artifact_dir, writes a
        provenance sidecar alongside it, and returns the Provenance record.
        """
        ...
