"""Shared ownership witness for selector calculation claims."""

from __future__ import annotations

from dataclasses import dataclass

from cosmic_foundry.computation.algorithm_capabilities import (
    AlgorithmCapability,
    AlgorithmRequest,
    CoverageRegion,
    ParameterDescriptor,
    ParameterSpaceSchema,
)


@dataclass(frozen=True)
class SelectionOwnership:
    """Unique implementation ownership for one descriptor in one region family."""

    descriptor: ParameterDescriptor
    regions: tuple[CoverageRegion, ...]
    schema: ParameterSpaceSchema | None = None

    @classmethod
    def from_request(
        cls,
        request: AlgorithmRequest,
        capabilities: tuple[AlgorithmCapability, ...],
    ) -> SelectionOwnership:
        assert request.descriptor is not None
        return cls(
            request.descriptor,
            tuple(
                region
                for capability in capabilities
                if capability.supports(request)
                for region in capability.coverage_regions
            ),
        )

    @property
    def owner(self) -> type:
        owners = tuple(
            region.owner for region in self.regions if region.contains(self.descriptor)
        )
        assert len(owners) == 1
        return owners[0]

    def assert_owned_cell(self) -> None:
        assert self.schema is not None
        self.schema.validate_descriptor(self.descriptor)
        assert self.schema.cell_status(self.descriptor, self.regions) == "owned"
        assert self.owner
