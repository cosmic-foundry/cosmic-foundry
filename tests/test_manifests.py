"""Tests for cosmic_foundry.manifests platform infrastructure."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from cosmic_foundry.manifests import (
    BOT_UA,
    STANDARD_UA,
    HTTPClient,
    Provenance,
    ValidationAdapter,
    generate_bibliography,
    load_schema,
    validate_manifest,
)

# ---------------------------------------------------------------------------
# HTTPClient
# ---------------------------------------------------------------------------


def test_http_client_default_ua() -> None:
    client = HTTPClient()
    assert client.user_agent == BOT_UA
    assert client.respect_robots is True


def test_http_client_research_identity() -> None:
    client = HTTPClient(user_agent=STANDARD_UA, respect_robots=False)
    assert client.user_agent == STANDARD_UA
    assert client.respect_robots is False


def test_bot_ua_references_cosmic_foundry() -> None:
    assert "cosmic-foundry" in BOT_UA.lower()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_sha256_stable(tmp_path: Path) -> None:
    f = tmp_path / "artifact.csv"
    f.write_text("a,b\n1,2\n")
    h1 = Provenance.sha256(f)
    h2 = Provenance.sha256(f)
    assert h1 == h2
    assert h1.algorithm == "sha256"
    assert len(h1.value) == 64


def test_provenance_sha256_changes_with_content(tmp_path: Path) -> None:
    f = tmp_path / "artifact.csv"
    f.write_text("a,b\n1,2\n")
    h1 = Provenance.sha256(f)
    f.write_text("a,b\n1,3\n")
    h2 = Provenance.sha256(f)
    assert h1.value != h2.value


def test_provenance_write_sidecar(tmp_path: Path) -> None:
    artifact = tmp_path / "out.csv"
    artifact.write_text("x\n1\n")
    h = Provenance.sha256(artifact)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    prov = Provenance(
        validation_set_id="test-set",
        built_at=now,
        adapter_script="adapters/test.py",
        adapter_version="abc123",
        upstream_catalog="test-catalog",
        upstream_release="v1.0",
        upstream_retrieved_at=now,
        artifact_path="artifacts/out.csv",
        artifact_hash=h,
        artifact_row_count=1,
    )

    sidecar = tmp_path / "out.provenance.yaml"
    prov.write_sidecar(sidecar)

    with open(sidecar) as f:
        doc = yaml.safe_load(f)

    assert doc["validation_set_id"] == "test-set"
    assert doc["adapter"]["version"] == "abc123"
    assert doc["upstream"]["catalog"] == "test-catalog"
    assert doc["artifact"]["row_count"] == 1
    assert doc["artifact"]["content_hash"]["algorithm"] == "sha256"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_load_schema_catalog() -> None:
    schema = load_schema("catalog")
    assert schema["title"] == "Upstream data catalog manifest"


def test_load_schema_validation_set() -> None:
    schema = load_schema("validation-set")
    assert schema["title"] == "Validation set manifest"


def test_load_schema_artifact_provenance() -> None:
    schema = load_schema("artifact-provenance")
    assert schema["title"] == "Validation artifact provenance sidecar"


def test_load_schema_unknown_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_schema("does-not-exist")


def test_validate_manifest_valid_catalog() -> None:
    manifest = {
        "id": "test-catalog",
        "title": "Test Catalog",
        "kind": "archive",
        "domain": ["test-domain"],
        "role": "A test catalog.",
        "homepage": "https://example.com",
        "status": "planned",
        "provenance": {
            "authority": "Test Authority",
            "accessed": "2026-01-01",
            "access": {"status": "public"},
        },
        "data_products": ["table"],
        "known_caveats": [],
        "next_actions": ["build adapter"],
    }
    validate_manifest(manifest, "catalog")  # should not raise


def test_validate_manifest_missing_required_field() -> None:
    import jsonschema

    manifest = {
        "id": "test-catalog",
        "title": "Test Catalog",
        # missing kind, domain, role, homepage, status, provenance, etc.
    }
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest(manifest, "catalog")


def test_validate_manifest_catalog_requires_release_when_available() -> None:
    import jsonschema

    manifest = {
        "id": "test-catalog",
        "title": "Test Catalog",
        "kind": "archive",
        "domain": ["test-domain"],
        "role": "A test catalog.",
        "homepage": "https://example.com",
        "status": "available",  # requires release field
        "provenance": {
            "authority": "Test Authority",
            "accessed": "2026-01-01",
            "access": {"status": "public"},
        },
        "data_products": ["table"],
        "known_caveats": [],
        "next_actions": [],
        # missing release
    }
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest(manifest, "catalog")


# ---------------------------------------------------------------------------
# Bibliography
# ---------------------------------------------------------------------------


def test_generate_bibliography_no_artifacts(tmp_path: Path) -> None:
    result = generate_bibliography(tmp_path, tmp_path)
    assert "No artifacts found" in result


def test_generate_bibliography_with_sidecar(tmp_path: Path) -> None:
    # Write a minimal catalog manifest
    catalog_dir = tmp_path / "observables" / "test" / "catalogs"
    catalog_dir.mkdir(parents=True)
    catalog = {
        "id": "test-cat",
        "title": "Test Catalog",
        "provenance": {
            "authority": "Test Org",
            "access": {"status": "public"},
        },
        "references": [
            {"label": "Smith+2020", "url": "https://example.com/smith2020"},
        ],
    }
    (catalog_dir / "test-cat.yaml").write_text(yaml.safe_dump(catalog))

    # Write a provenance sidecar
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    sidecar = {
        "upstream": {"catalog": "test-cat"},
    }
    (artifact_dir / "out.provenance.yaml").write_text(yaml.safe_dump(sidecar))

    result = generate_bibliography(artifact_dir, tmp_path / "observables")
    assert "Test Catalog" in result
    assert "Test Org" in result
    assert "Smith+2020" in result


# ---------------------------------------------------------------------------
# ValidationAdapter ABC
# ---------------------------------------------------------------------------


def test_validation_adapter_concrete_subclass() -> None:
    class MyAdapter(ValidationAdapter):
        catalog_id = "test"
        validation_set_id = "test-set"

        def __call__(self, artifact_dir: Path) -> Provenance:  # type: ignore[empty-body]
            ...

    adapter = MyAdapter()
    assert isinstance(adapter, ValidationAdapter)


def test_validation_adapter_incomplete_raises() -> None:
    import pytest

    class Incomplete(ValidationAdapter):
        catalog_id = "test"
        # __call__() not implemented

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]
