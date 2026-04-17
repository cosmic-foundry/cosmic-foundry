from pathlib import Path

import yaml  # type: ignore


def generate_bibliography(artifact_dir: Path, catalog_root: Path) -> str:
    """Generate a Markdown bibliography from artifact provenance sidecars.

    Scans artifact_dir for *.provenance.yaml files, looks up the upstream
    catalog manifest for each, and produces a Markdown bibliography.

    artifact_dir: directory tree containing provenance sidecar files.
    catalog_root: root directory under which catalog manifests live, searched
                  recursively for catalogs/<catalog_id>.yaml.
    """
    used_catalogs: set[str] = set()

    for sidecar in artifact_dir.rglob("*.provenance.yaml"):
        with open(sidecar) as f:
            prov = yaml.safe_load(f)
        used_catalogs.add(prov["upstream"]["catalog"])

    if not used_catalogs:
        msg = "No artifacts found. Build artifacts to populate attribution."
        return f"# Bibliography\n\n{msg}\n"

    lines = [
        "# Bibliography",
        "",
        "Data from the following upstream sources.",
        "",
    ]

    for catalog_id in sorted(used_catalogs):
        catalog_path = _find_catalog(catalog_id, catalog_root)
        with open(catalog_path) as f:
            cat = yaml.safe_load(f)

        lines.append(f"## {cat['title']}")
        lines.append(f"**Authority:** {cat['provenance']['authority']}")
        lines.append(f"**Status:** {cat['provenance']['access']['status']}")
        lines.append("")

        if cat.get("references"):
            lines.append("### References")
            for ref in cat["references"]:
                line = f"- [{ref['label']}]({ref['url']})"
                if ref.get("bibcode"):
                    bib = ref["bibcode"]
                    line += f" ([{bib}](https://ui.adsabs.harvard.edu/abs/{bib}))"
                lines.append(line)
            lines.append("")

    return "\n".join(lines)


def _find_catalog(catalog_id: str, catalog_root: Path) -> Path:
    for p in catalog_root.rglob(f"catalogs/{catalog_id}.yaml"):
        return p
    raise FileNotFoundError(
        f"Catalog manifest not found: {catalog_id} under {catalog_root}"
    )
