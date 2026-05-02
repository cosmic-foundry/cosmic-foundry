"""Generate capability-atlas documentation from structural claims."""

from __future__ import annotations

import html
from pathlib import Path

from tests import test_structure as atlas

_PROJECT_ROOT = Path(__file__).parent.parent
_DOCS_OUT = _PROJECT_ROOT / "docs"


_ATLAS_STYLE_AND_SCRIPT = """
<style>
.cf-atlas-interactive {
  border: 1px solid #d0d5dd;
  border-radius: 8px;
  margin: 1rem 0 1.5rem;
  overflow: hidden;
}
.cf-atlas-interactive-header {
  background: #f9fafb;
  border-bottom: 1px solid #eaecf0;
  padding: .75rem 1rem;
}
.cf-atlas-interactive-body {
  display: grid;
  grid-template-columns: 1fr;
  gap: .75rem;
  padding: .75rem;
}
.cf-atlas-interactive svg {
  width: 100%;
  height: auto;
}
.cf-atlas-region-overlay,
.cf-atlas-point-overlay {
  opacity: 0;
  pointer-events: none;
  transition: opacity .12s ease, filter .12s ease;
}
.cf-atlas-region-overlay.is-hovered,
.cf-atlas-region-overlay.is-pinned {
  filter: drop-shadow(0 0 4px rgba(16, 24, 40, .22));
  opacity: 1;
}
.cf-atlas-point-overlay.is-hovered,
.cf-atlas-point-overlay.is-pinned {
  filter: drop-shadow(0 0 3px rgba(16, 24, 40, .20));
  opacity: 1;
}
.cf-atlas-card-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr));
  gap: .28rem;
  max-height: 18rem;
  overflow: auto;
}
.cf-atlas-card {
  background: #fff;
  border: 1px solid #d0d5dd;
  border-radius: 4px;
  color: #101828;
  cursor: pointer;
  font: inherit;
  padding: .34rem .45rem;
  text-align: left;
}
.cf-atlas-card.is-hovered,
.cf-atlas-card.is-pinned,
.cf-atlas-card:hover,
.cf-atlas-card:focus {
  border-color: #475467;
  box-shadow: 0 0 0 2px rgba(71, 84, 103, .15);
}
.cf-atlas-card.is-pinned .cf-atlas-card-title::after {
  content: " pinned";
  color: #475467;
  font-size: .72rem;
  font-weight: 500;
}
.cf-atlas-card-title {
  display: block;
  font-size: .82rem;
  font-weight: 650;
  line-height: 1.15;
}
.cf-atlas-card-meta {
  color: #475467;
  display: block;
  font-size: .66rem;
  line-height: 1.16;
  margin-top: .12rem;
}
@media (max-width: 700px) {
  .cf-atlas-card-list {
    grid-template-columns: 1fr;
  }
}
</style>
<script>
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("[data-cf-atlas-target]").forEach(function (control) {
    var atlas = control.closest("[data-cf-atlas]");
    if (!atlas) return;
    var target = control.getAttribute("data-cf-atlas-target");
    var targets = function () {
      var selector = '[data-cf-atlas-member~="' + target + '"], ' +
        '[data-cf-atlas-target="' + target + '"]';
      return atlas.querySelectorAll(selector);
    };
    var setHovered = function (enabled) {
      targets().forEach(function (element) {
        element.classList.toggle("is-hovered", enabled);
      });
    };
    var togglePinned = function () {
      var pinned = !control.classList.contains("is-pinned");
      targets().forEach(function (element) {
        element.classList.toggle("is-pinned", pinned);
        if (element.hasAttribute("aria-pressed")) {
          element.setAttribute("aria-pressed", pinned ? "true" : "false");
        }
      });
    };
    control.addEventListener("mouseenter", function () { setHovered(true); });
    control.addEventListener("focus", function () { setHovered(true); });
    control.addEventListener("mouseleave", function () { setHovered(false); });
    control.addEventListener("blur", function () { setHovered(false); });
    control.addEventListener("click", togglePinned);
  });
});
</script>
""".strip()


def _html_attr(value: str) -> str:
    return html.escape(value, quote=True)


def _dom_id(*parts: object) -> str:
    raw = "-".join(str(part).lower() for part in parts)
    cleaned = "".join(character if character.isalnum() else "-" for character in raw)
    return "-".join(fragment for fragment in cleaned.split("-") if fragment)


def _status_style(status: str) -> tuple[str, str]:
    if status == "invalid":
        return "#b42318", "#fee4e2"
    if status == "owned":
        return "#027a48", "#d1fadf"
    return "#475467", "#f2f4f7"


def _plot_coordinate(value: float, *, axis_min: float, axis_max: float) -> float:
    if axis_max == axis_min:
        return 0.5
    return (value - axis_min) / (axis_max - axis_min)


def _svg_plot_point(
    x_value: float,
    y_value: float,
    *,
    left: float,
    top: float,
    plot_w: float,
    plot_h: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[float, float]:
    return (
        left + _plot_coordinate(x_value, axis_min=x_min, axis_max=x_max) * plot_w,
        top
        + plot_h
        - _plot_coordinate(y_value, axis_min=y_min, axis_max=y_max) * plot_h,
    )


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 13,
    anchor: str = "start",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        f'font-family="Inter, Arial, sans-serif" fill="#101828">'
        f"{html.escape(text)}</text>"
    )


def _render_region_shape(
    region: atlas._AtlasRegionShape,
    *,
    left: float,
    top: float,
    plot_w: float,
    plot_h: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    stroke, fill = _status_style(region.status)
    if region.geometry == "line":
        (x0, y0), (x1, y1) = region.points
        px0, py0 = _svg_plot_point(
            x0,
            y0,
            left=left,
            top=top,
            plot_w=plot_w,
            plot_h=plot_h,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        px1, py1 = _svg_plot_point(
            x1,
            y1,
            left=left,
            top=top,
            plot_w=plot_w,
            plot_h=plot_h,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        return (
            f'<line x1="{px0:.1f}" y1="{py0:.1f}" x2="{px1:.1f}" y2="{py1:.1f}" '
            f'stroke="{stroke}" stroke-width="7" stroke-linecap="round" '
            'stroke-opacity="0.72"/>'
        )
    if region.geometry == "polygon":
        points = [
            _svg_plot_point(
                x,
                y,
                left=left,
                top=top,
                plot_w=plot_w,
                plot_h=plot_h,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
            for x, y in region.points
        ]
        svg_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        return (
            f'<polygon points="{svg_points}" fill="{fill}" fill-opacity="0.22" '
            f'stroke="{stroke}" stroke-width="2" stroke-opacity="0.62"/>'
        )
    if region.geometry == "rectangle":
        (x0, y0), (x1, y1) = region.points
        px0, py0 = _svg_plot_point(
            min(x0, x1),
            max(y0, y1),
            left=left,
            top=top,
            plot_w=plot_w,
            plot_h=plot_h,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        px1, py1 = _svg_plot_point(
            max(x0, x1),
            min(y0, y1),
            left=left,
            top=top,
            plot_w=plot_w,
            plot_h=plot_h,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        return (
            f'<rect x="{px0:.1f}" y="{py0:.1f}" width="{px1 - px0:.1f}" '
            f'height="{py1 - py0:.1f}" fill="{fill}" fill-opacity="0.20" '
            f'stroke="{stroke}" stroke-width="2" stroke-dasharray="8 5" '
            'stroke-opacity="0.62"/>'
        )
    raise AssertionError(f"unsupported atlas geometry {region.geometry!r}")


def _matched_regions(projection: atlas._AtlasProjection) -> str:
    matched = [
        region.name
        for region in projection.schema.derived_regions
        if region.contains(projection.descriptor)
    ]
    return ", ".join(matched) if matched else "none"


def _coverage_region_name(region: atlas.CoverageRegion) -> str:
    return region.owner.__name__


def _shape_evidence_labels(
    schema: atlas.ParameterSpaceSchema,
    region: atlas._AtlasRegionShape,
    projections: tuple[atlas._AtlasProjection, ...],
) -> str:
    labels: list[str] = []
    if isinstance(region.source, atlas.InvalidCellRule):
        for index, projection in enumerate(projections, start=1):
            if region.source.matches(projection.descriptor):
                labels.append(str(index))
    elif isinstance(region.source, atlas.DerivedParameterRegion):
        for index, projection in enumerate(projections, start=1):
            if region.source.contains(projection.descriptor):
                labels.append(str(index))
    else:
        for index, projection in enumerate(projections, start=1):
            if region.source.contains(projection.descriptor):
                labels.append(str(index))
    return ", ".join(labels) if labels else "none"


def _projection_region_targets(
    schema: atlas.ParameterSpaceSchema,
    projection: atlas._AtlasProjection,
    region_targets_by_source: dict[int, list[str]],
) -> str:
    targets: list[str] = []
    for region in schema.derived_regions:
        if region.contains(projection.descriptor):
            targets.extend(region_targets_by_source.get(id(region), ()))
    for rule in schema.invalid_cells:
        if rule.matches(projection.descriptor):
            targets.extend(region_targets_by_source.get(id(rule), ()))
    for region in projection.regions:
        if region.contains(projection.descriptor):
            targets.extend(region_targets_by_source.get(id(region), ()))
    return " ".join(dict.fromkeys(targets))


def _render_region_card(
    *,
    index: int,
    region: atlas._AtlasRegionShape,
    target_id: str,
    evidence_labels: str,
) -> str:
    return "\n".join(
        [
            (
                f'<button type="button" class="cf-atlas-card" '
                'aria-pressed="false" '
                f'data-cf-atlas-target="{_html_attr(target_id)}">'
            ),
            (
                '<span class="cf-atlas-card-title">'
                f"{index}. {html.escape(region.name)}</span>"
            ),
            (
                '<span class="cf-atlas-card-meta">'
                f"{html.escape(region.status)} from "
                f"{html.escape(region.source_name)}</span>"
            ),
            (
                '<span class="cf-atlas-card-meta">test descriptors inside: '
                f"{html.escape(evidence_labels)}</span>"
            ),
            (
                '<span class="cf-atlas-card-meta">'
                f"{html.escape(region.condition)}</span>"
            ),
            "</button>",
        ]
    )


def _render_interactive_plot(spec: atlas._AtlasPlotSpec) -> str:
    selected = spec.projections
    schema = spec.schema
    region_shapes = atlas._projected_region_shapes(spec)
    x_min, x_max = spec.x_range
    y_min, y_max = spec.y_range
    width, height = 1280, 820
    left, right, top, bottom = 76, 36, 42, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    atlas_id = _dom_id("atlas", spec.filename)

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="{atlas_id}-title {atlas_id}-desc">',
        f'<title id="{atlas_id}-title">{html.escape(spec.title)}</title>',
        f'<desc id="{atlas_id}-desc">{html.escape(spec.caption)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#475467" stroke-width="1.4"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="#475467" stroke-width="1.4"/>',
        _svg_text(
            left + plot_w / 2, height - 24, spec.x_axis, size=13, anchor="middle"
        ),
        (
            f'<text x="22" y="{top + plot_h / 2:.1f}" font-size="13" '
            'font-family="Inter, Arial, sans-serif" fill="#101828" '
            f'text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2:.1f})">'
            f"{html.escape(spec.y_axis)}</text>"
        ),
    ]
    for tick in range(5):
        x_frac = tick / 4
        y_frac = tick / 4
        x = left + x_frac * plot_w
        y = top + plot_h - y_frac * plot_h
        x_value = x_min + x_frac * (x_max - x_min)
        y_value = y_min + y_frac * (y_max - y_min)
        svg_parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                f'y2="{top + plot_h}" stroke="#eaecf0" stroke-width="1"/>',
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" '
                f'y2="{y:.1f}" stroke="#eaecf0" stroke-width="1"/>',
                _svg_text(
                    x, top + plot_h + 20, f"{x_value:.1f}", size=10, anchor="middle"
                ),
                _svg_text(left - 10, y + 4, f"{y_value:.1f}", size=10, anchor="end"),
            ]
        )

    cards: list[str] = []
    region_targets_by_source: dict[int, list[str]] = {}
    for index, region in enumerate(region_shapes, start=1):
        target_id = _dom_id(atlas_id, "region", index, region.name)
        region_targets_by_source.setdefault(id(region.source), []).append(target_id)
        shape = _render_region_shape(
            region,
            left=left,
            top=top,
            plot_w=plot_w,
            plot_h=plot_h,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        svg_parts.extend(
            [
                (
                    f'<g class="cf-atlas-region-overlay" '
                    f'data-cf-atlas-member="{_html_attr(target_id)}">'
                ),
                (
                    f"<title>{html.escape(region.name)}: "
                    f"{html.escape(region.condition)}</title>"
                ),
                shape,
                "</g>",
            ]
        )
        cards.append(
            _render_region_card(
                index=index,
                region=region,
                target_id=target_id,
                evidence_labels=_shape_evidence_labels(schema, region, selected),
            )
        )

    seen_points: dict[tuple[float, float], int] = {}
    for index, projection in enumerate(selected, start=1):
        status = projection.schema.cell_status(
            projection.descriptor, projection.regions
        )
        stroke, _fill = _status_style(status)
        x_value = float(projection.descriptor.coordinate(spec.x_axis).value)
        y_value = float(projection.descriptor.coordinate(spec.y_axis).value)
        x, y = _svg_plot_point(
            x_value,
            y_value,
            left=left,
            top=top,
            plot_w=plot_w,
            plot_h=plot_h,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        duplicate_key = (x_value, y_value)
        duplicate_index = seen_points.get(duplicate_key, 0)
        seen_points[duplicate_key] = duplicate_index + 1
        offsets = ((0.0, 0.0), (14.0, 0.0), (-14.0, 0.0), (0.0, -14.0), (0.0, 14.0))
        dx, dy = offsets[duplicate_index % len(offsets)]
        region_targets = _projection_region_targets(
            schema, projection, region_targets_by_source
        )
        svg_parts.extend(
            [
                (
                    '<g class="cf-atlas-point-overlay" '
                    f'data-cf-atlas-member="{_html_attr(region_targets)}">'
                ),
                (
                    f"<title>{html.escape(projection.title)}; status {status}; "
                    f"regions {html.escape(_matched_regions(projection))}</title>"
                ),
                f'<circle cx="{x + dx:.1f}" cy="{y + dy:.1f}" r="8" fill="#ffffff" '
                f'stroke="{stroke}" stroke-width="2.4"/>',
                _svg_text(
                    x + dx,
                    y + dy + 4,
                    str(index),
                    size=10,
                    anchor="middle",
                    weight="700",
                ),
                "</g>",
            ]
        )

    svg_parts.append("</svg>")
    return "\n".join(
        [
            (
                '<div class="cf-atlas-interactive" '
                f'data-cf-atlas="{_html_attr(atlas_id)}">'
            ),
            '<div class="cf-atlas-interactive-header">',
            f"<strong>{html.escape(spec.title)}</strong>",
            f"<div>{html.escape(spec.caption)}</div>",
            "</div>",
            '<div class="cf-atlas-interactive-body">',
            '<div class="cf-atlas-diagram">',
            "\n".join(svg_parts),
            "</div>",
            '<div class="cf-atlas-card-list" aria-label="Projected regions">',
            "\n".join(cards),
            "</div>",
            "</div>",
            "</div>",
        ]
    )


def render_capability_atlas() -> str:
    """Return the generated capability atlas Markdown page."""
    lines = [
        "# Capability Coverage Atlas",
        "",
        "<!-- Generated from structural atlas data; do not edit by hand. -->",
        "",
        "This page is a projection of the parameter-space schemas used by the",
        "structural test registry.  Each plot names the axes shown directly, the",
        "coordinates fixed outside the projection, and the higher-dimensional axes",
        "that are only summarized.  Region geometry is drawn first; concrete",
        "descriptor fixtures from `tests/test_structure.py` are overlaid as",
        "numbered evidence points.  Plot region lists are autodiscovered from",
        "`DerivedParameterRegion`, `InvalidCellRule`, and `CoverageRegion`",
        "declarations reachable from each schema projection.",
        "",
        "Status legend:",
        "",
        "- `invalid`: the descriptor violates a schema validity rule.",
        "- `owned`: a coverage region owns the descriptor.",
        "- `uncovered`: the descriptor is valid but no coverage region owns it.",
        "",
        _ATLAS_STYLE_AND_SCRIPT,
        "",
        "## Projection Plots",
        "",
    ]
    for spec in atlas._capability_atlas_plot_specs():
        fixed = sorted(
            {
                field
                for projection in spec.projections
                for field in projection.fixed_axes
            }
        )
        marginalized = sorted(
            {
                field
                for projection in spec.projections
                for field in projection.marginalized_axes
            }
        )
        lines.extend(
            [
                f"### {spec.title}",
                "",
                _render_interactive_plot(spec),
                "",
                f"Shown axes: `{spec.x_axis}` and `{spec.y_axis}`.",
                "Fixed axes: " + ", ".join(f"`{field}`" for field in fixed) + ".",
                "Marginalized axes: "
                + ", ".join(f"`{field}`" for field in marginalized)
                + ".",
                "",
            ]
        )

    lines.extend(["", "## Coverage Regions", ""])
    regions = {
        _coverage_region_name(region): region
        for projection in atlas._capability_atlas_projections()
        for region in projection.regions
    }
    if regions:
        for region in sorted(regions.values(), key=_coverage_region_name):
            lines.extend(
                [
                    f"### {_coverage_region_name(region)}",
                    "",
                    f"- Owner: `{_coverage_region_name(region)}`",
                    "- Predicates:",
                ]
            )
            lines.extend(
                f"  - `{atlas._predicate_label(predicate)}`"
                for predicate in region.predicates
            )
            lines.append("")
    else:
        lines.extend(
            [
                "No solver or decomposition ownership regions are declared in this",
                "atlas yet.",
                "",
            ]
        )

    lines.extend(["## Known Gaps", ""])
    for gap in atlas._capability_atlas_gaps():
        lines.extend(
            [
                f"### {gap.name}",
                "",
                f"- Region: `{gap.region}`",
                f"- Selected owner: {gap.selected_owner}",
                "- Descriptor:",
            ]
        )
        lines.extend(f"  - `{entry}`" for entry in gap.descriptor)
        lines.append("- Existing partial owners:")
        lines.extend(f"  - {owner}" for owner in gap.partial_owners)
        lines.extend(
            [
                "- Required capability before this region is owned: "
                f"{gap.required_capability}",
                "",
            ]
        )

    lines.extend(
        [
            "## Numerical Evidence Overlay",
            "",
            "No owned solver or decomposition coverage region has numerical evidence",
            "metadata in this atlas yet.  Until ownership regions exist, numerical",
            "correctness, convergence, performance, and regression claims remain",
            "outside this projection rather than being attached to cells.",
            "",
        ]
    )
    return "\n".join(lines)


def generate(out_root: Path = _DOCS_OUT) -> None:
    """Write the capability atlas page and remove stale generated plot assets."""
    plots_out = out_root / "capability_atlas_plots"
    if plots_out.exists():
        for stale_plot in plots_out.glob("*.svg"):
            stale_plot.unlink()
    (out_root / "capability_atlas.md").write_text(render_capability_atlas())


if __name__ == "__main__":
    generate()
