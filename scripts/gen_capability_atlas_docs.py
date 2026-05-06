"""Generate capability-atlas documentation from structural claims."""

from __future__ import annotations

import html
import importlib
from pathlib import Path

from tests import test_structure as _structure_claims  # noqa: F401

atlas = importlib.import_module("tests.structure_atlas")

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
    source: atlas._AtlasRegionSource,
    points: tuple[tuple[float, float], ...],
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
    stroke, fill = _status_style(str(atlas._atlas_source_status(source)))
    if len(points) == 2:
        (x0, y0), (x1, y1) = points
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
    if len(points) >= 3:
        svg_points = [
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
            for x, y in points
        ]
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in svg_points)
        return (
            f'<polygon points="{point_text}" fill="{fill}" fill-opacity="0.22" '
            f'stroke="{stroke}" stroke-width="2" stroke-opacity="0.62"/>'
        )
    raise AssertionError(f"unsupported atlas geometry with {len(points)} points")


def _shape_name(source: atlas._AtlasRegionSource) -> str:
    return str(atlas._atlas_source_label(source))


def _shape_condition(
    source: atlas._AtlasRegionSource,
    predicates: tuple[object, ...],
) -> str:
    if isinstance(source, atlas.InvalidCellRule):
        return source.reason
    if isinstance(source, atlas.CoverageRegion):
        return f"{_coverage_region_name(source)} coverage region"
    return "; ".join(atlas._predicate_label(predicate) for predicate in predicates)


def _matched_regions(descriptor: atlas.ParameterDescriptor) -> str:
    schema = atlas._atlas_schema_for_descriptor(descriptor)
    matched = [
        region.name for region in schema.derived_regions if region.contains(descriptor)
    ]
    return ", ".join(matched) if matched else "none"


def _coverage_region_name(region: atlas.CoverageRegion) -> str:
    return region.owner.__name__


def _axis_partition_summary(axis: atlas.ParameterAxis) -> str:
    view = atlas._atlas_axis_view(axis)
    cells = "; ".join(_axis_cell_label(cell) for cell in view.cells)
    scale = "log-eligible" if view.use_log_scale else "partition"
    return f"{scale}: {cells}"


def _axis_cell_label(cell: atlas.ParameterBin | atlas.NumericInterval) -> str:
    if isinstance(cell, atlas.ParameterBin):
        values = ", ".join(str(value) for value in sorted(cell.values, key=str))
        return f"{cell.label} {{{values}}}"
    lower = "[" if cell.include_lower else "("
    upper = "]" if cell.include_upper else ")"
    return f"{cell.label} {lower}{cell.lower}, {cell.upper}{upper}"


def _predicate_list(predicates: tuple[atlas.StructuredPredicate, ...]) -> str:
    return "; ".join(atlas._predicate_label(predicate) for predicate in predicates)


def _derived_region_summary(region: atlas.DerivedParameterRegion) -> str:
    alternatives = tuple(
        _predicate_list(alternative) for alternative in region.alternatives
    )
    return " OR ".join(alternatives)


def _schema_descriptor_count(schema: atlas.ParameterSpaceSchema) -> int:
    return sum(
        1
        for descriptor in atlas._capability_atlas_descriptors()
        if atlas._atlas_schema_for_descriptor(descriptor) == schema
    )


def _descriptor_evidence_lines() -> list[str]:
    lines: list[str] = []
    for index, descriptor in enumerate(atlas._capability_atlas_descriptors(), start=1):
        schema = atlas._atlas_schema_for_descriptor(descriptor)
        regions = atlas._atlas_regions_for_schema(schema)
        status = schema.cell_status(descriptor, regions)
        lines.extend(
            [
                f"{index}. `{schema.name}` descriptor: `{status}`",
                f"   - Matched regions: `{_matched_regions(descriptor)}`",
                f"   - Coordinates: {_descriptor_coordinate_summary(descriptor)}",
            ]
        )
    return lines


def _descriptor_coordinate_summary(descriptor: atlas.ParameterDescriptor) -> str:
    schema = atlas._atlas_schema_for_descriptor(descriptor)
    return ", ".join(
        "`"
        f"{atlas._field_label(axis.field)}="
        f"{atlas._descriptor_value(descriptor, axis.field)}"
        "`"
        for axis in schema.axes
    )


def _schema_hierarchy_lines(schema: atlas.ParameterSpaceSchema) -> list[str]:
    regions = atlas._atlas_regions_for_schema(schema)
    uncovered = atlas._capability_atlas_uncovered_cells(schema, regions)
    lines = [
        f"### {schema.name}",
        "",
        f"- Evidence descriptors: `{_schema_descriptor_count(schema)}`",
        f"- Owned regions: `{len(regions)}`",
        f"- Computed uncovered cells: `{len(uncovered)}`",
        "- Primitive axes:",
    ]
    lines.extend(
        f"  - `{atlas._field_label(axis.field)}`: {_axis_partition_summary(axis)}"
        for axis in schema.axes
    )
    if schema.auxiliary_fields:
        lines.append("- Auxiliary coordinates:")
        lines.extend(
            f"  - `{atlas._field_label(field)}`"
            for field in sorted(schema.auxiliary_fields, key=atlas._field_label)
        )
    if schema.derived_regions:
        lines.append("- Derived regions:")
        lines.extend(
            f"  - `{region.name}`: {_derived_region_summary(region)}"
            for region in schema.derived_regions
        )
    if schema.invalid_cells:
        lines.append("- Invalid subregions:")
        lines.extend(
            f"  - `{rule.name}`: {rule.reason}" for rule in schema.invalid_cells
        )
    if regions:
        owners = ", ".join(
            f"`{_coverage_region_name(region)}`"
            for region in sorted(regions, key=_coverage_region_name)
        )
        lines.append(f"- Ownership projections: {owners}")
    lines.append("")
    return lines


def _render_parameter_space_hierarchy() -> str:
    lines = [
        "## Parameter Space Hierarchy",
        "",
        "Each entry below is generated from `ParameterSpaceSchema`.  The hierarchy",
        "is schema → primitive coordinates → derived regions → current",
        "implementation ownership and uncovered cells.  Human labels such as",
        "`linear_system` or `nonlinear_root` appear only as derived regions over",
        "primitive coordinates; they are not independent sources of truth.",
        "",
    ]
    for schema in atlas._capability_atlas_schemas():
        lines.extend(_schema_hierarchy_lines(schema))
    return "\n".join(lines)


def _shape_evidence_labels(
    schema: atlas.ParameterSpaceSchema,
    source: atlas._AtlasRegionSource,
    descriptors: tuple[atlas.ParameterDescriptor, ...],
) -> str:
    labels: list[str] = []
    if isinstance(source, atlas.InvalidCellRule):
        for index, descriptor in enumerate(descriptors, start=1):
            if source.matches(descriptor):
                labels.append(str(index))
    else:
        for index, descriptor in enumerate(descriptors, start=1):
            if source.contains(descriptor):
                labels.append(str(index))
    return ", ".join(labels) if labels else "none"


def _projection_region_targets(
    schema: atlas.ParameterSpaceSchema,
    descriptor: atlas.ParameterDescriptor,
    region_targets_by_source: dict[tuple[object, ...], list[str]],
) -> str:
    regions = atlas._atlas_regions_for_schema(schema)
    targets: list[str] = []
    for source in atlas._schema_atlas_regions(schema, regions):
        if isinstance(source, atlas.InvalidCellRule):
            contains = source.matches(descriptor)
        else:
            contains = source.contains(descriptor)
        if contains:
            targets.extend(
                region_targets_by_source.get(atlas._atlas_source_key(source), ())
            )
    return " ".join(dict.fromkeys(targets))


def _projection_title(
    descriptor: atlas.ParameterDescriptor,
    status: str,
    index: int,
) -> str:
    schema = atlas._atlas_schema_for_descriptor(descriptor)
    return (
        f"{schema.name} descriptor {index}; status {status}; "
        f"regions {_matched_regions(descriptor)}"
    )


def _render_region_card(
    *,
    index: int,
    source: atlas._AtlasRegionSource,
    predicates: tuple[object, ...],
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
                f"{index}. {html.escape(_shape_name(source))}</span>"
            ),
            (
                '<span class="cf-atlas-card-meta">'
                f"{html.escape(str(atlas._atlas_source_status(source)))} from "
                f"{html.escape(str(atlas._atlas_source_label(source)))}</span>"
            ),
            (
                '<span class="cf-atlas-card-meta">test descriptors inside: '
                f"{html.escape(evidence_labels)}</span>"
            ),
            (
                '<span class="cf-atlas-card-meta">'
                f"{html.escape(_shape_condition(source, predicates))}</span>"
            ),
            "</button>",
        ]
    )


def _render_interactive_plot(group: atlas._AtlasDescriptorGroup) -> str:
    selected = group
    schema = atlas._atlas_group_schema(group)
    region_shapes = atlas._projected_region_shapes(group)
    x_axis = atlas._atlas_group_x_axis(group)
    y_axis = atlas._atlas_group_y_axis(group)
    schema = atlas._atlas_group_schema(group)
    axis_by_field = {axis.field: axis for axis in schema.axes}
    x_axis_object = axis_by_field[x_axis]
    y_axis_object = axis_by_field[y_axis]
    x_axis_label = atlas._field_label(x_axis)
    y_axis_label = atlas._field_label(y_axis)
    x_min, x_max = atlas._atlas_group_x_range(group)
    y_min, y_max = atlas._atlas_group_y_range(group)
    title = atlas._atlas_group_title(group)
    caption = atlas._atlas_group_caption(group)
    width, height = 1280, 820
    left, right, top, bottom = 76, 36, 42, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    atlas_id = _dom_id("atlas", atlas._atlas_group_filename(group))

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="{atlas_id}-title {atlas_id}-desc">',
        f'<title id="{atlas_id}-title">{html.escape(title)}</title>',
        f'<desc id="{atlas_id}-desc">{html.escape(caption)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#475467" stroke-width="1.4"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="#475467" stroke-width="1.4"/>',
        _svg_text(
            left + plot_w / 2, height - 24, x_axis_label, size=13, anchor="middle"
        ),
        (
            f'<text x="22" y="{top + plot_h / 2:.1f}" font-size="13" '
            'font-family="Inter, Arial, sans-serif" fill="#101828" '
            f'text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2:.1f})">'
            f"{html.escape(y_axis_label)}</text>"
        ),
    ]
    x_view = atlas._atlas_axis_view(x_axis_object)
    y_view = atlas._atlas_axis_view(y_axis_object)
    for tick in range(len(x_view.cells) + 1):
        x_value = float(tick)
        x = left + _plot_coordinate(x_value, axis_min=x_min, axis_max=x_max) * plot_w
        svg_parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                f'y2="{top + plot_h}" stroke="#eaecf0" stroke-width="1"/>',
            ]
        )
    for index, cell in enumerate(x_view.cells):
        x_value = index + 0.5
        x = left + _plot_coordinate(x_value, axis_min=x_min, axis_max=x_max) * plot_w
        svg_parts.append(
            _svg_text(
                x,
                top + plot_h + 20,
                _axis_cell_label(cell),
                size=10,
                anchor="middle",
            )
        )
    for tick in range(len(y_view.cells) + 1):
        y_value = float(tick)
        y = (
            top
            + plot_h
            - _plot_coordinate(y_value, axis_min=y_min, axis_max=y_max) * plot_h
        )
        svg_parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" '
            f'y2="{y:.1f}" stroke="#eaecf0" stroke-width="1"/>'
        )
    for index, cell in enumerate(y_view.cells):
        y_value = index + 0.5
        y = (
            top
            + plot_h
            - _plot_coordinate(y_value, axis_min=y_min, axis_max=y_max) * plot_h
        )
        svg_parts.append(
            _svg_text(left - 10, y + 4, _axis_cell_label(cell), size=10, anchor="end")
        )

    cards: list[str] = []
    region_targets_by_source: dict[tuple[object, ...], list[str]] = {}
    for index, (source, predicates, points) in enumerate(region_shapes, start=1):
        target_id = _dom_id(atlas_id, "region", index, _shape_name(source))
        region_targets_by_source.setdefault(atlas._atlas_source_key(source), []).append(
            target_id
        )
        shape = _render_region_shape(
            source,
            points,
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
                    f"<title>"
                    f"{html.escape(_shape_name(source))}: "
                    f"{html.escape(_shape_condition(source, predicates))}"
                    f"</title>"
                ),
                shape,
                "</g>",
            ]
        )
        cards.append(
            _render_region_card(
                index=index,
                source=source,
                predicates=predicates,
                target_id=target_id,
                evidence_labels=_shape_evidence_labels(schema, source, selected),
            )
        )

    seen_points: dict[tuple[float, float], int] = {}
    for index, descriptor in enumerate(selected, start=1):
        schema = atlas._atlas_schema_for_descriptor(descriptor)
        regions = atlas._atlas_regions_for_schema(schema)
        status = schema.cell_status(descriptor, regions)
        stroke, _fill = _status_style(status)
        x_value = atlas._atlas_axis_coordinate(
            x_axis_object, descriptor.coordinate(x_axis)
        )
        y_value = atlas._atlas_axis_coordinate(
            y_axis_object, descriptor.coordinate(y_axis)
        )
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
            schema, descriptor, region_targets_by_source
        )
        title = html.escape(_projection_title(descriptor, status, index))
        svg_parts.extend(
            [
                (
                    '<g class="cf-atlas-point-overlay" '
                    f'data-cf-atlas-member="{_html_attr(region_targets)}">'
                ),
                f"<title>{title}</title>",
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
            f"<strong>{html.escape(title)}</strong>",
            f"<div>{html.escape(caption)}</div>",
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
        "structural test registry.  Projection axes are selected as quotient maps",
        "that preserve schema-cell source identities: invalid-rule predicates,",
        "coverage owner plus predicates, and uncovered schema-cell coordinates.",
        "Each plot names the axes shown directly, the coordinates fixed outside the",
        "projection, and the higher-dimensional axes that are only summarized.",
        "Region geometry is drawn first; concrete",
        "structural descriptor fixtures are overlaid as",
        "numbered evidence points.  Plot region lists are autodiscovered from",
        "`InvalidCellRule`, `CoverageRegion`, and computed uncovered schema cells.",
        "",
        "Status legend:",
        "",
        "- `invalid`: the descriptor violates a schema validity rule.",
        "- `owned`: a coverage region owns the descriptor.",
        "- `uncovered`: the descriptor is valid but no coverage region owns it.",
        "",
        _ATLAS_STYLE_AND_SCRIPT,
        "",
        _render_parameter_space_hierarchy(),
        "",
        "## Projection Plots",
        "",
    ]
    for group in atlas._capability_atlas_descriptor_groups():
        fixed = atlas._atlas_fixed_axes(group)
        marginalized = atlas._atlas_marginalized_axes(group)
        x_axis = atlas._atlas_group_x_axis(group)
        y_axis = atlas._atlas_group_y_axis(group)
        schema = atlas._atlas_group_schema(group)
        axis_by_field = {axis.field: axis for axis in schema.axes}
        lines.extend(
            [
                f"### {atlas._atlas_group_title(group)}",
                "",
                _render_interactive_plot(group),
                "",
                f"Shown axes: `{atlas._field_label(x_axis)}` and "
                f"`{atlas._field_label(y_axis)}`.",
                "Fixed axes: "
                + ", ".join(f"`{atlas._field_label(field)}`" for field in fixed)
                + ".",
                "Marginalized axes: "
                + ", ".join(f"`{atlas._field_label(field)}`" for field in marginalized)
                + ".",
                "Visual axis partitions:",
                f"- `{atlas._field_label(x_axis)}`: "
                f"{_axis_partition_summary(axis_by_field[x_axis])}",
                f"- `{atlas._field_label(y_axis)}`: "
                f"{_axis_partition_summary(axis_by_field[y_axis])}",
                "",
            ]
        )

    lines.extend(["", "## Coverage Regions", ""])
    regions = {
        _coverage_region_name(region): region
        for region in atlas._capability_atlas_coverage_regions()
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

    lines.extend(["## Computed Gaps", ""])
    uncovered = tuple(
        (
            schema,
            atlas._capability_atlas_uncovered_cells(
                schema, atlas._atlas_regions_for_schema(schema)
            ),
        )
        for schema in atlas._capability_atlas_schemas()
    )
    uncovered = tuple((schema, cells) for schema, cells in uncovered if cells)
    if uncovered:
        for schema, cells in uncovered:
            lines.extend(
                [
                    f"### {schema.name}",
                    "",
                    f"- Uncovered schema cells: `{len(cells)}`",
                    "- Cell basis: schema axis bins and intervals",
                    "- Excluded cells: invalid cells and owned coverage regions",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "No valid descriptor evidence is currently outside declared",
                "ownership regions.",
                "",
            ]
        )

    lines.extend(["## Descriptor Evidence Overlay", ""])
    lines.extend(_descriptor_evidence_lines())
    lines.append("")
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
