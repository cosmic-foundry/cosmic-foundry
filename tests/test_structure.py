"""Structural invariant claims for the cosmic_foundry codebase.

Each claim encodes one structural property of the codebase. Adding a new
claim requires only appending to _CLAIMS; the single parametric test covers
all entries.

Claim types:
  _AbcInstantiationClaim    — ABCs cannot be directly instantiated
  _HierarchyClaim           — cosmic_foundry subclass relations are correct
  _ModuleAllClaim           — every public class in a module appears in __all__
  _IterativeSolverJitClaim  — iterative solver runs on a small assembled LinearOperator
  _MaterializationGateClaim — converged() raises MaterializationError on .get()
  _FactorizationJitClaim    — Factorization.factorize/solve run on declared Tensors
  _GenericBasesClaim        — no subclass leaves a generic base's TypeVars unbound
  _ManifoldIsolationClaim   — Manifold and IndexedSet hierarchies are disjoint
  _ImportBoundaryClaim      — theory/ and geometry/ import only approved packages
  _ArchitectureOwnershipClaim — package exports and capability ownership are explicit
  _LinearSolverCoverageLocalityClaim — owned solver coverage lives in
                                      implementation classes
  _TestAxisConventionClaim  — module tests use correctness/convergence/performance
  _NoTopLevelDefaultBackendMutationClaim — tests do not mutate Tensor backend at import
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import sys
import types
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, NewType, TypeAlias, get_args, get_origin, get_type_hints

import pytest

from cosmic_foundry.computation.algorithm_capabilities import (
    AffineComparisonPredicate,
    ComparisonPredicate,
    CoverageRegion,
    DecompositionField,
    DerivedParameterRegion,
    DescriptorCoordinate,
    DescriptorField,
    EvidencePredicate,
    InvalidCellRule,
    LinearSolverField,
    MembershipPredicate,
    NumericInterval,
    ParameterAxis,
    ParameterBin,
    ParameterDescriptor,
    ParameterSpaceSchema,
    coverage_regions_are_disjoint,
    decomposition_parameter_schema,
    linear_operator_descriptor_from_assembled_operator,
    linear_solver_parameter_schema,
    predicate_sets_are_disjoint,
    solve_relation_parameter_schema,
)
from cosmic_foundry.computation.backends.python_backend import PythonBackend
from cosmic_foundry.computation.decompositions.factorization import Factorization
from cosmic_foundry.computation.solvers.capabilities import (
    linear_solver_coverage_regions,
    select_linear_solver_for_descriptor,
)
from cosmic_foundry.computation.solvers.iterative_solver import IterativeSolver
from cosmic_foundry.computation.tensor import MaterializationError, Tensor
from cosmic_foundry.theory.continuous.manifold import Manifold
from cosmic_foundry.theory.foundation.indexed_set import IndexedSet
from tests.claims import Claim

_PROJECT_ROOT = Path(__file__).parent.parent
_PACKAGE_ROOT = _PROJECT_ROOT / "cosmic_foundry"
_AtlasText = NewType("_AtlasText", str)
_AtlasDescriptorField: TypeAlias = LinearSolverField | DecompositionField
_AtlasRegionSource: TypeAlias = (
    DerivedParameterRegion | InvalidCellRule | CoverageRegion
)


class _AtlasGeometryKind(Enum):
    """Symbolic projected-geometry kind for rendered atlas regions."""

    LINE = auto()
    POLYGON = auto()
    RECTANGLE = auto()


_PACKAGES = [
    "cosmic_foundry.theory.foundation",
    "cosmic_foundry.theory.continuous",
    "cosmic_foundry.theory.discrete",
    "cosmic_foundry.geometry",
    "cosmic_foundry.computation",
    "cosmic_foundry.computation.decompositions",
    "cosmic_foundry.computation.solvers",
    "cosmic_foundry.computation.autotuning",
    "cosmic_foundry.computation.time_integrators",
]
_PURE_PACKAGES = [
    _PACKAGE_ROOT / "theory" / "foundation",
    _PACKAGE_ROOT / "theory" / "continuous",
    _PACKAGE_ROOT / "theory" / "discrete",
    _PACKAGE_ROOT / "geometry",
]
_STDLIB = sys.stdlib_module_names
_SYMBOLIC_PACKAGES = {"sympy"}
_JIT_N = 4
_JIT_BACKEND = PythonBackend()


class _JitLinearOperator:
    """Tiny tridiagonal SPD operator for iterative-solver structure checks."""

    def __init__(self, n: int) -> None:
        self._n = n

    def apply(self, u: Tensor) -> Tensor:
        values = []
        for i in range(self._n):
            value = 2.0 * float(u[i])
            if i > 0:
                value -= float(u[i - 1])
            if i < self._n - 1:
                value -= float(u[i + 1])
            values.append(value)
        return Tensor(values, backend=u.backend)

    def diagonal(self, backend: Any) -> Tensor:
        return Tensor([2.0] * self._n, backend=backend)

    def row_abs_sums(self, backend: Any) -> Tensor:
        return Tensor(
            [3.0 if i in (0, self._n - 1) else 4.0 for i in range(self._n)],
            backend=backend,
        )


_JIT_OP = _JitLinearOperator(_JIT_N)
_JIT_B = Tensor([1.0] * _JIT_N, backend=_JIT_BACKEND)


class _MatrixLinearOperator:
    """Small dense operator used by descriptor-construction structure claims."""

    def __init__(self, matrix: tuple[tuple[float, ...], ...]) -> None:
        self._matrix = matrix
        self._n = len(matrix)

    def apply(self, u: Tensor) -> Tensor:
        return Tensor(
            [
                sum(self._matrix[row][col] * float(u[col]) for col in range(self._n))
                for row in range(self._n)
            ],
            backend=u.backend,
        )


class _DeclaredLinearOperator:
    """LinearOperator returning declared Tensors; used by _MaterializationGateClaim.

    All methods return unallocated (declared) Tensors so that any solver
    method that calls .get() inside converged() is caught by MaterializationError.
    """

    def __init__(self, n: int) -> None:
        self._n = n

    def apply(self, u: Tensor) -> Tensor:
        return Tensor.declare(*u.shape)

    def diagonal(self, backend: Any) -> Tensor:
        return Tensor.declare(self._n)

    def row_abs_sums(self, backend: Any) -> Tensor:
        return Tensor.declare(self._n)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _discover_modules() -> list[tuple[str, types.ModuleType]]:
    result = []
    for pkg in _PACKAGES:
        pkg_path = _PROJECT_ROOT / pkg.replace(".", "/")
        for path in sorted(pkg_path.glob("*.py")):
            if path.stem == "__init__":
                continue
            mod_path = f"{pkg}.{path.stem}"
            try:
                mod = importlib.import_module(mod_path)
            except ImportError:
                continue
            result.append((mod_path, mod))
    return result


def _discover_abcs(
    modules: list[tuple[str, types.ModuleType]],
) -> list[type]:
    seen: set[type] = set()
    abcs: list[type] = []
    for mod_path, mod in modules:
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                obj.__module__ == mod_path
                and getattr(obj, "__abstractmethods__", None)
                and obj not in seen
            ):
                seen.add(obj)
                abcs.append(obj)
    return abcs


def _discover_hierarchy_pairs(
    abcs: list[type],
) -> list[tuple[type, type]]:
    seen: set[tuple[type, type]] = set()
    pairs: list[tuple[type, type]] = []
    for cls in abcs:
        for base in inspect.getmro(cls)[1:]:
            if base is object:
                continue
            if not getattr(base, "__module__", "").startswith("cosmic_foundry"):
                continue
            pair = (cls, base)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def _discover_concrete_iterative_solvers(
    modules: list[tuple[str, types.ModuleType]],
) -> list[type]:
    seen: set[type] = set()
    result: list[type] = []
    for _, mod in modules:
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                obj not in seen
                and issubclass(obj, IterativeSolver)
                and not getattr(obj, "__abstractmethods__", None)
                and obj is not IterativeSolver
            ):
                seen.add(obj)
                result.append(obj)
    return result


def _discover_matrix_free_iterative_solvers(
    modules: list[tuple[str, types.ModuleType]],
) -> list[type]:
    """Iterative solvers whose init_state does not assemble a dense matrix.

    Assembling solvers (those with an _assemble method) build a dense matrix
    during init_state, which requires materialized tensors and cannot run on
    declared (unallocated) Tensor placeholders.
    """
    return [
        cls
        for cls in _discover_concrete_iterative_solvers(modules)
        if not hasattr(cls, "_assemble")
    ]


def _discover_concrete_factorizations(
    modules: list[tuple[str, types.ModuleType]],
) -> list[type]:
    seen: set[type] = set()
    result: list[type] = []
    for _, mod in modules:
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                obj not in seen
                and issubclass(obj, Factorization)
                and not getattr(obj, "__abstractmethods__", None)
                and obj is not Factorization
            ):
                seen.add(obj)
                result.append(obj)
    return result


def _all_local_classes() -> list[tuple[str, str, type]]:
    import cosmic_foundry

    results = []
    for _, modname, _ in pkgutil.walk_packages(
        cosmic_foundry.__path__, prefix="cosmic_foundry."
    ):
        try:
            module = importlib.import_module(modname)
        except ImportError:
            continue
        for clsname, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == modname:
                results.append((modname, clsname, cls))
    return results


def _top_level(module: str) -> str:
    return module.split(".")[0]


def _third_party_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = _top_level(alias.name)
                if (
                    top not in _STDLIB
                    and top != "cosmic_foundry"
                    and top not in _SYMBOLIC_PACKAGES
                ):
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            top = _top_level(node.module)
            if (
                top not in _STDLIB
                and top != "cosmic_foundry"
                and top not in _SYMBOLIC_PACKAGES
            ):
                violations.append(node.module)
    return violations


# ---------------------------------------------------------------------------
# Claim classes
# ---------------------------------------------------------------------------


class _AbcInstantiationClaim(Claim[None]):
    def __init__(self, cls: type) -> None:
        self._cls = cls

    @property
    def description(self) -> str:
        return f"abc_not_instantiable/{self._cls.__qualname__}"

    def check(self, _calibration: None) -> None:
        with pytest.raises(TypeError):
            self._cls()


class _HierarchyClaim(Claim[None]):
    def __init__(self, child: type, parent: type) -> None:
        self._child = child
        self._parent = parent

    @property
    def description(self) -> str:
        return f"hierarchy/{self._child.__qualname__}->{self._parent.__qualname__}"

    def check(self, _calibration: None) -> None:
        assert issubclass(self._child, self._parent)


class _ModuleAllClaim(Claim[None]):
    def __init__(self, mod_path: str, mod: types.ModuleType) -> None:
        self._mod_path = mod_path
        self._mod = mod

    @property
    def description(self) -> str:
        return f"module_all/{self._mod_path}"

    def check(self, _calibration: None) -> None:
        exported = set(getattr(self._mod, "__all__", []))
        defined = {
            name
            for name, obj in inspect.getmembers(self._mod, inspect.isclass)
            if obj.__module__ == self._mod_path and not name.startswith("_")
        }
        missing = defined - exported
        assert not missing, f"defined but not in __all__: {missing}"


class _IterativeSolverJitClaim(Claim[None]):
    def __init__(self, cls: type) -> None:
        self._cls = cls

    @property
    def description(self) -> str:
        return f"iterative_jit/{self._cls.__qualname__}"

    def check(self, _calibration: None) -> None:
        solver: Any = self._cls()
        state = solver.init_state(_JIT_OP, _JIT_B)
        new_state = solver.step(_JIT_OP, state)
        assert type(new_state) is type(state)
        converged = solver.converged(state)
        assert isinstance(converged, Tensor)
        assert converged.shape == ()


class _MaterializationGateClaim(Claim[None]):
    def __init__(self, cls: type) -> None:
        self._cls = cls

    @property
    def description(self) -> str:
        return f"materialization_gate/{self._cls.__qualname__}"

    def check(self, _calibration: None) -> None:
        op = _DeclaredLinearOperator(_JIT_N)
        b = Tensor.declare(_JIT_N)
        solver: Any = self._cls()
        state = solver.init_state(op, b)
        converged = solver.converged(state)
        with pytest.raises(MaterializationError):
            converged.get()


class _FactorizationJitClaim(Claim[None]):
    def __init__(self, cls: type) -> None:
        self._cls = cls

    @property
    def description(self) -> str:
        return f"factorization_jit/{self._cls.__qualname__}"

    def check(self, _calibration: None) -> None:
        n = _JIT_N
        a = Tensor.declare(n, n)
        rhs = Tensor.declare(n)
        factored = self._cls().factorize(a)
        factored.solve(rhs)


class _GenericBasesClaim(Claim[None]):
    """Claim: every subclass of a generic class fully binds the TypeVar parameters."""

    @property
    def description(self) -> str:
        return "generic_bases/all_parameterized"

    def check(self, _calibration: None) -> None:
        violations: list[str] = []
        for modname, clsname, cls in _all_local_classes():
            if getattr(cls, "__parameters__", ()):
                continue  # cls itself is generic — TypeVars intentionally free
            for base in getattr(cls, "__orig_bases__", ()):
                if getattr(base, "__parameters__", ()):
                    violations.append(
                        f"{modname}.{clsname}: base '{base}' has unbound TypeVars"
                    )
                    break
        if violations:
            raise AssertionError(
                "Classes with unbound TypeVars in generic bases:\n"
                + "\n".join(f"  {v}" for v in violations)
            )


class _ManifoldIsolationClaim(Claim[None]):
    """Claim: the Manifold and IndexedSet hierarchies are disjoint."""

    @property
    def description(self) -> str:
        return "manifold/disjoint_from_indexed_set"

    def check(self, _calibration: None) -> None:
        assert not issubclass(Manifold, IndexedSet)
        assert not issubclass(IndexedSet, Manifold)


class _ImportBoundaryClaim(Claim[None]):
    """Claim: a theory/ or geometry/ source file imports no numerical packages."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def description(self) -> str:
        return f"import_boundary/{self._path.relative_to(_PACKAGE_ROOT.parent)}"

    def check(self, _calibration: None) -> None:
        violations = _third_party_imports(self._path)
        if violations:
            rel = self._path.relative_to(_PACKAGE_ROOT.parent)
            raise AssertionError(
                f"{rel} imports non-symbolic packages: {', '.join(violations)}"
            )


@dataclass(frozen=True)
class _CapabilityRequestExpectation:
    """Expected selected implementation for one capability request."""

    request: Any
    selected_implementation: str


@dataclass(frozen=True)
class _CapabilityRejectionExpectation:
    """Capability request that must not select an implementation."""

    request: Any


@dataclass(frozen=True)
class _ArchitectureOwnershipSpec:
    """Package-level ownership contract checked by _ArchitectureOwnershipClaim."""

    package: str
    public_categories: dict[str, frozenset[str]]
    forbidden_public_symbols: frozenset[str] = frozenset()
    capability_provider: str | None = None
    request_selector: str | None = None
    request_expectations: tuple[_CapabilityRequestExpectation, ...] = ()
    rejected_requests: tuple[_CapabilityRejectionExpectation, ...] = ()
    expected_class_modules: dict[str, str] | None = None
    required_name_fragments: dict[str, tuple[str, ...]] | None = None


def _resolve_dotted(name: str) -> Any:
    module_name, attr_name = name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr_name)


class _ArchitectureOwnershipClaim(Claim[None]):
    """Claim: a package's public ownership and capability map are explicit."""

    def __init__(self, spec: _ArchitectureOwnershipSpec) -> None:
        self._spec = spec

    @property
    def description(self) -> str:
        return f"architecture_ownership/{self._spec.package}"

    def check(self, _calibration: None) -> None:
        package = importlib.import_module(self._spec.package)
        exported = set(getattr(package, "__all__", []))
        categorized = set().union(*self._spec.public_categories.values())
        missing = exported - categorized
        extra = categorized - exported
        assert not missing, f"exports without ownership category: {sorted(missing)}"
        assert not extra, f"categorized symbols not exported: {sorted(extra)}"

        forbidden_exports = exported & self._spec.forbidden_public_symbols
        assert (
            not forbidden_exports
        ), f"retired public symbols exported: {sorted(forbidden_exports)}"

        local_class_names = {
            clsname
            for modname, clsname, _cls in _all_local_classes()
            if modname.startswith(f"{self._spec.package}.")
        }
        forbidden_classes = local_class_names & self._spec.forbidden_public_symbols
        assert (
            not forbidden_classes
        ), f"retired public class names still defined: {sorted(forbidden_classes)}"

        if self._spec.capability_provider is not None:
            self._check_capabilities(exported)
        self._check_class_modules(package)
        self._check_name_fragments(package)

    def _check_capabilities(self, exported: set[str]) -> None:
        provider = _resolve_dotted(self._spec.capability_provider)  # type: ignore[arg-type]
        capabilities = tuple(provider())
        implementations = [
            self._capability_implementation_name(cap) for cap in capabilities
        ]
        missing_exports = set(implementations) - exported
        assert not missing_exports, (
            "capability implementations not exported: " f"{sorted(missing_exports)}"
        )
        duplicates = sorted(
            name for name in set(implementations) if implementations.count(name) > 1
        )
        assert not duplicates, f"duplicate capability implementations: {duplicates}"

        if self._spec.request_selector is None:
            return
        selector = _resolve_dotted(self._spec.request_selector)
        for expectation in self._spec.request_expectations:
            selected = selector(expectation.request)
            selected_name = self._capability_implementation_name(selected)
            assert selected_name == expectation.selected_implementation, (
                f"{expectation.request!r} selected {selected_name}, "
                f"expected {expectation.selected_implementation}"
            )
        for expectation in self._spec.rejected_requests:
            try:
                selected = selector(expectation.request)
            except ValueError:
                continue
            selected_name = self._capability_implementation_name(selected)
            raise AssertionError(
                f"{expectation.request!r} unexpectedly selected " f"{selected_name}"
            )

    @staticmethod
    def _capability_implementation_name(capability: Any) -> str:
        if isinstance(capability, type):
            return capability.__name__
        implementation = getattr(capability, "implementation", None)
        if implementation is not None:
            return _ArchitectureOwnershipClaim._capability_implementation_name(
                implementation
            )
        owner = getattr(capability, "owner", None)
        if isinstance(owner, type):
            return owner.__name__
        return str(capability)

    def _check_class_modules(self, package: types.ModuleType) -> None:
        expected = self._spec.expected_class_modules or {}
        violations = []
        for clsname, module_stem in expected.items():
            cls = getattr(package, clsname)
            actual_stem = cls.__module__.split(".")[-1]
            if actual_stem != module_stem:
                violations.append(f"{clsname}: {actual_stem} != {module_stem}")
        assert not violations, "class/module ownership mismatch: " + "; ".join(
            violations
        )

    def _check_name_fragments(self, package: types.ModuleType) -> None:
        expected = self._spec.required_name_fragments or {}
        violations = []
        for clsname, fragments in expected.items():
            getattr(package, clsname)
            missing = [fragment for fragment in fragments if fragment not in clsname]
            if missing:
                violations.append(f"{clsname}: missing {missing}")
        assert not violations, "ownership-obscuring class names: " + "; ".join(
            violations
        )


class _ParametrizeEnforcementClaim(Claim[None]):
    """Claim: every top-level test_* function carries @pytest.mark.parametrize."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def description(self) -> str:
        return f"test_pattern/parametrize/{self._path.name}"

    def check(self, _calibration: None) -> None:
        tree = ast.parse(self._path.read_text())
        violations = []
        for node in tree.body:
            if not (
                isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            ):
                continue
            has_parametrize = any(
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Attribute)
                and d.func.attr == "parametrize"
                for d in node.decorator_list
            )
            if not has_parametrize:
                violations.append(node.name)
        if violations:
            raise AssertionError(
                f"{self._path.name}: test functions missing @pytest.mark.parametrize: "
                + ", ".join(violations)
            )


class _BodyDispatchClaim(Claim[None]):
    """Claim: every top-level test_* body is a single claim.check(...) dispatch."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def description(self) -> str:
        return f"test_pattern/body_dispatch/{self._path.name}"

    def check(self, _calibration: None) -> None:
        tree = ast.parse(self._path.read_text())
        violations = []
        for node in tree.body:
            if not (
                isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            ):
                continue
            body = node.body
            if len(body) != 1:
                violations.append(f"{node.name}: {len(body)} statements in body")
                continue
            stmt = body[0]
            if not isinstance(stmt, ast.Expr):
                violations.append(f"{node.name}: body is not an expression statement")
                continue
            call = stmt.value
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "check"
            ):
                violations.append(f"{node.name}: body does not call .check()")
                continue
            if len(call.args) != 1 or call.keywords:
                violations.append(
                    f"{node.name}: .check() does not receive exactly one calibration"
                )
        if violations:
            raise AssertionError(
                f"{self._path.name}: test functions with non-dispatch bodies: "
                + "; ".join(violations)
            )


class _TestAxisConventionClaim(Claim[None]):
    """Claim: module test functions are named by verification axis."""

    _ALLOWED_AXES = {"test_correctness", "test_convergence", "test_performance"}
    _EXEMPT_FILES = {"test_structure.py"}

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def description(self) -> str:
        return f"test_pattern/module_axes/{self._path.name}"

    def check(self, _calibration: None) -> None:
        if self._path.name in self._EXEMPT_FILES:
            return
        tree = ast.parse(self._path.read_text())
        violations = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name.startswith("test_")
            and node.name not in self._ALLOWED_AXES
        ]
        if violations:
            allowed = ", ".join(sorted(self._ALLOWED_AXES))
            raise AssertionError(
                f"{self._path.name}: test functions outside module-owned axes "
                f"({allowed}): {', '.join(violations)}"
            )


class _NoTopLevelDefaultBackendMutationClaim(Claim[None]):
    """Claim: test modules do not mutate Tensor's default backend at import time."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def description(self) -> str:
        return f"test_pattern/no_top_level_backend_mutation/{self._path.name}"

    def check(self, _calibration: None) -> None:
        tree = ast.parse(self._path.read_text())
        violations = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                if isinstance(func, ast.Name) and func.id == "set_default_backend":
                    violations.append(child.lineno)
                elif (
                    isinstance(func, ast.Attribute)
                    and func.attr == "set_default_backend"
                ):
                    violations.append(child.lineno)
        if violations:
            lines = ", ".join(str(line) for line in violations)
            raise AssertionError(
                f"{self._path.name}: top-level set_default_backend() call(s) "
                f"at line(s) {lines}; pass explicit backends or use a fixture"
            )


class _AutoDiscoveryImportClaim(Claim[None]):
    """Claim: test_structure.py imports no class that any _discover_concrete_* returns.

    Finds every function in this module whose name starts with _discover_concrete_,
    calls it, and takes the union of results.  Adding a new _discover_concrete_*
    function automatically extends the coverage without touching this claim.
    """

    @property
    def description(self) -> str:
        return "test_pattern/auto_discovery_imports"

    def check(self, _calibration: None) -> None:
        import tests.test_structure as _self

        discovered = {
            cls.__name__
            for name, fn in inspect.getmembers(_self, inspect.isfunction)
            if name.startswith("_discover_concrete_")
            for cls in fn(_MODULES)
        }
        tree = ast.parse(Path(__file__).read_text())
        violations = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("cosmic_foundry")
            ):
                continue
            for alias in node.names:
                if alias.name in discovered:
                    violations.append(f"{node.module}.{alias.name}")
        if violations:
            raise AssertionError(
                "test_structure.py imports auto-discovered classes directly "
                "(use _discover_concrete_* instead): " + ", ".join(violations)
            )


class _ParameterSpaceSchemaClaim(Claim[None]):
    """Claim: parameter-space coverage primitives fail closed structurally."""

    @property
    def description(self) -> str:
        return "algorithm_capabilities/parameter_space_schema"

    def check(self, _calibration: None) -> None:
        self._assert_axis_has_one_identity()
        self._assert_descriptor_has_no_schema_identity()
        schema = ParameterSpaceSchema(
            name="demo_solve_relation",
            axes=(
                ParameterAxis(
                    "map_linearity_defect",
                    (
                        NumericInterval(
                            "linear",
                            lower=0.0,
                            upper=1.0e-12,
                        ),
                        NumericInterval(
                            "nonlinear",
                            lower=1.0e-12,
                            include_lower=False,
                        ),
                    ),
                ),
                ParameterAxis(
                    "dim_x",
                    (NumericInterval("positive", lower=1.0),),
                ),
                ParameterAxis(
                    "dim_y",
                    (NumericInterval("positive", lower=1.0),),
                ),
                ParameterAxis(
                    "coercivity_lower_bound",
                    (
                        NumericInterval("nonpositive", upper=0.0),
                        NumericInterval(
                            "positive",
                            lower=0.0,
                            include_lower=False,
                        ),
                    ),
                ),
                ParameterAxis(
                    "condition_estimate",
                    (
                        NumericInterval("well_conditioned", upper=1.0e8),
                        NumericInterval(
                            "ill_conditioned",
                            lower=1.0e8,
                            include_lower=False,
                        ),
                    ),
                ),
                ParameterAxis(
                    "operator_representation",
                    (
                        ParameterBin(
                            "operator",
                            frozenset({"assembled_dense", "matrix_free"}),
                        ),
                    ),
                ),
            ),
            invalid_cells=(
                InvalidCellRule(
                    name="coercivity_requires_square_map",
                    predicates=(
                        AffineComparisonPredicate(
                            {"dim_x": 1.0, "dim_y": -1.0}, "!=", 0.0
                        ),
                        ComparisonPredicate("coercivity_lower_bound", ">", 0.0),
                    ),
                    reason="positive coercivity is meaningful only for square maps",
                ),
            ),
        )
        owned_region = CoverageRegion(
            owner=IterativeSolver,
            predicates=(
                ComparisonPredicate("map_linearity_defect", "<=", 1.0e-12),
                ComparisonPredicate("dim_x", "==", 4),
                ComparisonPredicate("dim_y", "==", 4),
                ComparisonPredicate("coercivity_lower_bound", ">", 0.0),
                ComparisonPredicate("condition_estimate", "<=", 1.0e8),
                MembershipPredicate(
                    "operator_representation",
                    frozenset({"assembled_dense", "matrix_free"}),
                ),
            ),
        )
        regions = (owned_region,)

        assert schema.cell_status(self._descriptor(), regions) == "owned"
        assert schema.covering_region(self._descriptor(), regions) == owned_region
        assert schema.cell_status(self._descriptor(dim_x=5), regions) == "invalid"
        assert (
            schema.cell_status(
                self._descriptor(
                    map_linearity_defect=2.0e-12,
                    coercivity_lower_bound=0.0,
                    operator_representation="matrix_free",
                ),
                regions,
            )
            == "uncovered"
        )
        unknown_condition = self._descriptor(condition_estimate=None)
        assert schema.cell_status(unknown_condition, regions) == "uncovered"
        assert schema.covering_region(unknown_condition, regions) is None

        with pytest.raises(ValueError):
            schema.cell_status(
                self._descriptor(),
                (
                    owned_region,
                    CoverageRegion(
                        owner=Factorization,
                        predicates=owned_region.predicates,
                    ),
                ),
            )
        with pytest.raises(ValueError):
            schema.covering_region(self._descriptor(dim_x=5), regions)

        with pytest.raises(ValueError):
            ParameterSpaceSchema(
                name="empty_axis",
                axes=(ParameterAxis("unbinned_axis", ()),),
            ).validate_schema()
        with pytest.raises(ValueError):
            ParameterSpaceSchema(
                name="duplicate_fields",
                axes=(
                    ParameterAxis("same_field", (NumericInterval("all", lower=0.0),)),
                    ParameterAxis("same_field", (NumericInterval("all", lower=0.0),)),
                ),
            ).validate_schema()
        with pytest.raises(ValueError):
            schema.validate_coverage_region(
                CoverageRegion(
                    owner=IterativeSolver,
                    predicates=(ComparisonPredicate("undeclared", "==", 1),),
                )
            )
        with pytest.raises(TypeError):
            schema.validate_coverage_region(
                CoverageRegion(
                    owner=IterativeSolver,
                    predicates=(object(),),  # type: ignore[arg-type]
                )
            )

    @staticmethod
    def _assert_axis_has_one_identity() -> None:
        tree = ast.parse(
            (_PACKAGE_ROOT / "computation" / "algorithm_capabilities.py").read_text()
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != "ParameterAxis":
                continue
            fields = {
                child.target.id
                for child in node.body
                if isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
            }
            assert "field" in fields
            assert not {"name", "descriptor_field"} & fields
            return
        raise AssertionError("ParameterAxis class not found")

    @staticmethod
    def _assert_descriptor_has_no_schema_identity() -> None:
        tree = ast.parse(
            (_PACKAGE_ROOT / "computation" / "algorithm_capabilities.py").read_text()
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != "ParameterDescriptor":
                continue
            fields = {
                child.target.id
                for child in node.body
                if isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
            }
            assert "coordinates" in fields
            assert "schema" not in fields
            return
        raise AssertionError("ParameterDescriptor class not found")

    @staticmethod
    def _descriptor(
        *,
        map_linearity_defect: float | None = 0.0,
        dim_x: int = 4,
        dim_y: int = 4,
        coercivity_lower_bound: float = 1.0,
        condition_estimate: float | None = 10.0,
        operator_representation: str = "assembled_dense",
    ) -> ParameterDescriptor:
        return ParameterDescriptor(
            {
                "map_linearity_defect": DescriptorCoordinate(map_linearity_defect),
                "dim_x": DescriptorCoordinate(dim_x),
                "dim_y": DescriptorCoordinate(dim_y),
                "coercivity_lower_bound": DescriptorCoordinate(coercivity_lower_bound),
                "condition_estimate": DescriptorCoordinate(condition_estimate),
                "operator_representation": DescriptorCoordinate(
                    operator_representation
                ),
            }
        )


class _SolveRelationSchemaClaim(Claim[None]):
    """Claim: solver schemas derive problem names from primitive coordinates."""

    @property
    def description(self) -> str:
        return "algorithm_capabilities/solve_relation_schemas"

    def check(self, _calibration: None) -> None:
        solve_schema = solve_relation_parameter_schema()
        linear_schema = linear_solver_parameter_schema()
        decomposition_schema = decomposition_parameter_schema()

        for schema in (solve_schema, linear_schema, decomposition_schema):
            schema.validate_schema()
            assert "problem_kind" not in schema.descriptor_fields

        solve_regions = self._regions(solve_schema)
        assert {
            "linear_system",
            "least_squares",
            "nonlinear_root",
            "eigenproblem",
        } <= set(solve_regions)
        for name in (
            "linear_system",
            "least_squares",
            "nonlinear_root",
            "eigenproblem",
        ):
            self._assert_region_uses_primitive_axes(solve_schema, solve_regions[name])

        linear_system = self._solve_descriptor()
        least_squares = self._solve_descriptor(
            dim_x=3,
            dim_y=5,
            objective_relation="least_squares",
        )
        nonlinear_root = self._solve_descriptor(
            map_linearity_defect=None,
            map_linearity_evidence="unavailable",
            residual_target_available=False,
        )
        eigenproblem = self._solve_descriptor(
            auxiliary_scalar_count=1,
            normalization_constraint_count=1,
            acceptance_relation="eigenpair_residual",
            objective_relation="spectral_residual",
        )

        for descriptor in (
            linear_system,
            least_squares,
            nonlinear_root,
            eigenproblem,
        ):
            solve_schema.validate_descriptor(descriptor)
        assert solve_regions["linear_system"].contains(linear_system)
        assert solve_regions["least_squares"].contains(least_squares)
        assert solve_regions["nonlinear_root"].contains(nonlinear_root)
        assert solve_regions["eigenproblem"].contains(eigenproblem)

        invalid_eigenproblem = self._solve_descriptor(
            acceptance_relation="eigenpair_residual",
        )
        assert solve_schema.cell_status(invalid_eigenproblem, ()) == "invalid"

        linear_regions = self._regions(linear_schema)
        for name in (
            "square",
            "overdetermined",
            "full_rank",
            "rank_deficient",
            "symmetric_positive_definite",
            "matrix_free",
        ):
            self._assert_region_uses_primitive_axes(linear_schema, linear_regions[name])

        spd_descriptor = self._linear_descriptor()
        rank_deficient_descriptor = self._linear_descriptor(
            singular_value_lower_bound=0.0,
            rank_estimate=3,
            nullity_estimate=1,
        )
        matrix_free_descriptor = self._linear_descriptor(
            linear_operator_matrix_available=False,
            matrix_representation_available=False,
        )
        overdetermined_descriptor = self._linear_descriptor(
            dim_x=3,
            dim_y=5,
            symmetry_defect=1.0,
            coercivity_lower_bound=0.0,
        )

        for descriptor in (
            spd_descriptor,
            rank_deficient_descriptor,
            matrix_free_descriptor,
            overdetermined_descriptor,
        ):
            linear_schema.validate_descriptor(descriptor)
        assert linear_regions["symmetric_positive_definite"].contains(spd_descriptor)
        assert linear_regions["full_rank"].contains(spd_descriptor)
        assert linear_regions["rank_deficient"].contains(rank_deficient_descriptor)
        assert linear_regions["matrix_free"].contains(matrix_free_descriptor)
        assert linear_regions["overdetermined"].contains(overdetermined_descriptor)
        assert linear_schema.cell_status(overdetermined_descriptor, ()) == "uncovered"
        assert (
            linear_schema.cell_status(
                self._linear_descriptor(dim_y=5),
                (),
            )
            == "invalid"
        )

        decomposition_regions = self._regions(decomposition_schema)
        decomp_descriptor = self._decomposition_descriptor()
        rank_deficient_decomp = self._decomposition_descriptor(
            singular_value_lower_bound=0.0,
            rank_estimate=3,
            nullity_estimate=1,
        )
        for descriptor in (decomp_descriptor, rank_deficient_decomp):
            decomposition_schema.validate_descriptor(descriptor)
        assert decomposition_regions["square"].contains(decomp_descriptor)
        assert decomposition_regions["full_rank"].contains(decomp_descriptor)
        assert decomposition_regions["rank_deficient"].contains(rank_deficient_decomp)
        assert (
            decomposition_schema.cell_status(
                self._decomposition_descriptor(matrix_columns=5),
                (),
            )
            == "invalid"
        )

        with pytest.raises(ValueError):
            ParameterSpaceSchema(
                name="bad_region",
                axes=(
                    ParameterAxis("declared", (ParameterBin("one", frozenset({1})),)),
                ),
                derived_regions=(
                    DerivedParameterRegion(
                        "uses_private_axis",
                        ((ComparisonPredicate("private", "==", 1),),),
                    ),
                ),
            ).validate_schema()

    @staticmethod
    def _regions(
        schema: ParameterSpaceSchema,
    ) -> dict[str, DerivedParameterRegion]:
        return {region.name: region for region in schema.derived_regions}

    @staticmethod
    def _assert_region_uses_primitive_axes(
        schema: ParameterSpaceSchema,
        region: DerivedParameterRegion,
    ) -> None:
        assert region.referenced_fields
        assert region.referenced_fields <= schema.descriptor_fields

    @staticmethod
    def _solve_descriptor(
        *,
        dim_x: int = 4,
        dim_y: int = 4,
        auxiliary_scalar_count: int = 0,
        equality_constraint_count: int = 0,
        normalization_constraint_count: int = 0,
        residual_target_available: bool = True,
        target_is_zero: bool = False,
        map_linearity_defect: float | None = 0.0,
        map_linearity_evidence: str = "exact",
        matrix_representation_available: bool = True,
        operator_application_available: bool = True,
        derivative_oracle_kind: str = "matrix",
        objective_relation: str = "none",
        acceptance_relation: str = "residual_below_tolerance",
        requested_residual_tolerance: float = 1.0e-8,
        requested_solution_tolerance: float = 1.0e-8,
        backend_kind: str = "python",
        device_kind: str = "cpu",
        work_budget_fmas: float = 1.0e9,
        memory_budget_bytes: float = 1.0e9,
    ) -> ParameterDescriptor:
        return ParameterDescriptor(
            {
                LinearSolverField.DIM_X: DescriptorCoordinate(dim_x),
                LinearSolverField.DIM_Y: DescriptorCoordinate(dim_y),
                LinearSolverField.AUXILIARY_SCALAR_COUNT: DescriptorCoordinate(
                    auxiliary_scalar_count
                ),
                LinearSolverField.EQUALITY_CONSTRAINT_COUNT: DescriptorCoordinate(
                    equality_constraint_count
                ),
                LinearSolverField.NORMALIZATION_CONSTRAINT_COUNT: DescriptorCoordinate(
                    normalization_constraint_count
                ),
                LinearSolverField.RESIDUAL_TARGET_AVAILABLE: DescriptorCoordinate(
                    residual_target_available
                ),
                LinearSolverField.TARGET_IS_ZERO: DescriptorCoordinate(target_is_zero),
                LinearSolverField.MAP_LINEARITY_DEFECT: DescriptorCoordinate(
                    map_linearity_defect,
                    evidence=map_linearity_evidence,  # type: ignore[arg-type]
                ),
                LinearSolverField.MATRIX_REPRESENTATION_AVAILABLE: DescriptorCoordinate(
                    matrix_representation_available
                ),
                LinearSolverField.OPERATOR_APPLICATION_AVAILABLE: DescriptorCoordinate(
                    operator_application_available
                ),
                LinearSolverField.DERIVATIVE_ORACLE_KIND: DescriptorCoordinate(
                    derivative_oracle_kind
                ),
                LinearSolverField.OBJECTIVE_RELATION: DescriptorCoordinate(
                    objective_relation
                ),
                LinearSolverField.ACCEPTANCE_RELATION: DescriptorCoordinate(
                    acceptance_relation
                ),
                LinearSolverField.REQUESTED_RESIDUAL_TOLERANCE: DescriptorCoordinate(
                    requested_residual_tolerance
                ),
                LinearSolverField.REQUESTED_SOLUTION_TOLERANCE: DescriptorCoordinate(
                    requested_solution_tolerance
                ),
                LinearSolverField.BACKEND_KIND: DescriptorCoordinate(backend_kind),
                LinearSolverField.DEVICE_KIND: DescriptorCoordinate(device_kind),
                LinearSolverField.WORK_BUDGET_FMAS: DescriptorCoordinate(
                    work_budget_fmas
                ),
                LinearSolverField.MEMORY_BUDGET_BYTES: DescriptorCoordinate(
                    memory_budget_bytes
                ),
            }
        )

    @classmethod
    def _linear_descriptor(
        cls,
        *,
        linear_operator_matrix_available: bool = True,
        assembly_cost_fmas: float = 64.0,
        matvec_cost_fmas: float = 32.0,
        linear_operator_memory_bytes: float = 512.0,
        symmetry_defect: float = 0.0,
        skew_symmetry_defect: float = 1.0,
        diagonal_nonzero_margin: float = 1.0,
        diagonal_dominance_margin: float = 1.0,
        coercivity_lower_bound: float = 1.0,
        singular_value_lower_bound: float = 1.0,
        condition_estimate: float | None = 10.0,
        condition_evidence: str = "exact",
        rank_estimate: int = 4,
        nullity_estimate: int = 0,
        rhs_consistency_defect: float = 0.0,
        **solve_overrides: Any,
    ) -> ParameterDescriptor:
        descriptor = cls._solve_descriptor(**solve_overrides)
        return ParameterDescriptor(
            descriptor.coordinates
            | {
                LinearSolverField.LINEAR_OPERATOR_MATRIX_AVAILABLE: (
                    DescriptorCoordinate(linear_operator_matrix_available)
                ),
                LinearSolverField.ASSEMBLY_COST_FMAS: DescriptorCoordinate(
                    assembly_cost_fmas
                ),
                LinearSolverField.MATVEC_COST_FMAS: DescriptorCoordinate(
                    matvec_cost_fmas
                ),
                LinearSolverField.LINEAR_OPERATOR_MEMORY_BYTES: DescriptorCoordinate(
                    linear_operator_memory_bytes
                ),
                LinearSolverField.SYMMETRY_DEFECT: DescriptorCoordinate(
                    symmetry_defect
                ),
                LinearSolverField.SKEW_SYMMETRY_DEFECT: DescriptorCoordinate(
                    skew_symmetry_defect
                ),
                LinearSolverField.DIAGONAL_NONZERO_MARGIN: DescriptorCoordinate(
                    diagonal_nonzero_margin
                ),
                LinearSolverField.DIAGONAL_DOMINANCE_MARGIN: DescriptorCoordinate(
                    diagonal_dominance_margin
                ),
                LinearSolverField.COERCIVITY_LOWER_BOUND: DescriptorCoordinate(
                    coercivity_lower_bound
                ),
                LinearSolverField.SINGULAR_VALUE_LOWER_BOUND: DescriptorCoordinate(
                    singular_value_lower_bound
                ),
                LinearSolverField.CONDITION_ESTIMATE: DescriptorCoordinate(
                    condition_estimate,
                    evidence=condition_evidence,  # type: ignore[arg-type]
                ),
                LinearSolverField.RANK_ESTIMATE: DescriptorCoordinate(rank_estimate),
                LinearSolverField.NULLITY_ESTIMATE: DescriptorCoordinate(
                    nullity_estimate
                ),
                LinearSolverField.RHS_CONSISTENCY_DEFECT: DescriptorCoordinate(
                    rhs_consistency_defect
                ),
            }
        )

    @staticmethod
    def _decomposition_descriptor(
        *,
        matrix_rows: int = 4,
        matrix_columns: int = 4,
        factorization_work_budget_fmas: float = 1.0e9,
        factorization_memory_budget_bytes: float = 1.0e9,
        assembly_cost_fmas: float = 64.0,
        matvec_cost_fmas: float = 32.0,
        linear_operator_memory_bytes: float = 512.0,
        symmetry_defect: float = 0.0,
        skew_symmetry_defect: float = 1.0,
        diagonal_nonzero_margin: float = 1.0,
        diagonal_dominance_margin: float = 1.0,
        coercivity_lower_bound: float = 1.0,
        singular_value_lower_bound: float = 1.0,
        condition_estimate: float = 10.0,
        rank_estimate: int = 4,
        nullity_estimate: int = 0,
        rhs_consistency_defect: float = 0.0,
        linear_operator_matrix_available: bool = True,
    ) -> ParameterDescriptor:
        return ParameterDescriptor(
            {
                DecompositionField.MATRIX_ROWS: DescriptorCoordinate(matrix_rows),
                DecompositionField.MATRIX_COLUMNS: DescriptorCoordinate(matrix_columns),
                DecompositionField.FACTORIZATION_WORK_BUDGET_FMAS: (
                    DescriptorCoordinate(factorization_work_budget_fmas)
                ),
                DecompositionField.FACTORIZATION_MEMORY_BUDGET_BYTES: (
                    DescriptorCoordinate(factorization_memory_budget_bytes)
                ),
                LinearSolverField.LINEAR_OPERATOR_MATRIX_AVAILABLE: (
                    DescriptorCoordinate(linear_operator_matrix_available)
                ),
                LinearSolverField.ASSEMBLY_COST_FMAS: DescriptorCoordinate(
                    assembly_cost_fmas
                ),
                LinearSolverField.MATVEC_COST_FMAS: DescriptorCoordinate(
                    matvec_cost_fmas
                ),
                LinearSolverField.LINEAR_OPERATOR_MEMORY_BYTES: DescriptorCoordinate(
                    linear_operator_memory_bytes
                ),
                LinearSolverField.SYMMETRY_DEFECT: DescriptorCoordinate(
                    symmetry_defect
                ),
                LinearSolverField.SKEW_SYMMETRY_DEFECT: DescriptorCoordinate(
                    skew_symmetry_defect
                ),
                LinearSolverField.DIAGONAL_NONZERO_MARGIN: DescriptorCoordinate(
                    diagonal_nonzero_margin
                ),
                LinearSolverField.DIAGONAL_DOMINANCE_MARGIN: DescriptorCoordinate(
                    diagonal_dominance_margin
                ),
                LinearSolverField.COERCIVITY_LOWER_BOUND: DescriptorCoordinate(
                    coercivity_lower_bound
                ),
                LinearSolverField.SINGULAR_VALUE_LOWER_BOUND: DescriptorCoordinate(
                    singular_value_lower_bound
                ),
                LinearSolverField.CONDITION_ESTIMATE: DescriptorCoordinate(
                    condition_estimate
                ),
                LinearSolverField.RANK_ESTIMATE: DescriptorCoordinate(rank_estimate),
                LinearSolverField.NULLITY_ESTIMATE: DescriptorCoordinate(
                    nullity_estimate
                ),
                LinearSolverField.RHS_CONSISTENCY_DEFECT: DescriptorCoordinate(
                    rhs_consistency_defect
                ),
            }
        )


class _LinearOperatorDescriptorClaim(Claim[None]):
    """Claim: small assembled operators produce schema-valid descriptors."""

    @property
    def description(self) -> str:
        return "algorithm_capabilities/linear_operator_descriptor"

    def check(self, _calibration: None) -> None:
        schema = linear_solver_parameter_schema()
        regions = _SolveRelationSchemaClaim._regions(schema)

        spd = linear_operator_descriptor_from_assembled_operator(
            _MatrixLinearOperator(((2.0, -1.0), (-1.0, 2.0))),
            Tensor([1.0, 0.0], backend=_JIT_BACKEND),
        )
        schema.validate_descriptor(spd.parameter_descriptor)
        assert spd.matrix == ((2.0, -1.0), (-1.0, 2.0))
        assert regions["linear_system"].contains(spd.parameter_descriptor)
        assert regions["symmetric_positive_definite"].contains(spd.parameter_descriptor)
        assert regions["full_rank"].contains(spd.parameter_descriptor)
        assert spd.coordinate("symmetry_defect").value == 0.0
        assert spd.coordinate("coercivity_lower_bound").value == 1.0
        assert spd.coordinate("coercivity_lower_bound").evidence == "lower_bound"
        assert spd.coordinate("rank_estimate").value == 2
        assert spd.coordinate("rhs_consistency_defect").value == 0.0

        rank_deficient = linear_operator_descriptor_from_assembled_operator(
            _MatrixLinearOperator(((1.0, 1.0), (1.0, 1.0))),
            Tensor([1.0, 1.0], backend=_JIT_BACKEND),
        )
        schema.validate_descriptor(rank_deficient.parameter_descriptor)
        assert regions["rank_deficient"].contains(rank_deficient.parameter_descriptor)
        assert rank_deficient.coordinate("rank_estimate").value == 1
        assert rank_deficient.coordinate("nullity_estimate").value == 1
        assert rank_deficient.coordinate("singular_value_lower_bound").value == 0.0
        assert rank_deficient.coordinate("rhs_consistency_defect").value == 0.0

        inconsistent = linear_operator_descriptor_from_assembled_operator(
            _MatrixLinearOperator(((1.0, 1.0), (1.0, 1.0))),
            Tensor([1.0, 0.0], backend=_JIT_BACKEND),
        )
        schema.validate_descriptor(inconsistent.parameter_descriptor)
        assert float(inconsistent.coordinate("rhs_consistency_defect").value) > 0.0


class _LinearSolverCoverageRegionClaim(Claim[None]):
    """Claim: linear-solver selection is driven by schema coverage regions."""

    @property
    def description(self) -> str:
        return "algorithm_capabilities/linear_solver_coverage_regions"

    def check(self, _calibration: None) -> None:
        self._assert_no_local_disjointness_algebra()
        self._assert_no_legacy_coverage_region_names()
        self._assert_coverage_region_identity_is_owner_class()
        self._assert_no_coverage_region_priority_model()
        self._assert_no_coverage_region_status_model()
        self._assert_no_solver_local_point_query()
        self._assert_no_solver_error_string_dispatch()
        self._assert_no_linear_solver_field_text_identity()
        schema = linear_solver_parameter_schema()
        assert set(LinearSolverField) == schema.descriptor_fields
        regions = linear_solver_coverage_regions()
        self._assert_no_declared_coverage_literals(regions)
        for region in regions:
            schema.validate_coverage_region(region)

        selected_owners: set[type] = set()
        self._assert_coverage_regions_disjoint(regions)
        for region in regions:
            descriptor = self._descriptor_witness_for_region(schema, region)
            assert schema.cell_status(descriptor, regions) == "owned"
            selected_owner = select_linear_solver_for_descriptor(descriptor)
            expected_owner = self._selected_region_owner(descriptor, regions)
            assert selected_owner is expected_owner
            selected_owners.add(expected_owner)
        assert selected_owners == {region.owner for region in regions}

    @classmethod
    def _selected_region_owner(
        cls,
        descriptor: ParameterDescriptor,
        regions: tuple[CoverageRegion, ...],
    ) -> type:
        matches = tuple(region for region in regions if region.contains(descriptor))
        assert matches
        assert len(matches) == cls._one(), (
            "owned linear-solver coverage regions must be pairwise disjoint at "
            "selected descriptors: "
            f"{[_coverage_region_name(region) for region in matches]}"
        )
        return next(iter(matches)).owner

    @classmethod
    def _assert_coverage_regions_disjoint(
        cls,
        regions: tuple[CoverageRegion, ...],
    ) -> None:
        cls._assert_coverage_disjointness_algebra()
        assert coverage_regions_are_disjoint(
            regions
        ), "linear-solver coverage regions must be pairwise disjoint"

    @classmethod
    def _assert_coverage_disjointness_algebra(cls) -> None:
        assert predicate_sets_are_disjoint(
            (MembershipPredicate("algebraic_kind", frozenset({"left"})),),
            (MembershipPredicate("algebraic_kind", frozenset({"right"})),),
        )
        assert predicate_sets_are_disjoint(
            (ComparisonPredicate("algebraic_bound", ">", cls._zero()),),
            (ComparisonPredicate("algebraic_bound", "<=", cls._zero()),),
        )
        assert predicate_sets_are_disjoint(
            (
                AffineComparisonPredicate(
                    {"algebraic_budget": cls._one(), "algebraic_cost": -cls._one()},
                    ">=",
                    cls._zero(),
                ),
            ),
            (
                AffineComparisonPredicate(
                    {"algebraic_budget": cls._one(), "algebraic_cost": -cls._one()},
                    "<",
                    cls._zero(),
                ),
            ),
        )
        assert not predicate_sets_are_disjoint(
            (ComparisonPredicate("algebraic_interval", "<=", cls._two()),),
            (ComparisonPredicate("algebraic_interval", ">", cls._one()),),
        )

    @classmethod
    def _assert_no_local_disjointness_algebra(cls) -> None:
        forbidden = {
            "_comparisons_are_disjoint",
            "_field_predicates_are_disjoint",
            "_predicate_sets_are_disjoint",
            "_strongest_lower_bound",
            "_strongest_upper_bound",
            "_value_satisfies_comparisons",
        }
        tree = ast.parse(inspect.getsource(cls))
        local_helpers = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        assert not forbidden & local_helpers

    @staticmethod
    def _assert_no_legacy_coverage_region_names() -> None:
        source_paths = (
            _PACKAGE_ROOT / "computation" / "algorithm_capabilities.py",
            _PACKAGE_ROOT / "computation" / "solvers" / "__init__.py",
            _PACKAGE_ROOT / "computation" / "solvers" / "capabilities.py",
            _PACKAGE_ROOT / "computation" / "solvers" / "coverage.py",
            _PROJECT_ROOT / "scripts" / "gen_capability_atlas_docs.py",
        )
        forbidden_fragments = ("CoveragePatch", "coverage_patch", "coverage_patches")
        forbidden_words = ("patch", "patches")
        for source_path in source_paths:
            tree = ast.parse(source_path.read_text())
            for node in ast.walk(tree):
                names: tuple[str, ...]
                if isinstance(node, ast.ClassDef | ast.FunctionDef):
                    names = (node.name,)
                elif isinstance(node, ast.Name):
                    names = (node.id,)
                elif isinstance(node, ast.Attribute):
                    names = (node.attr,)
                elif isinstance(node, ast.arg):
                    names = (node.arg,)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    names = (node.value,)
                else:
                    continue
                for name in names:
                    words = tuple(name.replace("_", " ").split())
                    if any(fragment in name for fragment in forbidden_fragments) or (
                        "coverage" in words
                        and any(word in words for word in forbidden_words)
                    ):
                        raise AssertionError(
                            "coverage ownership must use region terminology: "
                            f"{source_path.relative_to(_PROJECT_ROOT)}: {name!r}"
                        )

    @staticmethod
    def _assert_coverage_region_identity_is_owner_class() -> None:
        source_paths = (
            _PACKAGE_ROOT / "computation" / "solvers" / "capabilities.py",
            _PACKAGE_ROOT / "computation" / "solvers" / "coverage.py",
        )
        for source_path in source_paths:
            tree = ast.parse(source_path.read_text())
            for node in ast.walk(tree):
                if (
                    source_path.name == "coverage.py"
                    and isinstance(node, ast.Attribute)
                    and node.attr == "__name__"
                ):
                    raise AssertionError(
                        "coverage identity must not be stringified in the model: "
                        f"{source_path.relative_to(_PROJECT_ROOT)}"
                    )
                if isinstance(node, ast.Call):
                    call_name = (
                        node.func.id if isinstance(node.func, ast.Name) else None
                    )
                    if call_name == "CoverageRegion":
                        assert len(node.args) <= 2, (
                            "CoverageRegion identity is owner plus predicates, "
                            "not a separate name: "
                            f"{source_path.relative_to(_PROJECT_ROOT)}"
                        )

        tree = ast.parse(
            (_PACKAGE_ROOT / "computation" / "algorithm_capabilities.py").read_text()
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "CoverageRegion":
                fields = {
                    child.target.id: child.annotation
                    for child in node.body
                    if isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                }
                assert "name" not in fields
                assert isinstance(fields.get("owner"), ast.Name)
                assert fields["owner"].id == "type"
                return
        raise AssertionError("CoverageRegion class not found")

    @staticmethod
    def _assert_no_coverage_region_priority_model() -> None:
        source_paths = (_PROJECT_ROOT / "scripts" / "gen_capability_atlas_docs.py",)
        for source_path in source_paths:
            tree = ast.parse(source_path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "priority":
                    raise AssertionError(
                        "coverage machinery must not read priority: "
                        f"{source_path.relative_to(_PROJECT_ROOT)}"
                    )
                if isinstance(node, ast.keyword) and node.arg == "priority":
                    raise AssertionError(
                        "coverage machinery must not pass priority: "
                        f"{source_path.relative_to(_PROJECT_ROOT)}"
                    )

        tree = ast.parse(
            (_PACKAGE_ROOT / "computation" / "algorithm_capabilities.py").read_text()
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "CoverageRegion":
                fields = {
                    child.target.id
                    for child in node.body
                    if isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                }
                assert "priority" not in fields
                return
        raise AssertionError("CoverageRegion class not found")

    @staticmethod
    def _assert_no_coverage_region_status_model() -> None:
        source_paths = (
            _PACKAGE_ROOT / "computation" / "algorithm_capabilities.py",
            _PROJECT_ROOT / "scripts" / "gen_capability_atlas_docs.py",
        )
        for source_path in source_paths:
            tree = ast.parse(source_path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                call_name = node.func.id if isinstance(node.func, ast.Name) else None
                if call_name != "CoverageRegion":
                    continue
                for keyword in node.keywords:
                    if keyword.arg == "status":
                        raise AssertionError(
                            "coverage machinery must not pass region status: "
                            f"{source_path.relative_to(_PROJECT_ROOT)}"
                        )

        tree = ast.parse(next(iter(source_paths)).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "CoverageRegion":
                fields = {
                    child.target.id
                    for child in node.body
                    if isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                }
                assert "status" not in fields
                return
        raise AssertionError("CoverageRegion class not found")

    @staticmethod
    def _assert_no_solver_local_point_query() -> None:
        tree = ast.parse(
            (_PACKAGE_ROOT / "computation" / "solvers" / "capabilities.py").read_text()
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.BoolOp):
                continue
            call_names = {
                child.func.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
            }
            comparisons = {
                child.left.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Compare)
                and isinstance(child.left, ast.Attribute)
            }
            assert not (
                "contains" in call_names and "status" in comparisons
            ), "solver selection must use ParameterSpaceSchema.covering_region"

    @staticmethod
    def _assert_no_solver_error_string_dispatch() -> None:
        tree = ast.parse(
            (_PACKAGE_ROOT / "computation" / "solvers" / "capabilities.py").read_text()
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "str":
                    raise AssertionError(
                        "solver selection must not dispatch on exception text"
                    )

    @staticmethod
    def _assert_no_linear_solver_field_text_identity() -> None:
        tree = ast.parse(
            (_PACKAGE_ROOT / "computation" / "algorithm_capabilities.py").read_text()
        )
        field_labels = {field.value for field in LinearSolverField}
        identity_calls = {
            "AffineComparisonPredicate",
            "ComparisonPredicate",
            "EvidencePredicate",
            "MembershipPredicate",
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = node.func.id if isinstance(node.func, ast.Name) else None
            if call_name in identity_calls:
                candidates = []
                if node.args:
                    candidates.append(node.args[0])
                candidates.extend(
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg in {"field", "terms"}
                )
                for candidate in candidates:
                    if _LinearSolverCoverageRegionClaim._field_label_literal(
                        candidate, field_labels
                    ):
                        raise AssertionError(
                            "parameter-space field identity must be symbolic, "
                            "with strings limited to labels and rendering"
                        )
            if call_name == "ParameterDescriptor":
                coordinates = next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "coordinates"
                    ),
                    node.args[1] if len(node.args) > 1 else None,
                )
                if _LinearSolverCoverageRegionClaim._field_label_literal(
                    coordinates, field_labels
                ):
                    raise AssertionError(
                        "parameter descriptor coordinates must be keyed by "
                        "symbolic field identity"
                    )

    @classmethod
    def _field_label_literal(cls, node: ast.AST | None, labels: set[str]) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, str) and node.value in labels
        if isinstance(node, ast.Dict):
            return any(cls._field_label_literal(key, labels) for key in node.keys)
        return False

    @classmethod
    def _assert_no_declared_coverage_literals(
        cls,
        regions: tuple[CoverageRegion, ...],
    ) -> None:
        declared = cls._declared_coverage_literals(regions)
        leaked = declared & cls._claim_source_literals()
        assert not leaked, f"coverage facts leaked into structural claim: {leaked}"

    @staticmethod
    def _declared_coverage_literals(
        regions: tuple[CoverageRegion, ...],
    ) -> frozenset[str]:
        literals: set[str] = set()
        for region in regions:
            for predicate in region.predicates:
                literals.update(
                    _field_label(field) for field in predicate.referenced_fields
                )
                if isinstance(predicate, ComparisonPredicate):
                    if isinstance(predicate.value, str):
                        literals.add(predicate.value)
                elif isinstance(predicate, MembershipPredicate):
                    literals.update(
                        value for value in predicate.values if isinstance(value, str)
                    )
                elif isinstance(predicate, EvidencePredicate):
                    literals.update(predicate.evidence)
                elif isinstance(predicate, AffineComparisonPredicate):
                    literals.update(_field_label(field) for field in predicate.terms)
        return frozenset(literals)

    @classmethod
    def _claim_source_literals(cls) -> frozenset[str | int | float]:
        tree = ast.parse(inspect.getsource(cls))
        literals: set[str | int | float] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(
                node.value, str | int | float
            ):
                if not isinstance(node.value, bool):
                    literals.add(node.value)
        return frozenset(literals)

    @classmethod
    def _descriptor_witness_for_region(
        cls,
        schema: ParameterSpaceSchema,
        region: CoverageRegion,
    ) -> ParameterDescriptor:
        fields = {axis.field: axis for axis in schema.axes}
        coordinates = {
            field: DescriptorCoordinate(cls._axis_witness(axis))
            for field, axis in fields.items()
        }
        for predicate in region.predicates:
            if isinstance(predicate, MembershipPredicate):
                coordinates[predicate.field] = DescriptorCoordinate(
                    cls._membership_witness(fields[predicate.field], predicate)
                )
            elif isinstance(predicate, ComparisonPredicate):
                coordinates[predicate.field] = DescriptorCoordinate(
                    cls._comparison_witness(fields[predicate.field], predicate)
                )
            elif isinstance(predicate, EvidencePredicate):
                evidence = next(iter(sorted(predicate.evidence)))
                value = coordinates[predicate.field].value
                coordinates[predicate.field] = DescriptorCoordinate(
                    value,
                    evidence=evidence,  # type: ignore[arg-type]
                )
            elif isinstance(predicate, AffineComparisonPredicate):
                cls._satisfy_affine_predicate(fields, coordinates, predicate)
        descriptor = ParameterDescriptor(coordinates)
        assert region.contains(descriptor)
        return descriptor

    @classmethod
    def _axis_witness(cls, axis: ParameterAxis) -> Any:
        for bin_or_interval in axis.bins:
            if isinstance(bin_or_interval, ParameterBin):
                return next(iter(sorted(bin_or_interval.values, key=str)))
            value = cls._interval_witness(bin_or_interval)
            if axis.contains(DescriptorCoordinate(value)):
                return value
        raise AssertionError(f"axis {axis.label!r} has no witness value")

    @staticmethod
    def _interval_witness(interval: NumericInterval) -> float:
        if interval.lower is not None and interval.upper is not None:
            if interval.include_lower and interval.contains(interval.lower):
                return interval.lower
            if interval.include_upper and interval.contains(interval.upper):
                return interval.upper
            return (
                interval.lower + interval.upper
            ) / _LinearSolverCoverageRegionClaim._two()
        if interval.lower is not None:
            if interval.include_lower and interval.contains(interval.lower):
                return interval.lower
            return interval.lower + max(
                abs(interval.lower), _LinearSolverCoverageRegionClaim._one()
            )
        if interval.upper is not None:
            if interval.include_upper and interval.contains(interval.upper):
                return interval.upper
            return interval.upper - max(
                abs(interval.upper), _LinearSolverCoverageRegionClaim._one()
            )
        return _LinearSolverCoverageRegionClaim._zero()

    @classmethod
    def _membership_witness(
        cls,
        axis: ParameterAxis,
        predicate: MembershipPredicate,
    ) -> Any:
        for value in sorted(predicate.values, key=str):
            if axis.contains(DescriptorCoordinate(value)):
                return value
        raise AssertionError(f"membership predicate {predicate!r} has no axis witness")

    @classmethod
    def _comparison_witness(
        cls,
        axis: ParameterAxis,
        predicate: ComparisonPredicate,
    ) -> Any:
        if predicate.operator in {"==", "<=", ">="} and axis.contains(
            DescriptorCoordinate(predicate.value)
        ):
            return predicate.value
        for bin_or_interval in axis.bins:
            if isinstance(bin_or_interval, ParameterBin):
                candidates = sorted(bin_or_interval.values, key=str)
            else:
                candidates = (
                    cls._interval_witness(bin_or_interval),
                    cls._nudged_value(predicate),
                )
            for candidate in candidates:
                coordinate = DescriptorCoordinate(candidate)
                if axis.contains(coordinate) and predicate.evaluate(
                    ParameterDescriptor({axis.field: coordinate})
                ):
                    return candidate
        raise AssertionError(f"comparison predicate {predicate!r} has no axis witness")

    @classmethod
    def _nudged_value(cls, predicate: ComparisonPredicate) -> float:
        assert not isinstance(predicate.value, bool | str)
        value = float(predicate.value)
        step = max(abs(value), cls._one())
        if predicate.operator in {">", ">="}:
            return value + step
        if predicate.operator in {"<", "<="}:
            return value - step
        raise AssertionError(f"unsupported comparison witness {predicate!r}")

    @classmethod
    def _satisfy_affine_predicate(
        cls,
        fields: dict[DescriptorField, ParameterAxis],
        coordinates: dict[DescriptorField, DescriptorCoordinate],
        predicate: AffineComparisonPredicate,
    ) -> None:
        if predicate.evaluate(ParameterDescriptor(coordinates)):
            return
        adjustable = next(iter(predicate.terms))
        coefficient = predicate.terms[adjustable]
        other_total = predicate.offset + sum(
            term_coefficient * float(coordinates[field].value)
            for field, term_coefficient in predicate.terms.items()
            if field != adjustable
        )
        boundary = (predicate.value - other_total) / coefficient
        candidates = (
            boundary,
            cls._affine_nudged_value(boundary, coefficient, predicate),
        )
        for candidate in candidates:
            coordinates[adjustable] = DescriptorCoordinate(candidate)
            if fields[adjustable].contains(
                coordinates[adjustable]
            ) and predicate.evaluate(ParameterDescriptor(coordinates)):
                return
        raise AssertionError(
            f"affine predicate {predicate!r} has no descriptor witness"
        )

    @classmethod
    def _affine_nudged_value(
        cls,
        boundary: float,
        coefficient: float,
        predicate: AffineComparisonPredicate,
    ) -> float:
        step = max(abs(boundary), cls._one()) / cls._two()
        direction = cls._one() if coefficient > cls._zero() else -cls._one()
        if predicate.operator in {">", ">="}:
            return boundary + direction * step
        if predicate.operator in {"<", "<="}:
            return boundary - direction * step
        return boundary

    @staticmethod
    def _zero() -> float:
        return float(False)

    @staticmethod
    def _one() -> float:
        return float(True)

    @classmethod
    def _two(cls) -> float:
        return cls._one() + cls._one()


class _LinearSolverCoverageLocalityClaim(Claim[None]):
    """Claim: owned solver coverage regions are declared inside implementations."""

    @property
    def description(self) -> str:
        return "algorithm_capabilities/linear_solver_coverage_locality"

    def check(self, _calibration: None) -> None:
        support_tree = ast.parse(
            (_PACKAGE_ROOT / "computation" / "solvers" / "coverage.py").read_text()
        )
        self._assert_no_coverage_record_wrapper()
        assert not self._coverage_category_support(support_tree), (
            "linear-solver coverage records must not carry a parallel "
            "implementation category"
        )
        assert not self._coverage_priority_support(
            support_tree
        ), "linear-solver coverage records must not carry selector priority"
        support_field_literals = self._raw_descriptor_field_arguments(support_tree)
        assert not support_field_literals, (
            "linear-solver coverage predicates must use schema-owned field "
            "symbols: computation/solvers/coverage.py"
        )
        for path in sorted((_PACKAGE_ROOT / "computation" / "solvers").glob("*.py")):
            if path.name.startswith("_") or path.name in {
                "capabilities.py",
                "coverage.py",
            }:
                continue
            tree = ast.parse(path.read_text())
            for owner, class_name in self._owned_coverage_locations(tree):
                assert owner == class_name, (
                    "owned linear-solver coverage region must be declared in "
                    f"class {owner}: {path.relative_to(_PROJECT_ROOT)}"
                )
            manual_categories = self._manual_coverage_categories(tree)
            assert not manual_categories, (
                "linear-solver coverage must not declare categories; implementation "
                f"groupings come from inheritance: {path.relative_to(_PROJECT_ROOT)}"
            )
            manual_names = self._manual_coverage_names(tree)
            assert not manual_names, (
                "linear-solver coverage names must come from class identity: "
                f"{path.relative_to(_PROJECT_ROOT)}"
            )
            manual_regions = self._manual_owned_region_calls(tree)
            assert not manual_regions, (
                "owned linear-solver coverage regions must be built from class "
                f"identity: {path.relative_to(_PROJECT_ROOT)}"
            )
            manual_contracts = self._manual_contract_calls(tree)
            assert not manual_contracts, (
                "linear-solver implementation coverage must not declare tag contracts: "
                f"{path.relative_to(_PROJECT_ROOT)}"
            )
            manual_providers = self._manual_coverage_provider_methods(tree)
            assert not manual_providers, (
                "linear-solver coverage must be declared as class attributes, "
                f"not provider methods: {path.relative_to(_PROJECT_ROOT)}"
            )
            manual_priorities = self._manual_coverage_priorities(tree)
            assert not manual_priorities, (
                "linear-solver coverage partition boundaries must be expressed "
                "by predicates, not selector priority: "
                f"{path.relative_to(_PROJECT_ROOT)}"
            )
            nonlocal_requirements = self._nonlocal_coverage_requirements(tree)
            assert not nonlocal_requirements, (
                "implementation-local solver coverage must be irreducible "
                "predicate data; inherited contracts come from class structure: "
                f"{path.relative_to(_PROJECT_ROOT)}"
            )
            raw_field_literals = self._raw_descriptor_field_arguments(tree)
            assert not raw_field_literals, (
                "implementation-local solver coverage predicates must use "
                "schema-owned field symbols: "
                f"{path.relative_to(_PROJECT_ROOT)}"
            )

    @classmethod
    def _owned_coverage_locations(
        cls,
        tree: ast.Module,
    ) -> tuple[tuple[str, str | None], ...]:
        locations: list[tuple[str, str | None]] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        owner = cls._owned_coverage_owner(child)
                        if owner is not None:
                            locations.append((owner, node.name))
            else:
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        owner = cls._owned_coverage_owner(child)
                        if owner is not None:
                            locations.append((owner, None))
        return tuple(locations)

    @staticmethod
    def _assert_no_coverage_record_wrapper() -> None:
        source_paths = (
            _PACKAGE_ROOT / "computation" / "solvers" / "capabilities.py",
            _PACKAGE_ROOT / "computation" / "solvers" / "coverage.py",
        )
        for source_path in source_paths:
            tree = ast.parse(source_path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                fields = {
                    child.target.id
                    for child in node.body
                    if isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                }
                identity_fields = fields & {"implementation", "owner"}
                region_fields = {
                    field
                    for field in fields
                    if "coverage" in field and "region" in field
                }
                assert not (identity_fields and region_fields), (
                    "linear-solver coverage must not wrap an owner together "
                    "with owned regions; aggregate CoverageRegion directly: "
                    f"{source_path.relative_to(_PROJECT_ROOT)}:{node.name}"
                )

    @classmethod
    def _manual_coverage_categories(cls, tree: ast.Module) -> tuple[str, ...]:
        categories: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = cls._call_name(node.func)
            if call_name not in {"coverage", "capability"}:
                continue
            category = cls._string_arg(node, 2, "category")
            if category is not None:
                categories.append(category)
        return tuple(categories)

    @classmethod
    def _manual_coverage_names(cls, tree: ast.Module) -> tuple[str, ...]:
        names: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if cls._call_name(node.func) not in {"coverage", "capability"}:
                continue
            name = cls._string_arg(node, 1, "name")
            if name is not None:
                names.append(name)
        return tuple(names)

    @classmethod
    def _manual_owned_region_calls(cls, tree: ast.Module) -> tuple[str, ...]:
        calls: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = cls._call_name(node.func)
            if call_name == "owned_region":
                calls.append(call_name)
            elif call_name == "CoverageRegion":
                calls.append(call_name)
        return tuple(calls)

    @classmethod
    def _manual_contract_calls(cls, tree: ast.Module) -> tuple[str, ...]:
        calls: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if cls._call_name(node.func) != "contract":
                continue
            calls.append("contract")
        return tuple(calls)

    @staticmethod
    def _manual_coverage_provider_methods(tree: ast.Module) -> tuple[str, ...]:
        return tuple(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and ("coverage" in node.name or "capabilit" in node.name)
        )

    @staticmethod
    def _manual_coverage_priorities(tree: ast.Module) -> tuple[str, ...]:
        priorities: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("priority"):
                        priorities.append(target.id)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id.endswith("priority"):
                    priorities.append(node.target.id)
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "priority":
                        priorities.append(keyword.arg)
        return tuple(priorities)

    @classmethod
    def _nonlocal_coverage_requirements(cls, tree: ast.Module) -> tuple[str, ...]:
        violations: list[str] = []
        predicate_constructors = {
            "AffineComparisonPredicate",
            "ComparisonPredicate",
            "EvidencePredicate",
            "MembershipPredicate",
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            if not any(
                isinstance(target, ast.Name) and target.id == "linear_solver_coverage"
                for target in targets
            ):
                continue
            value = node.value
            if not isinstance(value, ast.Tuple):
                violations.append("linear_solver_coverage")
                continue
            for element in value.elts:
                if (
                    not isinstance(element, ast.Call)
                    or cls._call_name(element.func) not in predicate_constructors
                ):
                    violations.append("linear_solver_coverage")
                    break
        return tuple(violations)

    @classmethod
    def _raw_descriptor_field_arguments(cls, tree: ast.Module) -> tuple[str, ...]:
        violations: list[str] = []
        unary_predicates = {
            "ComparisonPredicate",
            "EvidencePredicate",
            "MembershipPredicate",
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = cls._call_name(node.func)
            if call_name in unary_predicates:
                field = cls._argument(node, 0, "field")
                if cls._contains_string_literal(field):
                    violations.append(call_name)
            if call_name == "AffineComparisonPredicate":
                terms = cls._argument(node, 0, "terms")
                if cls._mapping_has_string_keys(terms):
                    violations.append(call_name)
        return tuple(violations)

    @staticmethod
    def _argument(node: ast.Call, position: int, keyword: str) -> ast.expr | None:
        if len(node.args) > position:
            return node.args[position]
        return next((kw.value for kw in node.keywords if kw.arg == keyword), None)

    @staticmethod
    def _contains_string_literal(node: ast.AST | None) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, str)

    @classmethod
    def _mapping_has_string_keys(cls, node: ast.AST | None) -> bool:
        if isinstance(node, ast.Dict):
            return any(cls._contains_string_literal(key) for key in node.keys)
        return False

    @staticmethod
    def _coverage_category_support(tree: ast.Module) -> tuple[str, ...]:
        category_support: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "category":
                    category_support.append(node.target.id)
            if isinstance(node, ast.FunctionDef) and node.name == "category_for":
                category_support.append(node.name)
        return tuple(category_support)

    @staticmethod
    def _coverage_priority_support(tree: ast.Module) -> tuple[str, ...]:
        priority_support: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.arg) and node.arg.endswith("priority"):
                priority_support.append(node.arg)
            if isinstance(node, ast.keyword) and node.arg == "priority":
                priority_support.append(node.arg)
        return tuple(priority_support)

    @classmethod
    def _owned_coverage_owner(cls, node: ast.Call) -> str | None:
        call_name = cls._call_name(node.func)
        if call_name == "owned_region":
            return cls._string_arg(node, 1, "owner")
        if call_name != "CoverageRegion":
            return None
        return cls._string_arg(node, 1, "owner")

    @staticmethod
    def _call_name(func: ast.expr) -> str | None:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    @staticmethod
    def _string_arg(node: ast.Call, position: int, keyword: str) -> str | None:
        if len(node.args) > position:
            arg = node.args[position]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value
        for kw in node.keywords:
            if kw.arg == keyword and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    return kw.value.value
        return None


@dataclass(frozen=True)
class _AtlasGap:
    """Known uncovered descriptor region rendered into the atlas."""

    name: _AtlasText
    region: _AtlasText
    descriptor: tuple[_AtlasText, ...]
    selected_owner: _AtlasText
    partial_owners: tuple[_AtlasText, ...]
    required_capability: _AtlasText


@dataclass(frozen=True)
class _AtlasRegionShape:
    """Projected region geometry derived from schema predicates."""

    source: _AtlasRegionSource
    predicates: tuple[Any, ...]
    alternative_index: int
    alternative_count: int
    geometry: _AtlasGeometryKind
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class _AtlasPlotSpec:
    """One generated SVG projection of descriptor cells."""

    descriptors: tuple[ParameterDescriptor, ...]

    @property
    def schema(self) -> ParameterSpaceSchema:
        """Common schema for the descriptors shown by this plot."""
        schema = _atlas_schema_for_descriptor(self.descriptors[0])
        assert all(
            _atlas_schema_for_descriptor(descriptor) == schema
            for descriptor in self.descriptors
        )
        return schema

    @property
    def x_axis(self) -> _AtlasDescriptorField:
        """First coordinate axis of the schema projection plane."""
        return _atlas_axis_field(self.schema.axes[0])

    @property
    def y_axis(self) -> _AtlasDescriptorField:
        """Second coordinate axis of the schema projection plane."""
        return _atlas_axis_field(self.schema.axes[1])

    @property
    def x_range(self) -> tuple[float, float]:
        """Range derived from projected descriptor coordinates."""
        return _atlas_axis_range(self.descriptors, self.x_axis)

    @property
    def y_range(self) -> tuple[float, float]:
        """Range derived from projected descriptor coordinates."""
        return _atlas_axis_range(self.descriptors, self.y_axis)

    @property
    def filename(self) -> _AtlasText:
        """Human filename for the generated plot."""
        return _AtlasText(f"{self.schema.name}.svg")

    @property
    def title(self) -> _AtlasText:
        """Human title for the generated plot."""
        return _AtlasText(f"{self.schema.name.replace('_', '-').title()} Regions")

    @property
    def caption(self) -> _AtlasText:
        """Human caption for the generated plot."""
        return _AtlasText(
            f"{self.title} over {_field_label(self.x_axis)} and "
            f"{_field_label(self.y_axis)}."
        )


def _capability_atlas_descriptors() -> tuple[ParameterDescriptor, ...]:
    return (
        _SolveRelationSchemaClaim._solve_descriptor(),
        _SolveRelationSchemaClaim._solve_descriptor(
            dim_x=3,
            dim_y=5,
            objective_relation="least_squares",
        ),
        _SolveRelationSchemaClaim._solve_descriptor(
            map_linearity_defect=None,
            map_linearity_evidence="unavailable",
            residual_target_available=False,
        ),
        _SolveRelationSchemaClaim._solve_descriptor(
            auxiliary_scalar_count=1,
            normalization_constraint_count=1,
            acceptance_relation="eigenpair_residual",
            objective_relation="spectral_residual",
        ),
        _SolveRelationSchemaClaim._solve_descriptor(
            acceptance_relation="eigenpair_residual",
        ),
        _SolveRelationSchemaClaim._linear_descriptor(),
        _SolveRelationSchemaClaim._linear_descriptor(
            singular_value_lower_bound=0.0,
            rank_estimate=3,
            nullity_estimate=1,
        ),
        _SolveRelationSchemaClaim._linear_descriptor(
            linear_operator_matrix_available=False,
            matrix_representation_available=False,
        ),
        _SolveRelationSchemaClaim._linear_descriptor(dim_y=5),
        _SolveRelationSchemaClaim._decomposition_descriptor(),
        _SolveRelationSchemaClaim._decomposition_descriptor(
            matrix_columns=5,
        ),
    )


def _capability_atlas_schemas() -> tuple[ParameterSpaceSchema, ...]:
    return (
        solve_relation_parameter_schema(),
        linear_solver_parameter_schema(),
        decomposition_parameter_schema(),
    )


def _capability_atlas_coverage_regions() -> tuple[CoverageRegion, ...]:
    regions: dict[type, CoverageRegion] = {}
    for region in linear_solver_coverage_regions():
        regions[region.owner] = region
    return tuple(regions.values())


def _atlas_schema_for_descriptor(
    descriptor: ParameterDescriptor,
) -> ParameterSpaceSchema:
    candidates = tuple(
        schema
        for schema in _capability_atlas_schemas()
        if _descriptor_inhabits_schema(descriptor, schema)
    )
    minimal = tuple(
        schema
        for schema in candidates
        if not any(
            other.descriptor_fields < schema.descriptor_fields for other in candidates
        )
    )
    if len(minimal) != 1:
        raise AssertionError(
            "atlas descriptor inhabits "
            f"{len(minimal)} minimal schemas: {[schema.name for schema in minimal]}"
        )
    return minimal[0]


def _descriptor_inhabits_schema(
    descriptor: ParameterDescriptor,
    schema: ParameterSpaceSchema,
) -> bool:
    try:
        schema.validate_descriptor(descriptor)
    except ValueError:
        return False
    return True


def _capability_atlas_gaps() -> tuple[_AtlasGap, ...]:
    return (
        _AtlasGap(
            name=_AtlasText("nonlinear algebraic solve F(x) = 0"),
            region=_AtlasText("nonlinear_root"),
            descriptor=(
                _AtlasText("map_linearity_defect > eps or unavailable"),
                _AtlasText(
                    "residual_target_available = false or target_is_zero = true"
                ),
                _AtlasText(
                    "derivative_oracle_kind in "
                    "{none, matrix, jvp, vjp, jacobian_callback}"
                ),
                _AtlasText("acceptance_relation = residual_below_tolerance"),
                _AtlasText("requested_residual_tolerance = finite"),
            ),
            selected_owner=_AtlasText("none"),
            partial_owners=(
                _AtlasText(
                    "time_integrators._newton.nonlinear_solve is internal stage "
                    "machinery, not a public nonlinear-system solver capability."
                ),
            ),
            required_capability=_AtlasText(
                "NonlinearSolver with descriptor bounds for residual norm, "
                "Jacobian availability, local convergence radius or globalization "
                "policy, line-search or trust-region safeguards, max residual "
                "evaluations, and failure reporting."
            ),
        ),
    )


def _capability_atlas_plot_specs() -> tuple[_AtlasPlotSpec, ...]:
    groups: dict[frozenset[DescriptorField], list[ParameterDescriptor]] = {}
    for descriptor in _capability_atlas_descriptors():
        schema = _atlas_schema_for_descriptor(descriptor)
        groups.setdefault(schema.descriptor_fields, []).append(descriptor)
    return tuple(_AtlasPlotSpec(tuple(group)) for group in groups.values())


def _atlas_axis_range(
    descriptors: tuple[ParameterDescriptor, ...],
    field: _AtlasDescriptorField,
) -> tuple[float, float]:
    values = tuple(
        float(descriptor.coordinate(field).value)
        for descriptor in descriptors
        if descriptor.coordinate(field).known
    )
    assert values
    return (max(1.0, float(int(min(values))) - 2.0), float(int(max(values))) + 1.0)


def _atlas_axis_field(axis: ParameterAxis) -> _AtlasDescriptorField:
    assert isinstance(axis.field, LinearSolverField | DecompositionField)
    return axis.field


def _atlas_fixed_axes(spec: _AtlasPlotSpec) -> tuple[_AtlasDescriptorField, ...]:
    """Return non-plotted schema axes fixed to one known value across the plot."""
    return tuple(
        field
        for field in _atlas_hidden_axes(spec)
        if _axis_has_one_known_value(spec.descriptors, field)
    )


def _atlas_marginalized_axes(spec: _AtlasPlotSpec) -> tuple[_AtlasDescriptorField, ...]:
    """Return non-plotted schema axes not fixed by the descriptor evidence."""
    fixed = set(_atlas_fixed_axes(spec))
    return tuple(field for field in _atlas_hidden_axes(spec) if field not in fixed)


def _atlas_hidden_axes(spec: _AtlasPlotSpec) -> tuple[_AtlasDescriptorField, ...]:
    shown = {spec.x_axis, spec.y_axis}
    return tuple(
        field
        for axis in spec.schema.axes
        if (field := _atlas_axis_field(axis)) not in shown
    )


def _axis_has_one_known_value(
    descriptors: tuple[ParameterDescriptor, ...],
    field: _AtlasDescriptorField,
) -> bool:
    coordinates = tuple(descriptor.coordinate(field) for descriptor in descriptors)
    return (
        all(coordinate.known for coordinate in coordinates)
        and len({coordinate.value for coordinate in coordinates}) == 1
    )


def _descriptor_value(descriptor: ParameterDescriptor, field: DescriptorField) -> str:
    coordinate = descriptor.coordinate(field)
    value = "unknown" if coordinate.value is None else str(coordinate.value)
    if coordinate.evidence != "exact":
        return f"{value} ({coordinate.evidence})"
    return value


def _field_label(field: Any) -> str:
    if isinstance(field, LinearSolverField | DecompositionField):
        return str(field.value)
    return str(field)


def _predicate_label(predicate: Any) -> str:
    if isinstance(predicate, ComparisonPredicate):
        return f"{_field_label(predicate.field)} {predicate.operator} {predicate.value}"
    if isinstance(predicate, AffineComparisonPredicate):
        terms = " + ".join(
            f"{coefficient:g}*{_field_label(field)}"
            for field, coefficient in sorted(
                predicate.terms.items(), key=lambda item: _field_label(item[0])
            )
        )
        if predicate.offset:
            terms = f"{terms} + {predicate.offset:g}"
        return f"{terms} {predicate.operator} {predicate.value:g}"
    if isinstance(predicate, MembershipPredicate):
        values = ", ".join(str(value) for value in sorted(predicate.values, key=str))
        return f"{_field_label(predicate.field)} in {{{values}}}"
    if isinstance(predicate, EvidencePredicate):
        evidence = ", ".join(sorted(predicate.evidence))
        return f"{_field_label(predicate.field)} evidence in {{{evidence}}}"
    return repr(predicate)


def _source_alternatives(
    schema: ParameterSpaceSchema,
    source: _AtlasRegionSource,
    regions: tuple[CoverageRegion, ...] = (),
) -> tuple[tuple[Any, ...], ...]:
    if isinstance(source, DerivedParameterRegion):
        assert source in schema.derived_regions
        return source.alternatives
    if isinstance(source, InvalidCellRule):
        assert source in schema.invalid_cells
        return (source.predicates,)
    assert isinstance(source, CoverageRegion)
    assert source in regions
    schema.validate_coverage_region(source)
    return (source.predicates,)


def _schema_atlas_regions(
    schema: ParameterSpaceSchema,
    regions: tuple[CoverageRegion, ...] = (),
) -> tuple[_AtlasRegionSource, ...]:
    """Return every schema region that should appear in atlas projections."""
    return (*schema.derived_regions, *schema.invalid_cells, *regions)


def _atlas_regions_for_schema(
    schema: ParameterSpaceSchema,
) -> tuple[CoverageRegion, ...]:
    return tuple(
        region
        for region in _capability_atlas_coverage_regions()
        if _region_inhabits_schema(region, schema)
    )


def _region_inhabits_schema(
    region: CoverageRegion,
    schema: ParameterSpaceSchema,
) -> bool:
    try:
        schema.validate_coverage_region(region)
    except ValueError:
        return False
    return True


def _coverage_region_name(region: CoverageRegion) -> str:
    return region.owner.__name__


def _predicate_affine_projection(
    predicate: Any, x_axis: _AtlasDescriptorField, y_axis: _AtlasDescriptorField
) -> tuple[dict[_AtlasDescriptorField, float], str, float] | None:
    if isinstance(predicate, AffineComparisonPredicate):
        visible = {
            field: coefficient
            for field, coefficient in predicate.terms.items()
            if field in {x_axis, y_axis}
        }
        hidden = set(predicate.terms) - {x_axis, y_axis}
        if visible and not hidden:
            return (
                {
                    x_axis: visible.get(x_axis, 0.0),
                    y_axis: visible.get(y_axis, 0.0),
                },
                predicate.operator,
                predicate.value - predicate.offset,
            )
        return None
    if isinstance(predicate, ComparisonPredicate) and predicate.field in {
        x_axis,
        y_axis,
    }:
        field = predicate.field
        return (
            {
                x_axis: 1.0 if field == x_axis else 0.0,
                y_axis: 1.0 if field == y_axis else 0.0,
            },
            predicate.operator,
            float(predicate.value),
        )
    return None


def _affine_value(
    point: tuple[float, float],
    terms: dict[_AtlasDescriptorField, float],
    x_axis: _AtlasDescriptorField,
    y_axis: _AtlasDescriptorField,
    value: float,
) -> float:
    x, y = point
    return terms.get(x_axis, 0.0) * x + terms.get(y_axis, 0.0) * y - value


def _clip_polygon_to_half_plane(
    polygon: tuple[tuple[float, float], ...],
    terms: dict[_AtlasDescriptorField, float],
    operator: str,
    value: float,
    x_axis: _AtlasDescriptorField,
    y_axis: _AtlasDescriptorField,
) -> tuple[tuple[float, float], ...]:
    def inside(point: tuple[float, float]) -> bool:
        signed = _affine_value(point, terms, x_axis, y_axis, value)
        if operator in {">", ">="}:
            return signed >= -1.0e-12
        if operator in {"<", "<="}:
            return signed <= 1.0e-12
        raise AssertionError(f"unsupported half-plane operator {operator!r}")

    def intersection(
        start: tuple[float, float], end: tuple[float, float]
    ) -> tuple[float, float]:
        start_value = _affine_value(start, terms, x_axis, y_axis, value)
        end_value = _affine_value(end, terms, x_axis, y_axis, value)
        if abs(start_value - end_value) <= 1.0e-12:
            return end
        fraction = start_value / (start_value - end_value)
        return (
            start[0] + fraction * (end[0] - start[0]),
            start[1] + fraction * (end[1] - start[1]),
        )

    clipped: list[tuple[float, float]] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        start_inside = inside(start)
        end_inside = inside(end)
        if start_inside and end_inside:
            clipped.append(end)
        elif start_inside and not end_inside:
            clipped.append(intersection(start, end))
        elif not start_inside and end_inside:
            clipped.append(intersection(start, end))
            clipped.append(end)
    return tuple(clipped)


def _affine_equality_line(
    terms: dict[_AtlasDescriptorField, float],
    value: float,
    x_axis: _AtlasDescriptorField,
    y_axis: _AtlasDescriptorField,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_coefficient = terms.get(x_axis, 0.0)
    y_coefficient = terms.get(y_axis, 0.0)
    points: list[tuple[float, float]] = []

    if abs(y_coefficient) > 1.0e-12:
        for x_value in (x_min, x_max):
            y_value = (value - x_coefficient * x_value) / y_coefficient
            if y_min - 1.0e-12 <= y_value <= y_max + 1.0e-12:
                points.append((x_value, y_value))
    if abs(x_coefficient) > 1.0e-12:
        for y_value in (y_min, y_max):
            x_value = (value - y_coefficient * y_value) / x_coefficient
            if x_min - 1.0e-12 <= x_value <= x_max + 1.0e-12:
                points.append((x_value, y_value))

    deduplicated: list[tuple[float, float]] = []
    for point in points:
        rounded = (round(point[0], 12), round(point[1], 12))
        if rounded not in deduplicated:
            deduplicated.append(rounded)
    return tuple(deduplicated[:2])


def _project_alternative_geometry(
    predicates: tuple[Any, ...],
    *,
    x_axis: _AtlasDescriptorField,
    y_axis: _AtlasDescriptorField,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[tuple[_AtlasGeometryKind, tuple[tuple[float, float], ...]], ...]:
    rectangle = (
        (x_range[0], y_range[0]),
        (x_range[1], y_range[0]),
        (x_range[1], y_range[1]),
        (x_range[0], y_range[1]),
    )
    projected = [
        projection
        for predicate in predicates
        if (projection := _predicate_affine_projection(predicate, x_axis, y_axis))
        is not None
    ]
    if not projected:
        return (
            (
                _AtlasGeometryKind.RECTANGLE,
                ((x_range[0], y_range[0]), (x_range[1], y_range[1])),
            ),
        )

    equality = [projection for projection in projected if projection[1] == "=="]
    inequalities = [
        projection
        for projection in projected
        if projection[1] in {">", ">=", "<", "<="}
    ]
    not_equal = [projection for projection in projected if projection[1] == "!="]

    if equality:
        terms, _operator, value = equality[0]
        line = _affine_equality_line(terms, value, x_axis, y_axis, x_range, y_range)
        return ((_AtlasGeometryKind.LINE, line),) if len(line) == 2 else ()

    polygons: list[tuple[tuple[float, float], ...]] = [rectangle]
    for terms, operator, value in inequalities:
        polygons = [
            clipped
            for polygon in polygons
            if (
                clipped := _clip_polygon_to_half_plane(
                    polygon, terms, operator, value, x_axis, y_axis
                )
            )
        ]

    for terms, _operator, value in not_equal:
        split_polygons: list[tuple[tuple[float, float], ...]] = []
        for polygon in polygons:
            for operator in (">", "<"):
                clipped = _clip_polygon_to_half_plane(
                    polygon, terms, operator, value, x_axis, y_axis
                )
                if clipped:
                    split_polygons.append(clipped)
        polygons = split_polygons

    return tuple(
        (_AtlasGeometryKind.POLYGON, polygon)
        for polygon in polygons
        if len(polygon) >= 3
    )


def _projected_region_shapes(spec: _AtlasPlotSpec) -> tuple[_AtlasRegionShape, ...]:
    schema = spec.schema
    regions = _atlas_regions_for_schema(schema)
    shapes: list[_AtlasRegionShape] = []
    for source in _schema_atlas_regions(schema, regions):
        alternatives = _source_alternatives(schema, source, regions)
        for alternative_index, predicates in enumerate(alternatives, start=1):
            geometry = _project_alternative_geometry(
                predicates,
                x_axis=spec.x_axis,
                y_axis=spec.y_axis,
                x_range=spec.x_range,
                y_range=spec.y_range,
            )
            if not geometry:
                raise AssertionError(
                    f"atlas region {_atlas_source_label(source)!r} has no "
                    "visible projection "
                    f"onto {spec.x_axis!r}/{spec.y_axis!r}"
                )
            for geometry_name, points in geometry:
                shapes.append(
                    _AtlasRegionShape(
                        source,
                        predicates,
                        alternative_index,
                        len(alternatives),
                        geometry_name,
                        points,
                    )
                )
    return tuple(shapes)


def _capability_atlas_model_objects() -> tuple[object, ...]:
    specs = _capability_atlas_plot_specs()
    shapes = tuple(shape for spec in specs for shape in _projected_region_shapes(spec))
    return (*specs, *_capability_atlas_gaps(), *shapes)


def _capability_atlas_semantic_model_objects() -> tuple[object, ...]:
    return tuple(
        model
        for model in _capability_atlas_model_objects()
        if not isinstance(model, _AtlasGap)
    )


def _capability_atlas_model_classes() -> frozenset[type]:
    return frozenset(type(model) for model in _capability_atlas_model_objects())


def _capability_atlas_semantic_model_classes() -> frozenset[type]:
    return frozenset(
        type(model) for model in _capability_atlas_semantic_model_objects()
    )


def _atlas_source_label(
    source: DerivedParameterRegion | InvalidCellRule | CoverageRegion,
) -> _AtlasText:
    if isinstance(source, CoverageRegion):
        return _AtlasText(_coverage_region_name(source))
    return _AtlasText(source.name)


def _atlas_source_status(
    source: DerivedParameterRegion | InvalidCellRule | CoverageRegion,
) -> _AtlasText:
    if isinstance(source, InvalidCellRule):
        return _AtlasText("invalid")
    if isinstance(source, CoverageRegion):
        return _AtlasText("owned")
    return _AtlasText("uncovered")


class _CapabilityAtlasDocClaim(Claim[None]):
    """Claim: capability atlas documentation can be generated."""

    @property
    def description(self) -> str:
        return "algorithm_capabilities/capability_atlas_doc_generates"

    def check(self, _calibration: None) -> None:
        self._assert_atlas_models_do_not_store_raw_text()
        self._assert_semantic_atlas_models_do_not_store_presentation_text()
        self._assert_atlas_dataclasses_are_not_trivial_wrappers()
        self._assert_projection_axis_roles_are_derived()
        self._assert_evidence_schema_is_derived()
        self._assert_evidence_is_descriptors()
        self._assert_coverage_regions_are_schema_discovered()
        self._assert_plot_specs_select_descriptors()
        for spec in _capability_atlas_plot_specs():
            schema = spec.schema
            regions = _atlas_regions_for_schema(schema)
            discovered = _schema_atlas_regions(schema, regions)
            assert {id(source) for source in discovered} == {
                id(source)
                for source in schema.derived_regions + schema.invalid_cells + regions
            }
            shapes = _projected_region_shapes(spec)
            assert shapes
            for shape in shapes:
                assert _atlas_source_label(shape.source)
                assert shape.points

    @classmethod
    def _assert_atlas_models_do_not_store_raw_text(cls) -> None:
        for atlas_class in _capability_atlas_model_classes():
            for annotation in get_type_hints(atlas_class).values():
                assert not cls._annotation_contains_raw_text(annotation)

    @classmethod
    def _annotation_contains_raw_text(cls, annotation: object) -> bool:
        if annotation is str:
            return True
        return any(
            cls._annotation_contains_raw_text(argument)
            for argument in get_args(annotation)
        )

    @classmethod
    def _assert_semantic_atlas_models_do_not_store_presentation_text(cls) -> None:
        for atlas_class in _capability_atlas_semantic_model_classes():
            for annotation in get_type_hints(atlas_class).values():
                assert not cls._annotation_contains_type(annotation, _AtlasText)

    @classmethod
    def _assert_atlas_dataclasses_are_not_trivial_wrappers(cls) -> None:
        atlas_classes = {
            atlas_class
            for atlas_class in _capability_atlas_model_classes()
            if is_dataclass(atlas_class)
        }
        for atlas_class in atlas_classes:
            if len(fields(atlas_class)) != 1:
                continue
            derived_properties = [
                value
                for value in vars(atlas_class).values()
                if isinstance(value, property)
            ]
            assert derived_properties

    @classmethod
    def _assert_plot_specs_select_descriptors(cls) -> None:
        annotations = get_type_hints(_AtlasPlotSpec)
        assert any(
            cls._annotation_contains_type(annotation, ParameterDescriptor)
            for annotation in annotations.values()
        )
        assert not any(
            cls._annotation_contains_type(annotation, _AtlasText)
            for annotation in annotations.values()
        )
        assert not any(
            cls._annotation_is_text_collection(annotation)
            for annotation in annotations.values()
        )
        specs = _capability_atlas_plot_specs()
        plotted = tuple(descriptor for spec in specs for descriptor in spec.descriptors)
        assert plotted == _capability_atlas_descriptors()
        for spec in specs:
            assert spec.descriptors
            schema = _atlas_schema_for_descriptor(spec.descriptors[0])
            assert all(
                _atlas_schema_for_descriptor(descriptor) == schema
                for descriptor in spec.descriptors
            )
        for left in plotted:
            for right in plotted:
                assert (
                    _atlas_schema_for_descriptor(left)
                    == _atlas_schema_for_descriptor(right)
                ) == (
                    any(
                        left in spec.descriptors and right in spec.descriptors
                        for spec in specs
                    )
                )

    @classmethod
    def _assert_evidence_schema_is_derived(cls) -> None:
        annotations = get_type_hints(_AtlasPlotSpec)
        assert not any(
            cls._annotation_contains_type(annotation, ParameterSpaceSchema)
            for annotation in annotations.values()
        )
        for descriptor in _capability_atlas_descriptors():
            _atlas_schema_for_descriptor(descriptor)

    @classmethod
    def _assert_evidence_is_descriptors(cls) -> None:
        annotations = get_type_hints(_AtlasPlotSpec)
        assert any(
            cls._annotation_contains_type(annotation, ParameterDescriptor)
            for annotation in annotations.values()
        )

    @classmethod
    def _assert_coverage_regions_are_schema_discovered(cls) -> None:
        for schema in _capability_atlas_schemas():
            assert _atlas_regions_for_schema(schema) == tuple(
                region
                for region in _capability_atlas_coverage_regions()
                if _region_inhabits_schema(region, schema)
            )

    @classmethod
    def _assert_projection_axis_roles_are_derived(cls) -> None:
        annotations = get_type_hints(_AtlasPlotSpec)
        assert not any(
            cls._annotation_contains_type(annotation, _AtlasText)
            for annotation in annotations.values()
        )
        assert not any(
            cls._annotation_is_text_collection(annotation)
            for annotation in annotations.values()
        )
        assert not any(
            cls._annotation_contains_type(annotation, _AtlasDescriptorField)
            for annotation in annotations.values()
        )
        for spec in _capability_atlas_plot_specs():
            shown = {spec.x_axis, spec.y_axis}
            fixed = set(_atlas_fixed_axes(spec))
            marginalized = set(_atlas_marginalized_axes(spec))
            assert shown | fixed | marginalized == {
                _atlas_axis_field(axis) for axis in spec.schema.axes
            }
            assert not (shown & fixed)
            assert not (shown & marginalized)
            assert not (fixed & marginalized)

    @classmethod
    def _annotation_contains_type(
        cls, annotation: object, expected_type: object
    ) -> bool:
        if annotation is expected_type:
            return True
        return any(
            cls._annotation_contains_type(argument, expected_type)
            for argument in get_args(annotation)
        )

    @classmethod
    def _annotation_is_text_collection(cls, annotation: object) -> bool:
        return get_origin(annotation) in {tuple, list, set, frozenset} and any(
            argument is _AtlasText or cls._annotation_is_text_collection(argument)
            for argument in get_args(annotation)
        )


# ---------------------------------------------------------------------------
# Auto-discovery and registry
# ---------------------------------------------------------------------------

_MODULES = _discover_modules()
_ABCS = _discover_abcs(_MODULES)
_HIERARCHY_PAIRS = _discover_hierarchy_pairs(_ABCS)
_ITERATIVE_SOLVERS = _discover_concrete_iterative_solvers(_MODULES)
_MATRIX_FREE_ITERATIVE_SOLVERS = _discover_matrix_free_iterative_solvers(_MODULES)
_FACTORIZATIONS = _discover_concrete_factorizations(_MODULES)
_TEST_FILES = sorted(Path(__file__).parent.glob("test_*.py"))

_TimeIntegrationRequest = _resolve_dotted(
    "cosmic_foundry.computation.time_integrators.TimeIntegrationRequest"
)
_DecompositionRequest = _resolve_dotted(
    "cosmic_foundry.computation.decompositions.DecompositionRequest"
)
_DiscreteOperatorRequest = _resolve_dotted(
    "cosmic_foundry.theory.discrete.DiscreteOperatorRequest"
)
_GeometryRequest = _resolve_dotted("cosmic_foundry.geometry.GeometryRequest")
_TIME_INTEGRATOR_OWNERSHIP = _ArchitectureOwnershipSpec(
    package="cosmic_foundry.computation.time_integrators",
    public_categories={
        "capability_contract": frozenset(
            {
                "AlgorithmStructureContract",
                "TimeIntegrationCapability",
                "TimeIntegrationRegistry",
                "TimeIntegrationRequest",
                "select_time_integrator",
                "time_integration_capabilities",
            }
        ),
        "method_family": frozenset(
            {
                "AdditiveRungeKuttaIntegrator",
                "CompositionIntegrator",
                "ExplicitMultistepIntegrator",
                "ImplicitRungeKuttaIntegrator",
                "LawsonRungeKuttaIntegrator",
                "MultistepIntegrator",
                "RungeKuttaIntegrator",
                "SymplecticCompositionIntegrator",
            }
        ),
        "driver_controller": frozenset(
            {
                "AdaptiveNordsieckController",
                "AutoIntegrator",
                "ConstantStep",
                "ConstraintAwareController",
                "Controller",
                "IntegrationDriver",
                "PIController",
                "TimeIntegrator",
            }
        ),
        "policy": frozenset(
            {
                "FamilyName",
                "FamilySwitch",
                "OrderDecision",
                "OrderSelector",
                "StiffnessDiagnostic",
                "StiffnessSwitcher",
            }
        ),
        "rhs": frozenset(
            {
                "BlackBoxRHS",
                "CompositeRHS",
                "CompositeRHSProtocol",
                "FiniteDiffJacobianRHS",
                "HamiltonianRHS",
                "HamiltonianRHSProtocol",
                "JacobianRHS",
                "ReactionNetworkRHS",
                "RHSProtocol",
                "SemilinearRHS",
                "SemilinearRHSProtocol",
                "SplitRHS",
                "SplitRHSProtocol",
                "WithJacobianRHSProtocol",
            }
        ),
        "domain": frozenset(
            {
                "check_state_domain",
                "DomainCheck",
                "DomainViolation",
                "NonnegativeStateDomain",
                "predict_domain_step_limit",
                "StateDomain",
            }
        ),
        "state_result": frozenset(
            {
                "IntegrationSelectionResult",
                "NordsieckHistory",
                "ODEState",
                "PhiFunction",
            }
        ),
        "verification_helper": frozenset(
            {
                "elementary_weight",
                "gamma",
                "nonlinear_solve",
                "order",
                "project_conserved",
                "sigma",
                "solve_nse",
                "stability_function",
                "Tree",
                "trees_up_to_order",
            }
        ),
    },
    forbidden_public_symbols=frozenset(
        {
            "FamilySwitchingNordsieckIntegrator",
            "Integrator",
            "IntegratorSelectionResult",
            "VODEController",
            "VariableOrderNordsieckIntegrator",
        }
    ),
    capability_provider=(
        "cosmic_foundry.computation.time_integrators.time_integration_capabilities"
    ),
    request_selector="cosmic_foundry.computation.time_integrators.select_time_integrator",
    request_expectations=(
        _CapabilityRequestExpectation(
            _TimeIntegrationRequest(
                available_structure=frozenset({"plain_rhs"}),
                requested_properties=frozenset({"one_step", "explicit", "runge_kutta"}),
                order=4,
            ),
            "RungeKuttaIntegrator",
        ),
        _CapabilityRequestExpectation(
            _TimeIntegrationRequest(
                available_structure=frozenset({"jacobian_rhs"}),
                requested_properties=frozenset({"one_step", "implicit"}),
                order=2,
            ),
            "ImplicitRungeKuttaIntegrator",
        ),
        _CapabilityRequestExpectation(
            _TimeIntegrationRequest(
                available_structure=frozenset({"split_rhs"}),
                requested_properties=frozenset({"one_step", "imex"}),
                order=3,
            ),
            "AdditiveRungeKuttaIntegrator",
        ),
        _CapabilityRequestExpectation(
            _TimeIntegrationRequest(
                available_structure=frozenset({"hamiltonian_rhs"}),
                requested_properties=frozenset(
                    {"one_step", "symplectic", "composition"}
                ),
                order=4,
            ),
            "SymplecticCompositionIntegrator",
        ),
        _CapabilityRequestExpectation(
            _TimeIntegrationRequest(
                available_structure=frozenset({"composite_rhs"}),
                requested_properties=frozenset(
                    {"one_step", "operator_splitting", "composition"}
                ),
                order=4,
            ),
            "CompositionIntegrator",
        ),
        _CapabilityRequestExpectation(
            _TimeIntegrationRequest(
                available_structure=frozenset({"jacobian_rhs", "state_domain"}),
                requested_properties=frozenset(
                    {
                        "advance",
                        "nordsieck",
                        "adaptive_timestep",
                        "variable_order",
                        "stiffness_switching",
                        "domain_aware_acceptance",
                    }
                ),
                order=2,
            ),
            "AdaptiveNordsieckController",
        ),
        _CapabilityRequestExpectation(
            _TimeIntegrationRequest(
                available_structure=frozenset(
                    {"plain_rhs", "time_integrator", "controller"}
                ),
                requested_properties=frozenset({"advance", "adaptive_timestep"}),
                order=3,
            ),
            "IntegrationDriver",
        ),
        _CapabilityRequestExpectation(
            _TimeIntegrationRequest(
                available_structure=frozenset(
                    {"reaction_network_rhs", "conservation_constraints"}
                ),
                requested_properties=frozenset({"advance", "constraint_lifecycle"}),
                order=2,
            ),
            "ConstraintAwareController",
        ),
    ),
    rejected_requests=(
        _CapabilityRejectionExpectation(
            _TimeIntegrationRequest(
                available_structure=frozenset({"hamiltonian_rhs"}),
                requested_properties=frozenset(
                    {"one_step", "symplectic", "composition"}
                ),
                order=3,
            )
        ),
        _CapabilityRejectionExpectation(
            _TimeIntegrationRequest(
                available_structure=frozenset({"composite_rhs"}),
                requested_properties=frozenset(
                    {"one_step", "operator_splitting", "composition"}
                ),
                order=3,
            )
        ),
    ),
    expected_class_modules={
        "AdaptiveNordsieckController": "adaptive_nordsieck",
        "AdditiveRungeKuttaIntegrator": "imex",
        "AlgorithmStructureContract": "algorithm_capabilities",
        "AutoIntegrator": "auto",
        "BlackBoxRHS": "integrator",
        "CompositeRHS": "splitting",
        "CompositionIntegrator": "splitting",
        "ConstraintAwareController": "constraint_aware",
        "ConstantStep": "integrator",
        "DomainCheck": "domains",
        "DomainViolation": "domains",
        "ExplicitMultistepIntegrator": "explicit_multistep",
        "FiniteDiffJacobianRHS": "implicit",
        "HamiltonianRHS": "symplectic",
        "ImplicitRungeKuttaIntegrator": "implicit",
        "IntegrationDriver": "integration_driver",
        "IntegrationSelectionResult": "integration_driver",
        "JacobianRHS": "implicit",
        "LawsonRungeKuttaIntegrator": "exponential",
        "MultistepIntegrator": "nordsieck",
        "NonnegativeStateDomain": "domains",
        "NordsieckHistory": "nordsieck",
        "ODEState": "integrator",
        "OrderDecision": "variable_order",
        "OrderSelector": "variable_order",
        "PhiFunction": "exponential",
        "PIController": "integrator",
        "ReactionNetworkRHS": "reaction_network",
        "RungeKuttaIntegrator": "runge_kutta",
        "SemilinearRHS": "exponential",
        "SplitRHS": "imex",
        "StiffnessDiagnostic": "stiffness",
        "StiffnessSwitcher": "stiffness",
        "SymplecticCompositionIntegrator": "symplectic",
        "TimeIntegrationCapability": "algorithm_capabilities",
        "TimeIntegrationRegistry": "algorithm_capabilities",
        "TimeIntegrationRequest": "algorithm_capabilities",
    },
    required_name_fragments={
        "AdaptiveNordsieckController": ("Adaptive", "Nordsieck", "Controller"),
        "ConstraintAwareController": ("Constraint", "Controller"),
        "IntegrationDriver": ("Integration", "Driver"),
        "OrderSelector": ("Order", "Selector"),
        "StiffnessSwitcher": ("Stiffness", "Switcher"),
    },
)

_LINEAR_SOLVER_OWNERSHIP = _ArchitectureOwnershipSpec(
    package="cosmic_foundry.computation.solvers",
    public_categories={
        "capability_contract": frozenset(
            {
                "LINEAR_SOLVER_COVERAGE_REGIONS",
                "linear_solver_coverage_regions",
                "select_linear_solver_for_descriptor",
            }
        ),
        "abstract_interface": frozenset(
            {
                "DirectSolver",
                "IterativeSolver",
                "KrylovSolver",
                "LinearOperator",
                "LinearSolver",
                "StationaryIterationSolver",
            }
        ),
        "direct_solver": frozenset(
            {
                "DenseLUSolver",
                "DenseSVDSolver",
            }
        ),
        "iterative_solver": frozenset(
            {
                "DenseCGSolver",
                "DenseGMRESSolver",
                "DenseGaussSeidelSolver",
                "DenseJacobiSolver",
            }
        ),
    },
    capability_provider="cosmic_foundry.computation.solvers.linear_solver_coverage_regions",
    expected_class_modules={
        "DenseCGSolver": "dense_cg_solver",
        "DenseGMRESSolver": "dense_gmres_solver",
        "DenseGaussSeidelSolver": "dense_gauss_seidel_solver",
        "DenseJacobiSolver": "dense_jacobi_solver",
        "DenseLUSolver": "dense_lu_solver",
        "DenseSVDSolver": "dense_svd_solver",
        "DirectSolver": "direct_solver",
        "IterativeSolver": "iterative_solver",
        "KrylovSolver": "iterative_solver",
        "LinearOperator": "linear_solver",
        "LinearSolver": "linear_solver",
        "StationaryIterationSolver": "iterative_solver",
    },
    required_name_fragments={
        "DenseCGSolver": ("Dense", "Solver"),
        "DenseGMRESSolver": ("Dense", "Solver"),
        "DenseGaussSeidelSolver": ("Dense", "Solver"),
        "DenseJacobiSolver": ("Dense", "Solver"),
        "DenseLUSolver": ("Dense", "Solver"),
        "DenseSVDSolver": ("Dense", "Solver"),
    },
)

_DECOMPOSITION_OWNERSHIP = _ArchitectureOwnershipSpec(
    package="cosmic_foundry.computation.decompositions",
    public_categories={
        "capability_contract": frozenset(
            {
                "DecompositionCapability",
                "decomposition_capabilities",
                "DecompositionRegistry",
                "DecompositionRequest",
                "select_decomposition",
            }
        ),
        "abstract_interface": frozenset(
            {
                "DecomposedTensor",
                "Decomposition",
                "Factorization",
            }
        ),
        "factorization": frozenset(
            {
                "LUFactorization",
                "SVDFactorization",
            }
        ),
        "decomposed_result": frozenset(
            {
                "LUDecomposedTensor",
                "SVDDecomposedTensor",
            }
        ),
    },
    capability_provider=(
        "cosmic_foundry.computation.decompositions.decomposition_capabilities"
    ),
    request_selector="cosmic_foundry.computation.decompositions.select_decomposition",
    request_expectations=(
        _CapabilityRequestExpectation(
            _DecompositionRequest(
                available_structure=frozenset(
                    {"dense_matrix", "square_matrix", "full_rank"}
                ),
                requested_properties=frozenset(
                    {"decompose", "factorize", "direct_solve", "exact"}
                ),
            ),
            "LUFactorization",
        ),
        _CapabilityRequestExpectation(
            _DecompositionRequest(
                available_structure=frozenset({"dense_matrix"}),
                requested_properties=frozenset(
                    {
                        "decompose",
                        "factorize",
                        "direct_solve",
                        "minimum_norm",
                        "rank_deficient",
                    }
                ),
            ),
            "SVDFactorization",
        ),
    ),
    rejected_requests=(
        _CapabilityRejectionExpectation(
            _DecompositionRequest(
                available_structure=frozenset({"dense_matrix"}),
                requested_properties=frozenset({"decompose", "factorize", "exact"}),
            )
        ),
    ),
    expected_class_modules={
        "DecomposedTensor": "decomposition",
        "Decomposition": "decomposition",
        "DecompositionCapability": "algorithm_capabilities",
        "DecompositionRegistry": "algorithm_capabilities",
        "DecompositionRequest": "algorithm_capabilities",
        "Factorization": "factorization",
        "LUDecomposedTensor": "lu_factorization",
        "LUFactorization": "lu_factorization",
        "SVDDecomposedTensor": "svd_factorization",
        "SVDFactorization": "svd_factorization",
    },
    required_name_fragments={
        "LUDecomposedTensor": ("Decomposed", "Tensor"),
        "LUFactorization": ("Factorization",),
        "SVDDecomposedTensor": ("Decomposed", "Tensor"),
        "SVDFactorization": ("Factorization",),
    },
)

_DISCRETE_OPERATOR_OWNERSHIP = _ArchitectureOwnershipSpec(
    package="cosmic_foundry.theory.discrete",
    public_categories={
        "capability_contract": frozenset(
            {
                "DiscreteOperatorCapability",
                "discrete_operator_capabilities",
                "DiscreteOperatorRegistry",
                "DiscreteOperatorRequest",
                "select_discrete_operator",
            }
        ),
        "mesh_topology": frozenset(
            {
                "CellComplex",
                "Mesh",
                "StructuredMesh",
            }
        ),
        "field": frozenset(
            {
                "DiscreteField",
                "EdgeField",
                "FaceField",
                "PointField",
                "VolumeField",
            }
        ),
        "boundary_condition": frozenset(
            {
                "DirichletGhostCells",
                "DiscreteBoundaryCondition",
                "InhomogeneousDirichletGhostCells",
                "NeumannGhostCells",
                "PeriodicGhostCells",
                "ZeroGhostCells",
            }
        ),
        "operator_interface": frozenset(
            {
                "DiscreteExteriorDerivative",
                "DiscreteOperator",
                "Discretization",
                "NumericalFlux",
                "RestrictionOperator",
            }
        ),
        "numerical_flux": frozenset(
            {
                "AdvectionDiffusionFlux",
                "AdvectiveFlux",
                "DiffusiveFlux",
            }
        ),
        "discretization": frozenset(
            {
                "DivergenceFormDiscretization",
            }
        ),
    },
    capability_provider="cosmic_foundry.theory.discrete.discrete_operator_capabilities",
    request_selector="cosmic_foundry.theory.discrete.select_discrete_operator",
    request_expectations=(
        _CapabilityRequestExpectation(
            _DiscreteOperatorRequest(
                available_structure=frozenset(
                    {"cartesian_mesh", "cell_average_field", "smooth_scalar_field"}
                ),
                requested_properties=frozenset(
                    {"numerical_flux", "advective", "centered_stencil"}
                ),
                order=4,
            ),
            "AdvectiveFlux",
        ),
        _CapabilityRequestExpectation(
            _DiscreteOperatorRequest(
                available_structure=frozenset(
                    {"cartesian_mesh", "cell_average_field", "smooth_scalar_field"}
                ),
                requested_properties=frozenset(
                    {"numerical_flux", "diffusive", "antisymmetric_stencil"}
                ),
                order=4,
            ),
            "DiffusiveFlux",
        ),
        _CapabilityRequestExpectation(
            _DiscreteOperatorRequest(
                available_structure=frozenset(
                    {"cartesian_mesh", "cell_average_field", "smooth_scalar_field"}
                ),
                requested_properties=frozenset(
                    {"numerical_flux", "advective", "diffusive"}
                ),
                order=4,
            ),
            "AdvectionDiffusionFlux",
        ),
        _CapabilityRequestExpectation(
            _DiscreteOperatorRequest(
                available_structure=frozenset(
                    {
                        "cartesian_mesh",
                        "numerical_flux",
                        "discrete_boundary_condition",
                    }
                ),
                requested_properties=frozenset(
                    {"discrete_operator", "divergence_form", "boundary_aware"}
                ),
            ),
            "DivergenceFormDiscretization",
        ),
    ),
    rejected_requests=(
        _CapabilityRejectionExpectation(
            _DiscreteOperatorRequest(
                available_structure=frozenset(
                    {"cartesian_mesh", "cell_average_field", "smooth_scalar_field"}
                ),
                requested_properties=frozenset(
                    {"numerical_flux", "diffusive", "antisymmetric_stencil"}
                ),
                order=3,
            )
        ),
    ),
    expected_class_modules={
        "AdvectionDiffusionFlux": "advection_diffusion_flux",
        "AdvectiveFlux": "advective_flux",
        "CellComplex": "cell_complex",
        "DiffusiveFlux": "diffusive_flux",
        "DirichletGhostCells": "discrete_boundary_condition",
        "DiscreteBoundaryCondition": "discrete_boundary_condition",
        "DiscreteExteriorDerivative": "discrete_exterior_derivative",
        "DiscreteField": "discrete_field",
        "DiscreteOperator": "discrete_operator",
        "DiscreteOperatorCapability": "algorithm_capabilities",
        "DiscreteOperatorRegistry": "algorithm_capabilities",
        "DiscreteOperatorRequest": "algorithm_capabilities",
        "Discretization": "discretization",
        "DivergenceFormDiscretization": "divergence_form_discretization",
        "EdgeField": "edge_field",
        "FaceField": "face_field",
        "InhomogeneousDirichletGhostCells": "discrete_boundary_condition",
        "Mesh": "mesh",
        "NeumannGhostCells": "discrete_boundary_condition",
        "NumericalFlux": "numerical_flux",
        "PeriodicGhostCells": "discrete_boundary_condition",
        "PointField": "point_field",
        "RestrictionOperator": "restriction_operator",
        "StructuredMesh": "structured_mesh",
        "VolumeField": "volume_field",
        "ZeroGhostCells": "discrete_boundary_condition",
    },
    required_name_fragments={
        "AdvectionDiffusionFlux": ("Advection", "Diffusion", "Flux"),
        "AdvectiveFlux": ("Flux",),
        "DiffusiveFlux": ("Flux",),
        "DivergenceFormDiscretization": ("Divergence", "Discretization"),
    },
)

_GEOMETRY_OWNERSHIP = _ArchitectureOwnershipSpec(
    package="cosmic_foundry.geometry",
    public_categories={
        "capability_contract": frozenset(
            {
                "GeometryCapability",
                "geometry_capabilities",
                "GeometryRegistry",
                "GeometryRequest",
                "select_geometry",
            }
        ),
        "manifold": frozenset(
            {
                "EuclideanManifold",
                "SchwarzschildManifold",
            }
        ),
        "chart": frozenset(
            {
                "CartesianChart",
            }
        ),
        "mesh": frozenset(
            {
                "CartesianMesh",
            }
        ),
        "restriction_operator": frozenset(
            {
                "CartesianEdgeRestriction",
                "CartesianFaceRestriction",
                "CartesianPointRestriction",
                "CartesianRestrictionOperator",
                "CartesianVolumeRestriction",
            }
        ),
        "discrete_geometry_operator": frozenset(
            {
                "CartesianExteriorDerivative",
            }
        ),
    },
    capability_provider="cosmic_foundry.geometry.geometry_capabilities",
    request_selector="cosmic_foundry.geometry.select_geometry",
    request_expectations=(
        _CapabilityRequestExpectation(
            _GeometryRequest(
                available_structure=frozenset({"dimension"}),
                requested_properties=frozenset({"manifold", "flat_metric"}),
            ),
            "EuclideanManifold",
        ),
        _CapabilityRequestExpectation(
            _GeometryRequest(
                available_structure=frozenset({"central_mass_symbol"}),
                requested_properties=frozenset(
                    {"manifold", "lorentzian_metric", "schwarzschild_geometry"}
                ),
            ),
            "SchwarzschildManifold",
        ),
        _CapabilityRequestExpectation(
            _GeometryRequest(
                available_structure=frozenset(
                    {"euclidean_manifold", "origin", "spacing", "shape"}
                ),
                requested_properties=frozenset({"mesh", "cartesian_mesh"}),
            ),
            "CartesianMesh",
        ),
        _CapabilityRequestExpectation(
            _GeometryRequest(
                available_structure=frozenset(
                    {"cartesian_mesh", "discrete_field", "form_degree"}
                ),
                requested_properties=frozenset(
                    {"discrete_exterior_derivative", "chain_map", "exact_stokes"}
                ),
            ),
            "CartesianExteriorDerivative",
        ),
        _CapabilityRequestExpectation(
            _GeometryRequest(
                available_structure=frozenset({"cartesian_mesh", "zero_form"}),
                requested_properties=frozenset({"restriction", "volume_field"}),
            ),
            "CartesianVolumeRestriction",
        ),
        _CapabilityRequestExpectation(
            _GeometryRequest(
                available_structure=frozenset({"cartesian_mesh", "zero_form"}),
                requested_properties=frozenset({"restriction", "point_field"}),
            ),
            "CartesianPointRestriction",
        ),
    ),
    rejected_requests=(
        _CapabilityRejectionExpectation(
            _GeometryRequest(
                available_structure=frozenset({"cartesian_mesh", "zero_form"}),
                requested_properties=frozenset({"restriction", "edge_field"}),
            )
        ),
    ),
    expected_class_modules={
        "CartesianChart": "euclidean_manifold",
        "CartesianEdgeRestriction": "cartesian_restriction_operator",
        "CartesianExteriorDerivative": "cartesian_exterior_derivative",
        "CartesianFaceRestriction": "cartesian_restriction_operator",
        "CartesianMesh": "cartesian_mesh",
        "CartesianPointRestriction": "cartesian_restriction_operator",
        "CartesianRestrictionOperator": "cartesian_restriction_operator",
        "CartesianVolumeRestriction": "cartesian_restriction_operator",
        "EuclideanManifold": "euclidean_manifold",
        "GeometryCapability": "algorithm_capabilities",
        "GeometryRegistry": "algorithm_capabilities",
        "GeometryRequest": "algorithm_capabilities",
        "SchwarzschildManifold": "schwarzschild_manifold",
    },
    required_name_fragments={
        "CartesianExteriorDerivative": ("Cartesian", "Derivative"),
        "CartesianMesh": ("Cartesian", "Mesh"),
        "CartesianRestrictionOperator": ("Cartesian", "Restriction"),
        "EuclideanManifold": ("Manifold",),
        "SchwarzschildManifold": ("Manifold",),
    },
)

_CLAIMS: list[Claim[None]] = [
    *[_AbcInstantiationClaim(cls) for cls in _ABCS],
    *[_HierarchyClaim(child, parent) for child, parent in _HIERARCHY_PAIRS],
    *[_ModuleAllClaim(mod_path, mod) for mod_path, mod in _MODULES],
    *[_IterativeSolverJitClaim(cls) for cls in _MATRIX_FREE_ITERATIVE_SOLVERS],
    *[_MaterializationGateClaim(cls) for cls in _MATRIX_FREE_ITERATIVE_SOLVERS],
    *[_FactorizationJitClaim(cls) for cls in _FACTORIZATIONS],
    _GenericBasesClaim(),
    _ManifoldIsolationClaim(),
    *[
        _ImportBoundaryClaim(path)
        for pkg_dir in _PURE_PACKAGES
        for path in sorted(pkg_dir.rglob("*.py"))
    ],
    *[_ParametrizeEnforcementClaim(p) for p in _TEST_FILES],
    *[_BodyDispatchClaim(p) for p in _TEST_FILES],
    *[_TestAxisConventionClaim(p) for p in _TEST_FILES],
    *[_NoTopLevelDefaultBackendMutationClaim(p) for p in _TEST_FILES],
    _ArchitectureOwnershipClaim(_TIME_INTEGRATOR_OWNERSHIP),
    _ArchitectureOwnershipClaim(_LINEAR_SOLVER_OWNERSHIP),
    _ArchitectureOwnershipClaim(_DECOMPOSITION_OWNERSHIP),
    _ArchitectureOwnershipClaim(_DISCRETE_OPERATOR_OWNERSHIP),
    _ArchitectureOwnershipClaim(_GEOMETRY_OWNERSHIP),
    _AutoDiscoveryImportClaim(),
    _ParameterSpaceSchemaClaim(),
    _SolveRelationSchemaClaim(),
    _LinearOperatorDescriptorClaim(),
    _LinearSolverCoverageLocalityClaim(),
    _LinearSolverCoverageRegionClaim(),
    _CapabilityAtlasDocClaim(),
]


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_structure(claim: Claim[None]) -> None:
    claim.check(None)
