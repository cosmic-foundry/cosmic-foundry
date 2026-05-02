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
  _TestAxisConventionClaim  — module tests use correctness/convergence/performance
  _NoTopLevelDefaultBackendMutationClaim — tests do not mutate Tensor backend at import
"""

from __future__ import annotations

import ast
import html
import importlib
import inspect
import pkgutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from cosmic_foundry.computation.algorithm_capabilities import (
    AffineComparisonPredicate,
    ComparisonPredicate,
    CoveragePatch,
    DerivedParameterRegion,
    DescriptorCoordinate,
    InvalidCellRule,
    MembershipPredicate,
    NumericInterval,
    ParameterAxis,
    ParameterBin,
    ParameterDescriptor,
    ParameterSpaceSchema,
    decomposition_parameter_schema,
    linear_solver_parameter_schema,
    solve_relation_parameter_schema,
)
from cosmic_foundry.computation.backends.python_backend import PythonBackend
from cosmic_foundry.computation.decompositions.factorization import Factorization
from cosmic_foundry.computation.solvers.iterative_solver import IterativeSolver
from cosmic_foundry.computation.tensor import MaterializationError, Tensor
from cosmic_foundry.theory.continuous.manifold import Manifold
from cosmic_foundry.theory.foundation.indexed_set import IndexedSet
from tests.claims import Claim

_PROJECT_ROOT = Path(__file__).parent.parent
_PACKAGE_ROOT = _PROJECT_ROOT / "cosmic_foundry"
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
        implementations = [cap.implementation for cap in capabilities]
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
            assert selected.implementation == expectation.selected_implementation, (
                f"{expectation.request!r} selected {selected.implementation}, "
                f"expected {expectation.selected_implementation}"
            )
        for expectation in self._spec.rejected_requests:
            try:
                selected = selector(expectation.request)
            except ValueError:
                continue
            raise AssertionError(
                f"{expectation.request!r} unexpectedly selected "
                f"{selected.implementation}"
            )

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
        owned_patch = CoveragePatch(
            name="well_conditioned_spd",
            owner="DenseCGSolver",
            status="owned",
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
        rejected_patch = CoveragePatch(
            name="nonlinear_root_not_yet_public",
            owner="missing-public-nonlinear-solver",
            status="rejected",
            predicates=(
                ComparisonPredicate("map_linearity_defect", ">", 1.0e-12),
                MembershipPredicate(
                    "operator_representation",
                    frozenset({"matrix_free"}),
                ),
            ),
        )
        patches = (owned_patch, rejected_patch)

        assert schema.cell_status(self._descriptor(), patches) == "owned"
        assert schema.cell_status(self._descriptor(dim_x=5), patches) == "invalid"
        assert (
            schema.cell_status(
                self._descriptor(
                    map_linearity_defect=2.0e-12,
                    coercivity_lower_bound=0.0,
                    operator_representation="matrix_free",
                ),
                patches,
            )
            == "rejected"
        )
        unknown_condition = self._descriptor(condition_estimate=None)
        assert schema.cell_status(unknown_condition, patches) == "uncovered"

        with pytest.raises(ValueError):
            ParameterSpaceSchema(
                name="empty_axis",
                axes=(ParameterAxis("unbinned_axis", ()),),
            ).validate_schema()
        with pytest.raises(ValueError):
            ParameterSpaceSchema(
                name="duplicate_fields",
                axes=(
                    ParameterAxis(
                        "first",
                        (NumericInterval("all", lower=0.0),),
                        descriptor_field="same_field",
                    ),
                    ParameterAxis(
                        "second",
                        (NumericInterval("all", lower=0.0),),
                        descriptor_field="same_field",
                    ),
                ),
            ).validate_schema()
        with pytest.raises(ValueError):
            schema.validate_coverage_patch(
                CoveragePatch(
                    name="undeclared_axis",
                    owner="BadSolver",
                    status="owned",
                    predicates=(ComparisonPredicate("undeclared", "==", 1),),
                )
            )
        with pytest.raises(TypeError):
            schema.validate_coverage_patch(
                CoveragePatch(
                    name="unsupported_predicate",
                    owner="BadSolver",
                    status="owned",
                    predicates=(object(),),  # type: ignore[arg-type]
                )
            )

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
            schema="demo_solve_relation",
            coordinates={
                "map_linearity_defect": DescriptorCoordinate(map_linearity_defect),
                "dim_x": DescriptorCoordinate(dim_x),
                "dim_y": DescriptorCoordinate(dim_y),
                "coercivity_lower_bound": DescriptorCoordinate(coercivity_lower_bound),
                "condition_estimate": DescriptorCoordinate(condition_estimate),
                "operator_representation": DescriptorCoordinate(
                    operator_representation
                ),
            },
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
        schema: str = "solve_relation",
    ) -> ParameterDescriptor:
        return ParameterDescriptor(
            schema=schema,
            coordinates={
                "dim_x": DescriptorCoordinate(dim_x),
                "dim_y": DescriptorCoordinate(dim_y),
                "auxiliary_scalar_count": DescriptorCoordinate(auxiliary_scalar_count),
                "equality_constraint_count": DescriptorCoordinate(
                    equality_constraint_count
                ),
                "normalization_constraint_count": DescriptorCoordinate(
                    normalization_constraint_count
                ),
                "residual_target_available": DescriptorCoordinate(
                    residual_target_available
                ),
                "target_is_zero": DescriptorCoordinate(target_is_zero),
                "map_linearity_defect": DescriptorCoordinate(
                    map_linearity_defect,
                    evidence=map_linearity_evidence,  # type: ignore[arg-type]
                ),
                "matrix_representation_available": DescriptorCoordinate(
                    matrix_representation_available
                ),
                "operator_application_available": DescriptorCoordinate(
                    operator_application_available
                ),
                "derivative_oracle_kind": DescriptorCoordinate(derivative_oracle_kind),
                "objective_relation": DescriptorCoordinate(objective_relation),
                "acceptance_relation": DescriptorCoordinate(acceptance_relation),
                "requested_residual_tolerance": DescriptorCoordinate(
                    requested_residual_tolerance
                ),
                "requested_solution_tolerance": DescriptorCoordinate(
                    requested_solution_tolerance
                ),
                "backend_kind": DescriptorCoordinate(backend_kind),
                "device_kind": DescriptorCoordinate(device_kind),
                "work_budget_fmas": DescriptorCoordinate(work_budget_fmas),
                "memory_budget_bytes": DescriptorCoordinate(memory_budget_bytes),
            },
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
        condition_estimate: float = 10.0,
        rank_estimate: int = 4,
        nullity_estimate: int = 0,
        rhs_consistency_defect: float = 0.0,
        **solve_overrides: Any,
    ) -> ParameterDescriptor:
        descriptor = cls._solve_descriptor(schema="linear_solver", **solve_overrides)
        return ParameterDescriptor(
            schema=descriptor.schema,
            coordinates=descriptor.coordinates
            | {
                "linear_operator_matrix_available": DescriptorCoordinate(
                    linear_operator_matrix_available
                ),
                "assembly_cost_fmas": DescriptorCoordinate(assembly_cost_fmas),
                "matvec_cost_fmas": DescriptorCoordinate(matvec_cost_fmas),
                "linear_operator_memory_bytes": DescriptorCoordinate(
                    linear_operator_memory_bytes
                ),
                "symmetry_defect": DescriptorCoordinate(symmetry_defect),
                "skew_symmetry_defect": DescriptorCoordinate(skew_symmetry_defect),
                "diagonal_nonzero_margin": DescriptorCoordinate(
                    diagonal_nonzero_margin
                ),
                "diagonal_dominance_margin": DescriptorCoordinate(
                    diagonal_dominance_margin
                ),
                "coercivity_lower_bound": DescriptorCoordinate(coercivity_lower_bound),
                "singular_value_lower_bound": DescriptorCoordinate(
                    singular_value_lower_bound
                ),
                "condition_estimate": DescriptorCoordinate(condition_estimate),
                "rank_estimate": DescriptorCoordinate(rank_estimate),
                "nullity_estimate": DescriptorCoordinate(nullity_estimate),
                "rhs_consistency_defect": DescriptorCoordinate(rhs_consistency_defect),
            },
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
            schema="decomposition",
            coordinates={
                "matrix_rows": DescriptorCoordinate(matrix_rows),
                "matrix_columns": DescriptorCoordinate(matrix_columns),
                "factorization_work_budget_fmas": DescriptorCoordinate(
                    factorization_work_budget_fmas
                ),
                "factorization_memory_budget_bytes": DescriptorCoordinate(
                    factorization_memory_budget_bytes
                ),
                "linear_operator_matrix_available": DescriptorCoordinate(
                    linear_operator_matrix_available
                ),
                "assembly_cost_fmas": DescriptorCoordinate(assembly_cost_fmas),
                "matvec_cost_fmas": DescriptorCoordinate(matvec_cost_fmas),
                "linear_operator_memory_bytes": DescriptorCoordinate(
                    linear_operator_memory_bytes
                ),
                "symmetry_defect": DescriptorCoordinate(symmetry_defect),
                "skew_symmetry_defect": DescriptorCoordinate(skew_symmetry_defect),
                "diagonal_nonzero_margin": DescriptorCoordinate(
                    diagonal_nonzero_margin
                ),
                "diagonal_dominance_margin": DescriptorCoordinate(
                    diagonal_dominance_margin
                ),
                "coercivity_lower_bound": DescriptorCoordinate(coercivity_lower_bound),
                "singular_value_lower_bound": DescriptorCoordinate(
                    singular_value_lower_bound
                ),
                "condition_estimate": DescriptorCoordinate(condition_estimate),
                "rank_estimate": DescriptorCoordinate(rank_estimate),
                "nullity_estimate": DescriptorCoordinate(nullity_estimate),
                "rhs_consistency_defect": DescriptorCoordinate(rhs_consistency_defect),
            },
        )


@dataclass(frozen=True)
class _AtlasProjection:
    """One descriptor template rendered into the capability atlas."""

    schema: ParameterSpaceSchema
    descriptor: ParameterDescriptor
    title: str
    shown_axes: tuple[str, ...]
    fixed_axes: tuple[str, ...]
    marginalized_axes: tuple[str, ...]
    patches: tuple[CoveragePatch, ...] = ()


@dataclass(frozen=True)
class _AtlasGap:
    """Known uncovered descriptor region rendered into the atlas."""

    name: str
    region: str
    descriptor: tuple[str, ...]
    selected_owner: str
    partial_owners: tuple[str, ...]
    required_capability: str


@dataclass(frozen=True)
class _AtlasEvidence:
    """Numerical claim metadata rendered on top of an atlas cell."""

    cell: str
    claim_file: str
    evidence_kind: str
    sampling: str


@dataclass(frozen=True)
class _AtlasRegionShape:
    """Projected region geometry rendered before descriptor evidence points."""

    name: str
    status: str
    geometry: str
    points: tuple[tuple[float, float], ...]
    condition: str


@dataclass(frozen=True)
class _AtlasPlotSpec:
    """One generated SVG projection of descriptor cells."""

    filename: str
    title: str
    schema: str
    x_axis: str
    y_axis: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    regions: tuple[_AtlasRegionShape, ...]
    cells: tuple[str, ...]
    caption: str


def _capability_atlas_projections() -> tuple[_AtlasProjection, ...]:
    solve_schema = solve_relation_parameter_schema()
    linear_schema = linear_solver_parameter_schema()
    decomposition_schema = decomposition_parameter_schema()

    return (
        _AtlasProjection(
            solve_schema,
            _SolveRelationSchemaClaim._solve_descriptor(),
            "Solve relation: square linear system",
            (
                "map_linearity_defect",
                "dim_x",
                "dim_y",
                "acceptance_relation",
            ),
            (
                "residual_target_available",
                "objective_relation",
                "target_is_zero",
            ),
            ("backend_kind", "device_kind", "work_budget_fmas", "memory_budget_bytes"),
        ),
        _AtlasProjection(
            solve_schema,
            _SolveRelationSchemaClaim._solve_descriptor(
                dim_x=3,
                dim_y=5,
                objective_relation="least_squares",
            ),
            "Solve relation: least-squares relation",
            (
                "map_linearity_defect",
                "dim_x",
                "dim_y",
                "objective_relation",
            ),
            ("residual_target_available", "acceptance_relation"),
            ("backend_kind", "device_kind", "work_budget_fmas", "memory_budget_bytes"),
        ),
        _AtlasProjection(
            solve_schema,
            _SolveRelationSchemaClaim._solve_descriptor(
                map_linearity_defect=None,
                map_linearity_evidence="unavailable",
                residual_target_available=False,
            ),
            "Solve relation: public nonlinear-solver gap",
            (
                "map_linearity_defect",
                "residual_target_available",
                "acceptance_relation",
            ),
            ("target_is_zero", "derivative_oracle_kind"),
            ("backend_kind", "device_kind", "work_budget_fmas", "memory_budget_bytes"),
        ),
        _AtlasProjection(
            solve_schema,
            _SolveRelationSchemaClaim._solve_descriptor(
                auxiliary_scalar_count=1,
                normalization_constraint_count=1,
                acceptance_relation="eigenpair_residual",
                objective_relation="spectral_residual",
            ),
            "Solve relation: eigenproblem region",
            (
                "auxiliary_scalar_count",
                "normalization_constraint_count",
                "acceptance_relation",
            ),
            ("objective_relation", "dim_x", "dim_y"),
            ("backend_kind", "device_kind", "work_budget_fmas", "memory_budget_bytes"),
        ),
        _AtlasProjection(
            solve_schema,
            _SolveRelationSchemaClaim._solve_descriptor(
                acceptance_relation="eigenpair_residual",
            ),
            "Invalid solve relation: eigenpair without spectral data",
            (
                "auxiliary_scalar_count",
                "normalization_constraint_count",
                "acceptance_relation",
            ),
            ("objective_relation",),
            ("backend_kind", "device_kind", "work_budget_fmas", "memory_budget_bytes"),
        ),
        _AtlasProjection(
            linear_schema,
            _SolveRelationSchemaClaim._linear_descriptor(),
            "Linear solver: SPD full-rank dense descriptor",
            (
                "dim_x",
                "dim_y",
                "symmetry_defect",
                "coercivity_lower_bound",
                "condition_estimate",
            ),
            (
                "singular_value_lower_bound",
                "operator_application_available",
                "linear_operator_matrix_available",
            ),
            ("backend_kind", "device_kind", "work_budget_fmas", "memory_budget_bytes"),
        ),
        _AtlasProjection(
            linear_schema,
            _SolveRelationSchemaClaim._linear_descriptor(
                singular_value_lower_bound=0.0,
                rank_estimate=3,
                nullity_estimate=1,
            ),
            "Linear solver: rank-deficient descriptor",
            (
                "singular_value_lower_bound",
                "rank_estimate",
                "nullity_estimate",
                "rhs_consistency_defect",
            ),
            ("dim_x", "dim_y", "objective_relation"),
            ("backend_kind", "device_kind", "work_budget_fmas", "memory_budget_bytes"),
        ),
        _AtlasProjection(
            linear_schema,
            _SolveRelationSchemaClaim._linear_descriptor(
                linear_operator_matrix_available=False,
                matrix_representation_available=False,
            ),
            "Linear solver: matrix-free descriptor",
            (
                "linear_operator_matrix_available",
                "operator_application_available",
                "matvec_cost_fmas",
            ),
            ("dim_x", "dim_y", "map_linearity_defect"),
            ("backend_kind", "device_kind", "work_budget_fmas", "memory_budget_bytes"),
        ),
        _AtlasProjection(
            linear_schema,
            _SolveRelationSchemaClaim._linear_descriptor(dim_y=5),
            "Invalid linear solver: nonsquare SPD descriptor",
            (
                "dim_x",
                "dim_y",
                "symmetry_defect",
                "coercivity_lower_bound",
            ),
            ("map_linearity_defect",),
            ("backend_kind", "device_kind", "work_budget_fmas", "memory_budget_bytes"),
        ),
        _AtlasProjection(
            decomposition_schema,
            _SolveRelationSchemaClaim._decomposition_descriptor(),
            "Decomposition: square full-rank dense descriptor",
            (
                "matrix_rows",
                "matrix_columns",
                "singular_value_lower_bound",
                "condition_estimate",
            ),
            ("factorization_work_budget_fmas", "factorization_memory_budget_bytes"),
            ("linear_operator_memory_bytes", "assembly_cost_fmas"),
        ),
        _AtlasProjection(
            decomposition_schema,
            _SolveRelationSchemaClaim._decomposition_descriptor(
                matrix_columns=5,
            ),
            "Invalid decomposition: nonsquare coercive descriptor",
            (
                "matrix_rows",
                "matrix_columns",
                "coercivity_lower_bound",
            ),
            ("factorization_work_budget_fmas", "factorization_memory_budget_bytes"),
            ("linear_operator_memory_bytes", "assembly_cost_fmas"),
        ),
    )


def _capability_atlas_gaps() -> tuple[_AtlasGap, ...]:
    return (
        _AtlasGap(
            name="nonlinear algebraic solve F(x) = 0",
            region="nonlinear_root",
            descriptor=(
                "map_linearity_defect > eps or unavailable",
                "residual_target_available = false or target_is_zero = true",
                "derivative_oracle_kind in {none, matrix, jvp, vjp, jacobian_callback}",
                "acceptance_relation = residual_below_tolerance",
                "requested_residual_tolerance = finite",
            ),
            selected_owner="none",
            partial_owners=(
                "time_integrators._newton.nonlinear_solve is internal stage machinery, "
                "not a public nonlinear-system solver capability.",
            ),
            required_capability=(
                "NonlinearSolver with descriptor bounds for residual norm, Jacobian "
                "availability, local convergence radius or globalization policy, "
                "line-search or trust-region safeguards, max residual evaluations, "
                "and failure reporting."
            ),
        ),
    )


def _capability_atlas_evidence() -> tuple[_AtlasEvidence, ...]:
    return ()


def _capability_atlas_plot_specs() -> tuple[_AtlasPlotSpec, ...]:
    return (
        _AtlasPlotSpec(
            "solve_relation.svg",
            "Solve-Relation Regions",
            "solve_relation",
            "dim_x",
            "dim_y",
            (1.0, 6.0),
            (1.0, 6.0),
            (
                _AtlasRegionShape(
                    "linear_system",
                    "uncovered",
                    "line",
                    ((1.0, 1.0), (6.0, 6.0)),
                    "linear map; target available; residual acceptance",
                ),
                _AtlasRegionShape(
                    "least_squares",
                    "uncovered",
                    "polygon",
                    ((1.0, 1.0), (1.0, 6.0), (6.0, 6.0)),
                    "linear map; least_squares objective; target available",
                ),
                _AtlasRegionShape(
                    "nonlinear_root",
                    "uncovered",
                    "rectangle",
                    ((1.0, 1.0), (6.0, 6.0)),
                    "nonlinear or unknown linearity; residual acceptance",
                ),
                _AtlasRegionShape(
                    "eigenproblem",
                    "uncovered",
                    "rectangle",
                    ((1.25, 1.25), (5.75, 5.75)),
                    "spectral scalar and normalization; eigenpair residual",
                ),
                _AtlasRegionShape(
                    "invalid_eigenpair_without_spectral_data",
                    "invalid",
                    "rectangle",
                    ((1.5, 1.5), (5.5, 5.5)),
                    "eigenpair residual missing spectral data",
                ),
            ),
            (
                "Solve relation: square linear system",
                "Solve relation: least-squares relation",
                "Solve relation: public nonlinear-solver gap",
                "Solve relation: eigenproblem region",
                "Invalid solve relation: eigenpair without spectral data",
            ),
            "Solve-relation projection over unknown and residual dimensions.",
        ),
        _AtlasPlotSpec(
            "linear_solver.svg",
            "Linear-Solver Regions",
            "linear_solver",
            "dim_x",
            "dim_y",
            (1.0, 6.0),
            (1.0, 6.0),
            (
                _AtlasRegionShape(
                    "linear_system",
                    "uncovered",
                    "line",
                    ((1.0, 1.0), (6.0, 6.0)),
                    "linear map; target available; residual acceptance",
                ),
                _AtlasRegionShape(
                    "symmetric_positive_definite",
                    "uncovered",
                    "line",
                    ((1.0, 1.0), (6.0, 6.0)),
                    "square; symmetric; positive coercivity",
                ),
                _AtlasRegionShape(
                    "overdetermined",
                    "uncovered",
                    "polygon",
                    ((1.0, 1.0), (1.0, 6.0), (6.0, 6.0)),
                    "dim_y > dim_x",
                ),
                _AtlasRegionShape(
                    "invalid_nonsquare_spd",
                    "invalid",
                    "polygon",
                    ((1.0, 1.0), (1.0, 6.0), (6.0, 6.0)),
                    "symmetry/coercivity asserted while nonsquare",
                ),
            ),
            (
                "Linear solver: SPD full-rank dense descriptor",
                "Linear solver: rank-deficient descriptor",
                "Linear solver: matrix-free descriptor",
                "Invalid linear solver: nonsquare SPD descriptor",
            ),
            "Linear-solver projection over unknown and residual dimensions.",
        ),
        _AtlasPlotSpec(
            "decomposition.svg",
            "Decomposition Regions",
            "decomposition",
            "matrix_rows",
            "matrix_columns",
            (1.0, 6.0),
            (1.0, 6.0),
            (
                _AtlasRegionShape(
                    "square",
                    "uncovered",
                    "line",
                    ((1.0, 1.0), (6.0, 6.0)),
                    "matrix_rows == matrix_columns",
                ),
                _AtlasRegionShape(
                    "invalid_nonsquare_coercive",
                    "invalid",
                    "polygon",
                    ((1.0, 1.0), (1.0, 6.0), (6.0, 6.0)),
                    "coercivity_lower_bound > 0 while matrix_rows != matrix_columns",
                ),
            ),
            (
                "Decomposition: square full-rank dense descriptor",
                "Invalid decomposition: nonsquare coercive descriptor",
            ),
            "Decomposition projection over matrix row and column dimensions.",
        ),
    )


def _descriptor_value(descriptor: ParameterDescriptor, field: str) -> str:
    coordinate = descriptor.coordinate(field)
    value = "unknown" if coordinate.value is None else str(coordinate.value)
    if coordinate.evidence != "exact":
        return f"{value} ({coordinate.evidence})"
    return value


def _matched_regions(
    schema: ParameterSpaceSchema,
    descriptor: ParameterDescriptor,
) -> str:
    matched = [
        region.name for region in schema.derived_regions if region.contains(descriptor)
    ]
    return ", ".join(matched) if matched else "none"


def _status_style(status: str) -> tuple[str, str]:
    if status == "invalid":
        return "#b42318", "#fee4e2"
    if status == "owned":
        return "#027a48", "#d1fadf"
    if status == "rejected":
        return "#b54708", "#fef0c7"
    return "#475467", "#f2f4f7"


def _region_opacity(status: str) -> str:
    if status == "invalid":
        return "0.20"
    if status == "owned":
        return "0.24"
    if status == "rejected":
        return "0.22"
    return "0.18"


def _plot_coordinate(value: float, *, axis_min: float, axis_max: float) -> float:
    if axis_max == axis_min:
        return 0.5
    return (value - axis_min) / (axis_max - axis_min)


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


def _render_region_shape(
    region: _AtlasRegionShape,
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
    opacity = _region_opacity(region.status)
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
            f'<polygon points="{svg_points}" fill="{fill}" fill-opacity="{opacity}" '
            f'stroke="{stroke}" stroke-width="2" stroke-opacity="0.55"/>'
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
            f'height="{py1 - py0:.1f}" fill="{fill}" fill-opacity="{opacity}" '
            f'stroke="{stroke}" stroke-width="2" stroke-dasharray="8 5" '
            'stroke-opacity="0.62"/>'
        )
    raise AssertionError(f"unsupported atlas region geometry {region.geometry!r}")


def _render_capability_atlas_plot(spec: _AtlasPlotSpec) -> str:
    projections = {
        projection.title: projection
        for projection in _capability_atlas_projections()
        if projection.schema.name == spec.schema
    }
    selected = [projections[cell] for cell in spec.cells]
    x_min, x_max = spec.x_range
    y_min, y_max = spec.y_range

    width = 1180
    height = 820
    left = 94
    right = 440
    top = 72
    bottom = 88
    plot_w = width - left - right
    plot_h = height - top - bottom

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="{spec.filename}-title {spec.filename}-desc">',
        f'<title id="{spec.filename}-title">{html.escape(spec.title)}</title>',
        f'<desc id="{spec.filename}-desc">{html.escape(spec.caption)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(30, 38, spec.title, size=24, weight="700"),
        _svg_text(30, 60, spec.caption, size=13),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#475467" stroke-width="1.4"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="#475467" stroke-width="1.4"/>',
        _svg_text(
            left + plot_w / 2, height - 28, spec.x_axis, size=14, anchor="middle"
        ),
        (
            f'<text x="24" y="{top + plot_h / 2:.1f}" font-size="14" '
            'font-family="Inter, Arial, sans-serif" fill="#101828" '
            f'text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2:.1f})">'
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
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                f'y2="{top + plot_h}" stroke="#eaecf0" stroke-width="1"/>',
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" '
                f'y2="{y:.1f}" stroke="#eaecf0" stroke-width="1"/>',
                _svg_text(
                    x, top + plot_h + 22, f"{x_value:.1f}", size=11, anchor="middle"
                ),
                _svg_text(left - 12, y + 4, f"{y_value:.1f}", size=11, anchor="end"),
            ]
        )

    for region in spec.regions:
        parts.append(
            _render_region_shape(
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
        )

    legend_y = 98
    for status in ("uncovered", "invalid", "owned", "rejected"):
        stroke, fill = _status_style(status)
        parts.extend(
            [
                f'<rect x="{width - 257}" y="{legend_y - 8}" width="14" '
                f'height="14" fill="{fill}" fill-opacity="{_region_opacity(status)}" '
                f'stroke="{stroke}" stroke-width="2"/>',
                _svg_text(width - 234, legend_y + 3, f"{status} region", size=12),
            ]
        )
        legend_y += 24

    label_y = 206
    parts.append(
        _svg_text(width - 292, label_y, "Projected regions", size=13, weight="700")
    )
    label_y += 24
    for index, region in enumerate(spec.regions, start=1):
        stroke, fill = _status_style(region.status)
        opacity = _region_opacity(region.status)
        parts.extend(
            [
                f'<rect x="{width - 292}" y="{label_y - 11}" width="12" '
                f'height="12" fill="{fill}" fill-opacity="{opacity}" '
                f'stroke="{stroke}" stroke-width="1.5"/>',
                _svg_text(width - 274, label_y, f"{index}. {region.name}", size=12),
                _svg_text(width - 274, label_y + 15, region.condition, size=10),
            ]
        )
        label_y += 42

    label_y += 8
    parts.append(
        _svg_text(
            width - 292, label_y, "Test descriptor overlays", size=13, weight="700"
        )
    )
    label_y += 24
    seen_points: dict[tuple[float, float], int] = {}
    for index, projection in enumerate(selected, start=1):
        status = projection.schema.cell_status(
            projection.descriptor, projection.patches
        )
        stroke, fill = _status_style(status)
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
        offsets = ((0.0, 0.0), (16.0, 0.0), (-16.0, 0.0), (0.0, -16.0), (0.0, 16.0))
        dx, dy = offsets[duplicate_index % len(offsets)]
        x += dx
        y += dy
        regions = _matched_regions(projection.schema, projection.descriptor)
        parts.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#ffffff" '
                f'stroke="{stroke}" stroke-width="2.4"/>',
                _svg_text(x, y + 4, str(index), size=10, anchor="middle", weight="700"),
                _svg_text(
                    width - 292, label_y, f"{index}. {projection.title}", size=11
                ),
                _svg_text(width - 274, label_y + 17, f"status: {status}", size=11),
                _svg_text(width - 274, label_y + 32, f"regions: {regions}", size=11),
            ]
        )
        label_y += 58

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _render_capability_atlas_plots() -> dict[str, str]:
    return {
        spec.filename: _render_capability_atlas_plot(spec)
        for spec in _capability_atlas_plot_specs()
    }


def _render_capability_atlas() -> str:
    lines = [
        "# Capability Coverage Atlas",
        "",
        "<!-- Generated from tests/test_structure.py; do not edit by hand. -->",
        "",
        "This page is a projection of the parameter-space schemas used by the",
        "structural test registry.  Each plot names the axes shown directly, the",
        "coordinates fixed outside the projection, and the higher-dimensional axes",
        "that are only summarized.  Region geometry is drawn first; concrete",
        "descriptor fixtures from `tests/test_structure.py` are overlaid as",
        "numbered evidence points.  Ownership is intentionally sparse at this",
        "stage: solver and decomposition implementations have not yet been",
        "converted from string-set capability tags to coverage patches.",
        "",
        "Status legend:",
        "",
        "- `invalid`: the descriptor violates a schema validity rule.",
        "- `owned`: at least one coverage patch owns the descriptor.",
        "- `rejected`: coverage patches intentionally reject the descriptor.",
        "- `uncovered`: the descriptor is valid but no coverage patch owns it.",
        "",
        "## Projection Plots",
        "",
    ]
    projections = {
        projection.title: projection for projection in _capability_atlas_projections()
    }
    for spec in _capability_atlas_plot_specs():
        lines.extend(
            [
                f"### {spec.title}",
                "",
                f"![{spec.title}](capability_atlas_plots/{spec.filename})",
                "",
                f"Shown axes: `{spec.x_axis}` and `{spec.y_axis}`.",
            ]
        )
        fixed = sorted(
            {field for cell in spec.cells for field in projections[cell].fixed_axes}
        )
        marginalized = sorted(
            {
                field
                for cell in spec.cells
                for field in projections[cell].marginalized_axes
            }
        )
        lines.extend(
            [
                "Fixed axes: " + ", ".join(f"`{field}`" for field in fixed) + ".",
                "Marginalized axes: "
                + ", ".join(f"`{field}`" for field in marginalized)
                + ".",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## Coverage Patches",
            "",
            "No solver or decomposition ownership patches are declared in this",
            "atlas yet.  The next sprint items convert existing linear-solver and",
            "decomposition capabilities into bounded coverage patches with explicit",
            "cost models and priority rules.",
            "",
            "## Known Gaps",
            "",
        ]
    )
    for gap in _capability_atlas_gaps():
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
            "No owned solver or decomposition coverage patch has numerical evidence",
            "metadata in this atlas yet.  Until ownership patches exist, numerical",
            "correctness, convergence, performance, and regression claims remain",
            "outside this projection rather than being attached to cells.",
            "",
        ]
    )
    return "\n".join(lines)


class _CapabilityAtlasDocClaim(Claim[None]):
    """Claim: capability atlas documentation can be generated."""

    @property
    def description(self) -> str:
        return "algorithm_capabilities/capability_atlas_doc_generates"

    def check(self, _calibration: None) -> None:
        expected = _render_capability_atlas()
        assert "![Solve-Relation Regions]" in expected
        assert "![Linear-Solver Regions]" in expected
        assert "![Decomposition Regions]" in expected
        assert _render_capability_atlas_plots()


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
_LinearSolverRequest = _resolve_dotted(
    "cosmic_foundry.computation.solvers.LinearSolverRequest"
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
                "LinearSolverCapability",
                "linear_solver_capabilities",
                "LinearSolverRegistry",
                "LinearSolverRequest",
                "select_linear_solver",
            }
        ),
        "abstract_interface": frozenset(
            {
                "DirectSolver",
                "IterativeSolver",
                "LinearOperator",
                "LinearSolver",
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
    capability_provider="cosmic_foundry.computation.solvers.linear_solver_capabilities",
    request_selector="cosmic_foundry.computation.solvers.select_linear_solver",
    request_expectations=(
        _CapabilityRequestExpectation(
            _LinearSolverRequest(
                available_structure=frozenset(
                    {"dense_operator", "square_system", "full_rank"}
                ),
                requested_properties=frozenset({"solve", "direct", "exact"}),
            ),
            "DenseLUSolver",
        ),
        _CapabilityRequestExpectation(
            _LinearSolverRequest(
                available_structure=frozenset({"dense_operator"}),
                requested_properties=frozenset(
                    {"solve", "direct", "minimum_norm", "rank_deficient"}
                ),
            ),
            "DenseSVDSolver",
        ),
        _CapabilityRequestExpectation(
            _LinearSolverRequest(
                available_structure=frozenset(
                    {"linear_operator", "symmetric_positive_definite"}
                ),
                requested_properties=frozenset({"solve", "iterative", "krylov"}),
            ),
            "DenseCGSolver",
        ),
        _CapabilityRequestExpectation(
            _LinearSolverRequest(
                available_structure=frozenset({"linear_operator", "nonsingular"}),
                requested_properties=frozenset(
                    {"solve", "iterative", "krylov", "general"}
                ),
            ),
            "DenseGMRESSolver",
        ),
        _CapabilityRequestExpectation(
            _LinearSolverRequest(
                available_structure=frozenset(
                    {"linear_operator", "diagonal", "row_abs_sums"}
                ),
                requested_properties=frozenset(
                    {"solve", "iterative", "stationary", "matrix_free"}
                ),
            ),
            "DenseJacobiSolver",
        ),
    ),
    rejected_requests=(
        _CapabilityRejectionExpectation(
            _LinearSolverRequest(
                available_structure=frozenset({"linear_operator"}),
                requested_properties=frozenset({"solve", "direct", "exact"}),
            )
        ),
    ),
    expected_class_modules={
        "DenseCGSolver": "dense_cg_solver",
        "DenseGMRESSolver": "dense_gmres_solver",
        "DenseGaussSeidelSolver": "dense_gauss_seidel_solver",
        "DenseJacobiSolver": "dense_jacobi_solver",
        "DenseLUSolver": "dense_lu_solver",
        "DenseSVDSolver": "dense_svd_solver",
        "DirectSolver": "direct_solver",
        "IterativeSolver": "iterative_solver",
        "LinearOperator": "linear_solver",
        "LinearSolver": "linear_solver",
        "LinearSolverCapability": "algorithm_capabilities",
        "LinearSolverRegistry": "algorithm_capabilities",
        "LinearSolverRequest": "algorithm_capabilities",
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
    _CapabilityAtlasDocClaim(),
]


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_structure(claim: Claim[None]) -> None:
    claim.check(None)
