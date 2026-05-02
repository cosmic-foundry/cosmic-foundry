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
    EvidencePredicate,
    InvalidCellRule,
    MembershipPredicate,
    NumericInterval,
    ParameterAxis,
    ParameterBin,
    ParameterDescriptor,
    ParameterSpaceSchema,
    decomposition_parameter_schema,
    linear_operator_descriptor_from_assembled_operator,
    linear_solver_parameter_schema,
    solve_relation_parameter_schema,
)
from cosmic_foundry.computation.backends.python_backend import PythonBackend
from cosmic_foundry.computation.decompositions.factorization import Factorization
from cosmic_foundry.computation.solvers.capabilities import (
    linear_solver_coverage_patches,
    select_linear_solver_for_descriptor,
)
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
        condition_estimate: float | None = 10.0,
        condition_evidence: str = "exact",
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
                "condition_estimate": DescriptorCoordinate(
                    condition_estimate,
                    evidence=condition_evidence,  # type: ignore[arg-type]
                ),
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


class _LinearSolverCoveragePatchClaim(Claim[None]):
    """Claim: linear-solver selection is driven by schema coverage patches."""

    @property
    def description(self) -> str:
        return "algorithm_capabilities/linear_solver_coverage_patches"

    def check(self, _calibration: None) -> None:
        schema = linear_solver_parameter_schema()
        patches = linear_solver_coverage_patches()
        self._assert_no_declared_coverage_literals(patches)
        for patch in patches:
            schema.validate_coverage_patch(patch)

        selected_owners: set[str] = set()
        owned_patches = tuple(patch for patch in patches if patch.status == "owned")
        rejected_patches = tuple(
            patch for patch in patches if patch.status == "rejected"
        )
        for patch in owned_patches:
            descriptor = self._descriptor_witness_for_patch(schema, patch)
            assert schema.cell_status(descriptor, patches) == "owned"
            selected = select_linear_solver_for_descriptor(descriptor)
            expected_owner = self._selected_patch_owner(descriptor, patches)
            assert selected.implementation == expected_owner
            selected_owners.add(expected_owner)
        assert selected_owners == {patch.owner for patch in owned_patches}

        for patch in rejected_patches:
            descriptor = self._descriptor_witness_for_patch(schema, patch)
            assert schema.cell_status(descriptor, patches) == "rejected"
            with pytest.raises(ValueError):
                select_linear_solver_for_descriptor(descriptor)

    @staticmethod
    def _selected_patch_owner(
        descriptor: ParameterDescriptor,
        patches: tuple[CoveragePatch, ...],
    ) -> str:
        matches = tuple(
            patch
            for patch in patches
            if patch.status == "owned" and patch.contains(descriptor)
        )
        assert matches
        ranked = sorted(
            matches,
            key=lambda patch: (
                patch.priority if patch.priority is not None else float("inf")
            ),
        )
        ranked_iter = iter(ranked)
        first = next(ranked_iter)
        second = next(ranked_iter, None)
        assert second is None or first.priority != second.priority
        return first.owner

    @classmethod
    def _assert_no_declared_coverage_literals(
        cls,
        patches: tuple[CoveragePatch, ...],
    ) -> None:
        declared = cls._declared_coverage_literals(patches)
        leaked = declared & cls._claim_source_literals()
        assert not leaked, f"coverage facts leaked into structural claim: {leaked}"

    @staticmethod
    def _declared_coverage_literals(
        patches: tuple[CoveragePatch, ...],
    ) -> frozenset[str | int | float]:
        literals: set[str | int | float] = set()
        for patch in patches:
            literals.update({patch.name, patch.owner})
            if patch.priority is not None:
                literals.add(patch.priority)
            for predicate in patch.predicates:
                literals.update(predicate.referenced_fields)
                if isinstance(predicate, ComparisonPredicate):
                    if not isinstance(predicate.value, bool):
                        literals.add(predicate.value)
                elif isinstance(predicate, MembershipPredicate):
                    literals.update(
                        value
                        for value in predicate.values
                        if isinstance(value, str | int | float)
                        and not isinstance(value, bool)
                    )
                elif isinstance(predicate, EvidencePredicate):
                    literals.update(predicate.evidence)
                elif isinstance(predicate, AffineComparisonPredicate):
                    literals.update(predicate.terms)
                    literals.add(predicate.value)
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
    def _descriptor_witness_for_patch(
        cls,
        schema: ParameterSpaceSchema,
        patch: CoveragePatch,
    ) -> ParameterDescriptor:
        fields = {axis.field: axis for axis in schema.axes}
        coordinates = {
            field: DescriptorCoordinate(cls._axis_witness(axis))
            for field, axis in fields.items()
        }
        for predicate in patch.predicates:
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
        descriptor = ParameterDescriptor(schema.name, coordinates)
        assert patch.contains(descriptor)
        return descriptor

    @classmethod
    def _axis_witness(cls, axis: ParameterAxis) -> Any:
        for bin_or_interval in axis.bins:
            if isinstance(bin_or_interval, ParameterBin):
                return next(iter(sorted(bin_or_interval.values, key=str)))
            value = cls._interval_witness(bin_or_interval)
            if axis.contains(DescriptorCoordinate(value)):
                return value
        raise AssertionError(f"axis {axis.name!r} has no witness value")

    @staticmethod
    def _interval_witness(interval: NumericInterval) -> float:
        if interval.lower is not None and interval.upper is not None:
            if interval.include_lower and interval.contains(interval.lower):
                return interval.lower
            if interval.include_upper and interval.contains(interval.upper):
                return interval.upper
            return (
                interval.lower + interval.upper
            ) / _LinearSolverCoveragePatchClaim._two()
        if interval.lower is not None:
            if interval.include_lower and interval.contains(interval.lower):
                return interval.lower
            return interval.lower + max(
                abs(interval.lower), _LinearSolverCoveragePatchClaim._one()
            )
        if interval.upper is not None:
            if interval.include_upper and interval.contains(interval.upper):
                return interval.upper
            return interval.upper - max(
                abs(interval.upper), _LinearSolverCoveragePatchClaim._one()
            )
        return _LinearSolverCoveragePatchClaim._zero()

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
                    ParameterDescriptor(
                        axis.name,
                        {axis.field: coordinate},
                    )
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
        fields: dict[str, ParameterAxis],
        coordinates: dict[str, DescriptorCoordinate],
        predicate: AffineComparisonPredicate,
    ) -> None:
        if predicate.evaluate(ParameterDescriptor("witness", coordinates)):
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
            ) and predicate.evaluate(ParameterDescriptor("witness", coordinates)):
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
    """Claim: owned solver coverage patches are declared inside implementations."""

    @property
    def description(self) -> str:
        return "algorithm_capabilities/linear_solver_coverage_locality"

    def check(self, _calibration: None) -> None:
        for path in sorted((_PACKAGE_ROOT / "computation" / "solvers").glob("*.py")):
            if path.name.startswith("_") or path.name == "capabilities.py":
                continue
            tree = ast.parse(path.read_text())
            for owner, class_name in self._owned_coverage_locations(tree):
                assert owner == class_name, (
                    "owned linear-solver coverage patch must be declared in "
                    f"class {owner}: {path.relative_to(_PROJECT_ROOT)}"
                )
            manual_categories = self._manual_capability_categories(tree)
            assert not manual_categories, (
                "linear-solver categories must be inferred from implementation "
                f"inheritance: {path.relative_to(_PROJECT_ROOT)}"
            )
            manual_names = self._manual_capability_names(tree)
            assert not manual_names, (
                "linear-solver capability names must come from class identity: "
                f"{path.relative_to(_PROJECT_ROOT)}"
            )
            manual_patches = self._manual_owned_patch_calls(tree)
            assert not manual_patches, (
                "owned linear-solver coverage patches must be built from class "
                f"identity: {path.relative_to(_PROJECT_ROOT)}"
            )
            manual_contract_atoms = self._manual_contract_atoms(tree)
            assert not manual_contract_atoms, (
                "linear-solver contract atoms must use canonical typed atoms: "
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

    @classmethod
    def _manual_capability_categories(cls, tree: ast.Module) -> tuple[str, ...]:
        categories: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if cls._call_name(node.func) != "LinearSolverCapability":
                continue
            category = cls._string_arg(node, 2, "category")
            if category is not None:
                categories.append(category)
        return tuple(categories)

    @classmethod
    def _manual_capability_names(cls, tree: ast.Module) -> tuple[str, ...]:
        names: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if cls._call_name(node.func) != "capability":
                continue
            name = cls._string_arg(node, 1, "name")
            if name is not None:
                names.append(name)
        return tuple(names)

    @classmethod
    def _manual_owned_patch_calls(cls, tree: ast.Module) -> tuple[str, ...]:
        calls: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = cls._call_name(node.func)
            if call_name == "owned_patch":
                calls.append(call_name)
            elif call_name == "CoveragePatch":
                status = cls._string_arg(node, 2, "status")
                if status == "owned":
                    calls.append(call_name)
        return tuple(calls)

    @classmethod
    def _manual_contract_atoms(cls, tree: ast.Module) -> tuple[str, ...]:
        atoms: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if cls._call_name(node.func) != "contract":
                continue
            for kw in node.keywords:
                if kw.arg not in {"requires", "provides"}:
                    continue
                atoms.extend(cls._string_literals(kw.value))
        return tuple(atoms)

    @classmethod
    def _string_literals(cls, node: ast.AST) -> tuple[str, ...]:
        values: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                values.append(child.value)
        return tuple(values)

    @classmethod
    def _owned_coverage_owner(cls, node: ast.Call) -> str | None:
        call_name = cls._call_name(node.func)
        if call_name == "owned_patch":
            return cls._string_arg(node, 1, "owner")
        if call_name != "CoveragePatch":
            return None
        status = cls._string_arg(node, 2, "status")
        if status != "owned":
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
class _AtlasRegionProjection:
    """Schema region selected for projection onto a plot."""

    name: str
    status: str
    source_kind: str
    source_name: str
    condition: str = ""


@dataclass(frozen=True)
class _AtlasRegionShape:
    """Projected region geometry derived from schema predicates."""

    name: str
    status: str
    source_name: str
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
    cells: tuple[str, ...]
    caption: str


def _capability_atlas_projections() -> tuple[_AtlasProjection, ...]:
    solve_schema = solve_relation_parameter_schema()
    linear_schema = linear_solver_parameter_schema()
    decomposition_schema = decomposition_parameter_schema()
    solver_patches = linear_solver_coverage_patches()

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
            solver_patches,
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
            solver_patches,
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
            solver_patches,
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
            solver_patches,
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


def _predicate_label(predicate: Any) -> str:
    if isinstance(predicate, ComparisonPredicate):
        return f"{predicate.field} {predicate.operator} {predicate.value}"
    if isinstance(predicate, AffineComparisonPredicate):
        terms = " + ".join(
            f"{coefficient:g}*{field}"
            for field, coefficient in sorted(predicate.terms.items())
        )
        if predicate.offset:
            terms = f"{terms} + {predicate.offset:g}"
        return f"{terms} {predicate.operator} {predicate.value:g}"
    if isinstance(predicate, MembershipPredicate):
        values = ", ".join(str(value) for value in sorted(predicate.values, key=str))
        return f"{predicate.field} in {{{values}}}"
    if isinstance(predicate, EvidencePredicate):
        evidence = ", ".join(sorted(predicate.evidence))
        return f"{predicate.field} evidence in {{{evidence}}}"
    return repr(predicate)


def _source_alternatives(
    schema: ParameterSpaceSchema,
    region: _AtlasRegionProjection,
    patches: tuple[CoveragePatch, ...] = (),
) -> tuple[tuple[Any, ...], ...]:
    if region.source_kind == "derived_region":
        matches = [
            candidate
            for candidate in schema.derived_regions
            if candidate.name == region.source_name
        ]
        if not matches:
            raise AssertionError(
                f"atlas region {region.name!r} references missing "
                f"derived region {region.source_name!r}"
            )
        return matches[0].alternatives
    if region.source_kind == "invalid_cell":
        matches = [
            candidate
            for candidate in schema.invalid_cells
            if candidate.name == region.source_name
        ]
        if not matches:
            raise AssertionError(
                f"atlas region {region.name!r} references missing "
                f"invalid cell {region.source_name!r}"
            )
        return (matches[0].predicates,)
    if region.source_kind == "coverage_patch":
        matches = [
            candidate for candidate in patches if candidate.name == region.source_name
        ]
        if not matches:
            raise AssertionError(
                f"atlas region {region.name!r} references missing "
                f"coverage patch {region.source_name!r}"
            )
        schema.validate_coverage_patch(matches[0])
        return (matches[0].predicates,)
    raise AssertionError(f"unsupported atlas source kind {region.source_kind!r}")


def _schema_atlas_regions(
    schema: ParameterSpaceSchema,
    patches: tuple[CoveragePatch, ...] = (),
) -> tuple[_AtlasRegionProjection, ...]:
    """Return every schema region that should appear in atlas projections."""
    derived = tuple(
        _AtlasRegionProjection(
            name=region.name,
            status="uncovered",
            source_kind="derived_region",
            source_name=region.name,
        )
        for region in schema.derived_regions
    )
    invalid = tuple(
        _AtlasRegionProjection(
            name=rule.name,
            status="invalid",
            source_kind="invalid_cell",
            source_name=rule.name,
            condition=rule.reason,
        )
        for rule in schema.invalid_cells
    )
    coverage = tuple(
        _AtlasRegionProjection(
            name=patch.name,
            status=patch.status,
            source_kind="coverage_patch",
            source_name=patch.name,
            condition=f"{patch.owner} coverage patch",
        )
        for patch in patches
    )
    return derived + invalid + coverage


def _atlas_patches_for_schema(schema_name: str) -> tuple[CoveragePatch, ...]:
    patches: dict[str, CoveragePatch] = {}
    for projection in _capability_atlas_projections():
        if projection.schema.name != schema_name:
            continue
        for patch in projection.patches:
            patches[patch.name] = patch
    return tuple(patches.values())


def _predicate_affine_projection(
    predicate: Any, x_axis: str, y_axis: str
) -> tuple[dict[str, float], str, float] | None:
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
        return (
            {
                x_axis: 1.0 if predicate.field == x_axis else 0.0,
                y_axis: 1.0 if predicate.field == y_axis else 0.0,
            },
            predicate.operator,
            float(predicate.value),
        )
    return None


def _affine_value(
    point: tuple[float, float],
    terms: dict[str, float],
    x_axis: str,
    y_axis: str,
    value: float,
) -> float:
    x, y = point
    return terms.get(x_axis, 0.0) * x + terms.get(y_axis, 0.0) * y - value


def _clip_polygon_to_half_plane(
    polygon: tuple[tuple[float, float], ...],
    terms: dict[str, float],
    operator: str,
    value: float,
    x_axis: str,
    y_axis: str,
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
    terms: dict[str, float],
    value: float,
    x_axis: str,
    y_axis: str,
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
    x_axis: str,
    y_axis: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
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
        return (("rectangle", ((x_range[0], y_range[0]), (x_range[1], y_range[1]))),)

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
        return (("line", line),) if len(line) == 2 else ()

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

    return tuple(("polygon", polygon) for polygon in polygons if len(polygon) >= 3)


def _projected_region_shapes(spec: _AtlasPlotSpec) -> tuple[_AtlasRegionShape, ...]:
    schemas = {
        schema.name: schema
        for schema in (
            solve_relation_parameter_schema(),
            linear_solver_parameter_schema(),
            decomposition_parameter_schema(),
        )
    }
    schema = schemas[spec.schema]
    patches = _atlas_patches_for_schema(spec.schema)
    shapes: list[_AtlasRegionShape] = []
    for region in _schema_atlas_regions(schema, patches):
        alternatives = _source_alternatives(schema, region, patches)
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
                    f"atlas region {region.name!r} has no visible projection "
                    f"onto {spec.x_axis!r}/{spec.y_axis!r}"
                )
            predicate_summary = "; ".join(
                _predicate_label(predicate) for predicate in predicates
            )
            condition = region.condition or predicate_summary
            if len(alternatives) > 1:
                condition = f"alternative {alternative_index}: {condition}"
            shape_name = (
                region.name
                if len(alternatives) == 1
                else f"{region.name} alt {alternative_index}"
            )
            for geometry_name, points in geometry:
                shapes.append(
                    _AtlasRegionShape(
                        shape_name,
                        region.status,
                        region.source_name,
                        geometry_name,
                        points,
                        condition,
                    )
                )
    return tuple(shapes)


class _CapabilityAtlasDocClaim(Claim[None]):
    """Claim: capability atlas documentation can be generated."""

    @property
    def description(self) -> str:
        return "algorithm_capabilities/capability_atlas_doc_generates"

    def check(self, _calibration: None) -> None:
        schemas = {
            schema.name: schema
            for schema in (
                solve_relation_parameter_schema(),
                linear_solver_parameter_schema(),
                decomposition_parameter_schema(),
            )
        }
        for spec in _capability_atlas_plot_specs():
            schema = schemas[spec.schema]
            patches = _atlas_patches_for_schema(spec.schema)
            discovered = _schema_atlas_regions(schema, patches)
            assert {
                (region.source_kind, region.source_name) for region in discovered
            } == {
                ("derived_region", region.name) for region in schema.derived_regions
            } | {
                ("invalid_cell", rule.name) for rule in schema.invalid_cells
            } | {
                ("coverage_patch", patch.name) for patch in patches
            }
            shapes = _projected_region_shapes(spec)
            assert shapes
            for shape in shapes:
                assert shape.source_name
                assert shape.points


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
                "linear_solver_coverage_patches",
                "LinearSolverRegistry",
                "LinearSolverRequest",
                "select_linear_solver",
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
        "KrylovSolver": "iterative_solver",
        "LinearOperator": "linear_solver",
        "LinearSolver": "linear_solver",
        "LinearSolverCapability": "algorithm_capabilities",
        "LinearSolverRegistry": "algorithm_capabilities",
        "LinearSolverRequest": "algorithm_capabilities",
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
    _LinearSolverCoveragePatchClaim(),
    _CapabilityAtlasDocClaim(),
]


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_structure(claim: Claim[None]) -> None:
    claim.check(None)
