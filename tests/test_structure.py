"""Structural invariant claims for the cosmic_foundry codebase.

Each claim encodes one structural property of the codebase. Adding a new
claim requires only appending to _CLAIMS; the single parametric test covers
all entries.

Claim types:
  _AbcInstantiationClaim    — discovered ABCs cannot be directly instantiated
  _HierarchyClaim           — discovered cosmic_foundry subclass relations are correct
  _ModuleAllClaim           — discovered public classes appear in __all__
  _IterativeSolverJitClaim  — iterative solver runs on a small assembled LinearOperator
  _MaterializationGateClaim — converged() raises MaterializationError on .get()
  _FactorizationJitClaim    — Factorization.factorize/solve run on declared Tensors
  _GenericBasesClaim        — no subclass leaves a generic base's TypeVars unbound
  _ManifoldIsolationClaim   — Manifold and IndexedSet hierarchies are disjoint
  _ImportBoundaryClaim      — pure packages import only approved packages
  _ArchitectureOwnershipClaim — package exports and capability ownership are explicit
  _LinearSolverCoverageLocalityClaim — owned solver coverage lives in
                                      implementation classes
  _TestFileStructureClaim   — test modules use claim-dispatch structure
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import sys
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NewType, TypeAlias

import pytest

from cosmic_foundry.computation.algorithm_capabilities import (
    AffineComparisonPredicate,
    AlgorithmCapability,
    AlgorithmRegistry,
    AlgorithmRequest,
    AlgorithmStructureContract,
    ComparisonPredicate,
    CoverageRegion,
    DecompositionField,
    DerivedParameterRegion,
    DescriptorCoordinate,
    DescriptorField,
    EvidencePredicate,
    InvalidCellRule,
    LinearSolverField,
    MapStructureField,
    MembershipPredicate,
    NumericInterval,
    ParameterAxis,
    ParameterBin,
    ParameterDescriptor,
    ParameterSpaceSchema,
    ReactionNetworkField,
    SolveRelationField,
    coverage_regions_are_disjoint,
    decomposition_descriptor_from_linear_operator_descriptor,
    decomposition_parameter_schema,
    linear_operator_descriptor_from_assembled_operator,
    linear_solver_parameter_schema,
    map_structure_parameter_schema,
    predicate_sets_are_disjoint,
    reaction_network_parameter_schema,
    solve_relation_parameter_schema,
)
from cosmic_foundry.computation.backends.python_backend import PythonBackend
from cosmic_foundry.computation.decompositions.factorization import Factorization
from cosmic_foundry.computation.solvers.capabilities import (
    linear_solver_coverage_regions,
    select_linear_solver_for_descriptor,
)
from cosmic_foundry.computation.solvers.iterative_solver import (
    IterativeSolver,
    StationaryIterationSolver,
)
from cosmic_foundry.computation.tensor import MaterializationError, Tensor
from cosmic_foundry.theory.continuous.manifold import Manifold
from cosmic_foundry.theory.foundation.indexed_set import IndexedSet
from tests.claims import Claim

_PROJECT_ROOT = Path(__file__).parent.parent
_PACKAGE_ROOT = _PROJECT_ROOT / "cosmic_foundry"
_AtlasText = NewType("_AtlasText", str)
_AtlasDescriptorField: TypeAlias = (
    SolveRelationField | LinearSolverField | DecompositionField | ReactionNetworkField
)
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


class _AffineTestRHS:
    """Tiny affine RHS used by solve-relation descriptor structure claims."""

    def __call__(self, t: float, u: Tensor) -> Tensor:
        return self.linear_operator(t, u) @ u

    def linear_operator(self, _t: float, u: Tensor) -> Tensor:
        return Tensor([[0.0, 8.0], [0.0, 0.0]], backend=u.backend)


def _unknown_test_rhs(_t: float, u: Tensor) -> Tensor:
    return u


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


def _discover_concrete_least_squares_solvers(
    modules: list[tuple[str, types.ModuleType]],
) -> list[type]:
    least_squares_solver = _resolve_dotted(
        "cosmic_foundry.computation.solvers.LeastSquaresSolver"
    )
    seen: set[type] = set()
    result: list[type] = []
    for _, mod in modules:
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                obj not in seen
                and issubclass(obj, least_squares_solver)
                and not getattr(obj, "__abstractmethods__", None)
                and obj is not least_squares_solver
            ):
                seen.add(obj)
                result.append(obj)
    return result


def _discover_concrete_time_integrators(
    modules: list[tuple[str, types.ModuleType]],
) -> list[type]:
    time_integrator = _resolve_dotted(
        "cosmic_foundry.computation.time_integrators.TimeIntegrator"
    )
    seen: set[type] = set()
    result: list[type] = []
    for _, mod in modules:
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                obj not in seen
                and issubclass(obj, time_integrator)
                and not getattr(obj, "__abstractmethods__", None)
                and obj is not time_integrator
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
    @property
    def description(self) -> str:
        return "abc_not_instantiable/discovered"

    def check(self, _calibration: None) -> None:
        instantiable = []
        for cls in _ABCS:
            try:
                cls()
            except TypeError:
                continue
            instantiable.append(f"{cls.__module__}.{cls.__qualname__}")
        assert not instantiable, "ABCs were directly instantiable: " + ", ".join(
            sorted(instantiable)
        )


class _HierarchyClaim(Claim[None]):
    @property
    def description(self) -> str:
        return "hierarchy/discovered"

    def check(self, _calibration: None) -> None:
        violations = [
            f"{child.__module__}.{child.__qualname__} !< "
            f"{parent.__module__}.{parent.__qualname__}"
            for child, parent in _HIERARCHY_PAIRS
            if not issubclass(child, parent)
        ]
        assert not violations, "hierarchy violations: " + "; ".join(violations)


class _ModuleAllClaim(Claim[None]):
    @property
    def description(self) -> str:
        return "module_all/discovered"

    def check(self, _calibration: None) -> None:
        violations = []
        for mod_path, mod in _MODULES:
            exported = set(getattr(mod, "__all__", []))
            defined = {
                name
                for name, obj in inspect.getmembers(mod, inspect.isclass)
                if obj.__module__ == mod_path and not name.startswith("_")
            }
            missing = defined - exported
            if missing:
                violations.append(f"{mod_path}: {sorted(missing)}")
        assert (
            not violations
        ), "defined public classes missing from __all__: " + "; ".join(violations)


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
    """Claim: theory/ and geometry/ source files import no numerical packages."""

    @property
    def description(self) -> str:
        return "import_boundary/pure_packages"

    def check(self, _calibration: None) -> None:
        violations = [
            f"{path.relative_to(_PACKAGE_ROOT.parent)}: {', '.join(imports)}"
            for pkg_dir in _PURE_PACKAGES
            for path in sorted(pkg_dir.rglob("*.py"))
            if (imports := _third_party_imports(path))
        ]
        if violations:
            raise AssertionError(
                "pure packages import non-symbolic packages: " + "; ".join(violations)
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
    descriptor_owned_capabilities: bool = False
    descriptor_request_property_limit: int | None = None
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
        if self._spec.descriptor_owned_capabilities:
            structure_gated = [
                self._capability_implementation_name(cap)
                for cap in capabilities
                if cap.contract.requires
            ]
            assert not structure_gated, (
                "descriptor-owned capabilities require ad hoc structure gates: "
                f"{structure_gated}"
            )

        if self._spec.request_selector is None:
            return
        selector = _resolve_dotted(self._spec.request_selector)
        self._check_descriptor_requests()
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

    def _check_descriptor_requests(self) -> None:
        limit = self._spec.descriptor_request_property_limit
        if limit is None:
            return
        oversized = [
            request
            for request in (
                *(
                    expectation.request
                    for expectation in self._spec.request_expectations
                ),
                *(expectation.request for expectation in self._spec.rejected_requests),
            )
            if request.descriptor is not None
            and len(request.requested_properties) > limit
        ]
        assert not oversized, (
            "descriptor-owned requests must not smuggle family labels through "
            f"property lists: {oversized}"
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


class _AlgorithmSelectionAmbiguityClaim(Claim[None]):
    """Claim: overlapping algorithm ownership is an error, not a ranking."""

    @property
    def description(self) -> str:
        return "algorithm_capabilities/overlap_is_ambiguous"

    def check(self, _calibration: None) -> None:
        contract = AlgorithmStructureContract(frozenset(), frozenset())
        capabilities = (
            AlgorithmCapability("first", "First", "same", contract),
            AlgorithmCapability("second", "Second", "same", contract),
        )
        with pytest.raises(ValueError, match="ambiguous algorithm request"):
            AlgorithmRegistry(capabilities).select(AlgorithmRequest())


class _TestFileStructureClaim(Claim[None]):
    """Claim: top-level test functions are claim-dispatch verification axes."""

    _ALLOWED_AXES = {"test_correctness", "test_convergence", "test_performance"}
    _EXEMPT_FILES = {"test_structure.py"}

    @property
    def description(self) -> str:
        return "test_pattern/claim_dispatch_modules"

    def check(self, _calibration: None) -> None:
        self._assert_no_top_level_default_backend_mutation()
        violations = []
        for path in _TEST_FILES:
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if not (
                    isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                ):
                    continue
                violations.extend(self._test_function_violations(path, node))
        if violations:
            raise AssertionError(
                "test module structure violations: " + "; ".join(violations)
            )

    @classmethod
    def _test_function_violations(
        cls,
        path: Path,
        node: ast.FunctionDef,
    ) -> list[str]:
        violations = []
        if not cls._has_parametrize(node):
            violations.append(f"{path.name}.{node.name}: missing @parametrize")
        violations.extend(cls._dispatch_violations(path, node))
        if path.name not in cls._EXEMPT_FILES and node.name not in cls._ALLOWED_AXES:
            allowed = ", ".join(sorted(cls._ALLOWED_AXES))
            violations.append(
                f"{path.name}.{node.name}: outside module-owned axes ({allowed})"
            )
        return violations

    @staticmethod
    def _has_parametrize(node: ast.FunctionDef) -> bool:
        return any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr == "parametrize"
            for d in node.decorator_list
        )

    @staticmethod
    def _dispatch_violations(path: Path, node: ast.FunctionDef) -> list[str]:
        body = node.body
        if len(body) != 1:
            return [f"{path.name}.{node.name}: {len(body)} statements in body"]
        stmt = body[0]
        if not isinstance(stmt, ast.Expr):
            return [f"{path.name}.{node.name}: body is not an expression statement"]
        call = stmt.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "check"
        ):
            return [f"{path.name}.{node.name}: body does not call .check()"]
        if len(call.args) != 1 or call.keywords:
            return [
                f"{path.name}.{node.name}: .check() does not receive exactly "
                "one calibration"
            ]
        return []

    @staticmethod
    def _assert_no_top_level_default_backend_mutation() -> None:
        violations = []
        for path in _TEST_FILES:
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(
                    node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                ):
                    continue
                for child in ast.walk(node):
                    if not isinstance(child, ast.Call):
                        continue
                    func = child.func
                    if isinstance(func, ast.Name) and func.id == "set_default_backend":
                        violations.append(f"{path.name}:{child.lineno}")
                    elif (
                        isinstance(func, ast.Attribute)
                        and func.attr == "set_default_backend"
                    ):
                        violations.append(f"{path.name}:{child.lineno}")
        if violations:
            raise AssertionError(
                "top-level set_default_backend() calls: "
                + ", ".join(violations)
                + "; pass explicit backends or use a fixture"
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


class _SelectorExpectationDerivationClaim(Claim[None]):
    """Claim: selector tests project expected implementations from owners."""

    @property
    def description(self) -> str:
        return "test_pattern/selector_expectation_derivation"

    def check(self, _calibration: None) -> None:
        violations = []
        for path in _TEST_FILES:
            if path.name == "test_structure.py":
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Compare):
                    continue
                expressions = (node.left, *node.comparators)
                implementation_exprs = [
                    expr for expr in expressions if self._is_implementation_attr(expr)
                ]
                if not implementation_exprs:
                    continue
                if not any(self._is_owner_projection(expr) for expr in expressions):
                    violations.append(f"{path.name}:{node.lineno}")
        assert not violations, (
            "selector implementation expectations must project an ownership "
            "witness through .owner.__name__: " + ", ".join(violations)
        )

    @staticmethod
    def _is_implementation_attr(node: ast.expr) -> bool:
        return isinstance(node, ast.Attribute) and node.attr == "implementation"

    @staticmethod
    def _is_owner_projection(node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "__name__"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "owner"
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
        reaction_network_schema = reaction_network_parameter_schema()
        map_structure_schema = map_structure_parameter_schema()

        self._assert_solve_relation_fields_are_domain_neutral()
        self._assert_descriptor_axes_have_primitive_owners(
            (
                solve_schema,
                linear_schema,
                decomposition_schema,
                reaction_network_schema,
                map_structure_schema,
            )
        )
        assert solve_schema.descriptor_fields == set(SolveRelationField)
        assert {axis.field for axis in linear_schema.axes} == set(
            SolveRelationField
        ) | set(LinearSolverField)
        assert linear_schema.auxiliary_fields == set(DecompositionField)
        assert linear_schema.descriptor_fields == (
            set(SolveRelationField) | set(LinearSolverField) | set(DecompositionField)
        )
        assert reaction_network_schema.descriptor_fields == set(ReactionNetworkField)
        assert map_structure_schema.descriptor_fields == set(MapStructureField)
        for schema in (
            solve_schema,
            linear_schema,
            decomposition_schema,
            reaction_network_schema,
            map_structure_schema,
        ):
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
        explicit_time_step = self._explicit_time_step_descriptor()
        implicit_stage_solve = self._implicit_stage_descriptor()

        for descriptor in (
            linear_system,
            least_squares,
            nonlinear_root,
            eigenproblem,
            explicit_time_step,
            implicit_stage_solve,
        ):
            solve_schema.validate_descriptor(descriptor)
        assert solve_regions["linear_system"].contains(linear_system)
        assert solve_regions["linear_system"].contains(explicit_time_step)
        assert solve_regions["least_squares"].contains(least_squares)
        assert solve_regions["nonlinear_root"].contains(nonlinear_root)
        assert solve_regions["nonlinear_root"].contains(implicit_stage_solve)
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
        rectangular_decomp = self._decomposition_descriptor(matrix_columns=5)
        decomposition_schema.validate_descriptor(rectangular_decomp)
        assert decomposition_schema.cell_status(rectangular_decomp, ()) == "uncovered"

        reaction_regions = self._regions(reaction_network_schema)
        reaction_descriptor = self._reaction_network_descriptor()
        reaction_network_schema.validate_descriptor(reaction_descriptor)
        assert reaction_regions["conserved_network"].contains(reaction_descriptor)
        assert reaction_regions["fully_constrained_equilibrium"].contains(
            reaction_descriptor
        )
        assert (
            reaction_network_schema.cell_status(
                self._reaction_network_descriptor(stoichiometry_rank=5),
                (),
            )
            == "invalid"
        )

        map_regions = self._regions(map_structure_schema)
        rhs_descriptor = _rhs_evaluation_descriptor()
        rhs_history_descriptor = _rhs_history_descriptor()
        nordsieck_history_descriptor = _nordsieck_history_descriptor()
        for descriptor in (
            rhs_descriptor,
            rhs_history_descriptor,
            nordsieck_history_descriptor,
        ):
            map_structure_schema.validate_descriptor(descriptor)
        assert map_regions["single_step_rhs_evaluation"].contains(rhs_descriptor)
        assert map_regions["rhs_history_state"].contains(rhs_history_descriptor)
        assert map_regions["nordsieck_state"].contains(nordsieck_history_descriptor)

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
    def _assert_solve_relation_fields_are_domain_neutral() -> None:
        assert not {field.value for field in SolveRelationField} & {
            field.value for field in LinearSolverField
        }
        assert not {
            field.value for field in set(SolveRelationField) | set(LinearSolverField)
        } & {field.value for field in DecompositionField}
        assert not {
            field.value
            for field in (
                set(SolveRelationField)
                | set(LinearSolverField)
                | set(DecompositionField)
            )
        } & {field.value for field in ReactionNetworkField}

    @staticmethod
    def _assert_descriptor_axes_have_primitive_owners(
        schemas: tuple[ParameterSpaceSchema, ...],
    ) -> None:
        field_enums = (
            SolveRelationField,
            LinearSolverField,
            DecompositionField,
            ReactionNetworkField,
            MapStructureField,
        )
        fields = [field for enum_type in field_enums for field in enum_type]
        assert len({field.value for field in fields}) == len(fields)
        for schema in schemas:
            assert all(isinstance(axis.field, Enum) for axis in schema.axes)

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
        field = SolveRelationField
        return ParameterDescriptor(
            {
                field.DIM_X: DescriptorCoordinate(dim_x),
                field.DIM_Y: DescriptorCoordinate(dim_y),
                field.AUXILIARY_SCALAR_COUNT: DescriptorCoordinate(
                    auxiliary_scalar_count
                ),
                field.EQUALITY_CONSTRAINT_COUNT: DescriptorCoordinate(
                    equality_constraint_count
                ),
                field.NORMALIZATION_CONSTRAINT_COUNT: DescriptorCoordinate(
                    normalization_constraint_count
                ),
                field.RESIDUAL_TARGET_AVAILABLE: DescriptorCoordinate(
                    residual_target_available
                ),
                field.TARGET_IS_ZERO: DescriptorCoordinate(target_is_zero),
                field.MAP_LINEARITY_DEFECT: DescriptorCoordinate(
                    map_linearity_defect,
                    evidence=map_linearity_evidence,  # type: ignore[arg-type]
                ),
                field.MATRIX_REPRESENTATION_AVAILABLE: DescriptorCoordinate(
                    matrix_representation_available
                ),
                field.OPERATOR_APPLICATION_AVAILABLE: DescriptorCoordinate(
                    operator_application_available
                ),
                field.DERIVATIVE_ORACLE_KIND: DescriptorCoordinate(
                    derivative_oracle_kind
                ),
                field.OBJECTIVE_RELATION: DescriptorCoordinate(objective_relation),
                field.ACCEPTANCE_RELATION: DescriptorCoordinate(acceptance_relation),
                field.REQUESTED_RESIDUAL_TOLERANCE: DescriptorCoordinate(
                    requested_residual_tolerance
                ),
                field.REQUESTED_SOLUTION_TOLERANCE: DescriptorCoordinate(
                    requested_solution_tolerance
                ),
                field.BACKEND_KIND: DescriptorCoordinate(backend_kind),
                field.DEVICE_KIND: DescriptorCoordinate(device_kind),
                field.WORK_BUDGET_FMAS: DescriptorCoordinate(work_budget_fmas),
                field.MEMORY_BUDGET_BYTES: DescriptorCoordinate(memory_budget_bytes),
            }
        )

    @staticmethod
    def _explicit_time_step_descriptor() -> ParameterDescriptor:
        runge_kutta = _resolve_dotted(
            "cosmic_foundry.computation.time_integrators.RungeKuttaIntegrator"
        )
        state_cls = _resolve_dotted(
            "cosmic_foundry.computation.time_integrators.ODEState"
        )
        state = state_cls(0.0, Tensor([1.0, 2.0], backend=_JIT_BACKEND))
        return runge_kutta(1).step_solve_relation_descriptor(
            _unknown_test_rhs, state, 0.125
        )

    @staticmethod
    def _implicit_stage_descriptor() -> ParameterDescriptor:
        implicit_runge_kutta = _resolve_dotted(
            "cosmic_foundry.computation.time_integrators.ImplicitRungeKuttaIntegrator"
        )
        state_cls = _resolve_dotted(
            "cosmic_foundry.computation.time_integrators.ODEState"
        )
        state = state_cls(0.0, Tensor([1.0, 2.0], backend=_JIT_BACKEND))
        return implicit_runge_kutta(1).step_solve_relation_descriptor(
            _unknown_test_rhs, state, 0.125
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
        singular_value_lower_bound: float = 1.0,
        condition_estimate: float = 10.0,
        rank_estimate: int = 4,
        nullity_estimate: int = 0,
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
                DecompositionField.SINGULAR_VALUE_LOWER_BOUND: DescriptorCoordinate(
                    singular_value_lower_bound
                ),
                DecompositionField.CONDITION_ESTIMATE: DescriptorCoordinate(
                    condition_estimate
                ),
                DecompositionField.MATRIX_RANK_ESTIMATE: DescriptorCoordinate(
                    rank_estimate
                ),
                DecompositionField.MATRIX_NULLITY_ESTIMATE: DescriptorCoordinate(
                    nullity_estimate
                ),
            }
        )

    @staticmethod
    def _reaction_network_descriptor(
        *,
        species_count: int = 4,
        reaction_count: int = 3,
        stoichiometry_rank: int = 3,
        conservation_law_count: int = 1,
        equilibrium_constraint_count: int = 3,
    ) -> ParameterDescriptor:
        field = ReactionNetworkField
        return ParameterDescriptor(
            {
                field.SPECIES_COUNT: DescriptorCoordinate(species_count),
                field.REACTION_COUNT: DescriptorCoordinate(reaction_count),
                field.STOICHIOMETRY_RANK: DescriptorCoordinate(stoichiometry_rank),
                field.CONSERVATION_LAW_COUNT: DescriptorCoordinate(
                    conservation_law_count
                ),
                field.EQUILIBRIUM_CONSTRAINT_COUNT: DescriptorCoordinate(
                    equilibrium_constraint_count
                ),
            }
        )

    @staticmethod
    def _constraint_aware_descriptor(
        *,
        species_count: int = 4,
        reaction_count: int = 3,
        stoichiometry_rank: int = 3,
        conserved_linear_form_count: int = 1,
        equilibrium_constraint_count: int = 3,
    ) -> ParameterDescriptor:
        reaction_field = ReactionNetworkField
        map_field = MapStructureField
        return ParameterDescriptor(
            {
                reaction_field.SPECIES_COUNT: DescriptorCoordinate(species_count),
                reaction_field.REACTION_COUNT: DescriptorCoordinate(reaction_count),
                reaction_field.STOICHIOMETRY_RANK: DescriptorCoordinate(
                    stoichiometry_rank
                ),
                reaction_field.EQUILIBRIUM_CONSTRAINT_COUNT: DescriptorCoordinate(
                    equilibrium_constraint_count
                ),
                map_field.RHS_EVALUATION_AVAILABLE: DescriptorCoordinate(True),
                map_field.RHS_HISTORY_AVAILABLE: DescriptorCoordinate(False),
                map_field.NORDSIECK_HISTORY_AVAILABLE: DescriptorCoordinate(False),
                map_field.CONSERVED_LINEAR_FORM_COUNT: DescriptorCoordinate(
                    conserved_linear_form_count
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
        spd_decomposition = decomposition_descriptor_from_linear_operator_descriptor(
            spd
        )
        decomposition_schema = decomposition_parameter_schema()
        decomposition_regions = _SolveRelationSchemaClaim._regions(decomposition_schema)
        decomposition_schema.validate_descriptor(spd_decomposition)
        assert decomposition_regions["square"].contains(spd_decomposition)
        assert decomposition_regions["full_rank"].contains(spd_decomposition)
        assert (
            spd_decomposition.coordinate(DecompositionField.MATRIX_RANK_ESTIMATE).value
            == spd.coordinate(LinearSolverField.RANK_ESTIMATE).value
        )

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


class _FiniteStateTransitionSystemClaim(Claim[None]):
    """Claim: finite directed unit transfers imply total conservation."""

    @property
    def description(self) -> str:
        return "theory/finite_state_transition_system"

    def check(self, _calibration: None) -> None:
        transition_system = _resolve_dotted(
            "cosmic_foundry.theory.discrete.FiniteStateTransitionSystem"
        )
        indexed_set = _resolve_dotted("cosmic_foundry.theory.foundation.IndexedSet")
        system = transition_system(4, ((0, 1), (1, 2), (1, 3)))

        stoichiometry = system.stoichiometry_matrix()
        conserved = system.conserved_total_form()

        assert isinstance(system, indexed_set)
        assert system.shape == (4,)
        assert system.ndim == 1
        assert system.transition_count == len(system.transitions)
        assert conserved == (1, 1, 1, 1)
        assert len(stoichiometry) == 4
        assert all(len(row) == system.transition_count for row in stoichiometry)
        assert all(
            sum(row[column] for row in stoichiometry) == 0
            for column in range(system.transition_count)
        )
        assert transition_system.chain(4).transitions == ((0, 1), (1, 2), (2, 3))
        assert system.intersect(transition_system(3, ((0, 1), (1, 2)))) == (
            transition_system(3, ((0, 1), (1, 2)))
        )


class _TimeIntegratorSolveRelationClaim(Claim[None]):
    """Claim: time-step solve relations are inferred from stage equations."""

    @property
    def description(self) -> str:
        return "algorithm_capabilities/time_integrator_solve_relation"

    def check(self, _calibration: None) -> None:
        schema = solve_relation_parameter_schema()
        state_cls = _resolve_dotted(
            "cosmic_foundry.computation.time_integrators.ODEState"
        )
        state = state_cls(0.0, Tensor([1.0, 2.0], backend=_JIT_BACKEND))
        explicit_owners = []
        implicit_owners = []
        for integrator in self._single_order_integrators():
            stage_matrix = getattr(integrator, "A_sym", ())
            if not stage_matrix:
                continue
            descriptor = integrator.step_solve_relation_descriptor(
                _unknown_test_rhs, state, 0.25
            )
            schema.validate_descriptor(descriptor)
            if self._stage_matrix_is_strictly_lower(stage_matrix):
                assert self._regions(schema)["linear_system"].contains(descriptor)
                assert descriptor.coordinate(SolveRelationField.DIM_X).value == 2
                assert descriptor.coordinate(SolveRelationField.DIM_Y).value == 2
                assert (
                    descriptor.coordinate(SolveRelationField.MAP_LINEARITY_DEFECT).value
                    == 0.0
                )
                explicit_owners.append(type(integrator))
            elif self._stage_matrix_has_implicit_coupling(stage_matrix):
                assert self._regions(schema)["nonlinear_root"].contains(descriptor)
                assert descriptor.coordinate(SolveRelationField.DIM_X).value == (
                    2 * len(stage_matrix)
                )
                assert descriptor.coordinate(SolveRelationField.DIM_Y).value == (
                    2 * len(stage_matrix)
                )
                assert (
                    descriptor.coordinate(
                        SolveRelationField.MAP_LINEARITY_DEFECT
                    ).evidence
                    == "unavailable"
                )
                assert (
                    descriptor.coordinate(
                        SolveRelationField.DERIVATIVE_ORACLE_KIND
                    ).value
                    == "jacobian_callback"
                )
                implicit_owners.append(type(integrator))
                affine_descriptor = integrator.step_solve_relation_descriptor(
                    _AffineTestRHS(), state, 0.25
                )
                schema.validate_descriptor(affine_descriptor)
                assert self._regions(schema)["linear_system"].contains(
                    affine_descriptor
                )
                assert affine_descriptor.coordinate(SolveRelationField.DIM_X).value == (
                    2 * len(stage_matrix)
                )
                assert (
                    affine_descriptor.coordinate(
                        SolveRelationField.MAP_LINEARITY_DEFECT
                    ).value
                    == 0.0
                )
                assert (
                    affine_descriptor.coordinate(
                        SolveRelationField.DERIVATIVE_ORACLE_KIND
                    ).value
                    == "matrix"
                )
                linear_descriptor = integrator.step_linear_operator_descriptor(
                    _AffineTestRHS(), state, 0.25
                )
                linear_schema = linear_solver_parameter_schema()
                linear_regions = self._regions(linear_schema)
                linear_schema.validate_descriptor(
                    linear_descriptor.parameter_descriptor
                )
                assert linear_regions["linear_system"].contains(
                    linear_descriptor.parameter_descriptor
                )
                assert linear_regions["full_rank"].contains(
                    linear_descriptor.parameter_descriptor
                )
                assert (
                    linear_descriptor.coordinate(
                        LinearSolverField.RHS_CONSISTENCY_DEFECT
                    ).value
                    == 0.0
                )
                selected = select_linear_solver_for_descriptor(
                    linear_descriptor.parameter_descriptor
                )
                dense_lu_solver = _resolve_dotted(
                    "cosmic_foundry.computation.solvers.DenseLUSolver"
                )
                assert selected is dense_lu_solver
        assert explicit_owners
        assert implicit_owners

    @staticmethod
    def _single_order_integrators() -> tuple[Any, ...]:
        instances = []
        for cls in _TIME_INTEGRATORS:
            parameters = tuple(inspect.signature(cls).parameters.values())
            required = tuple(
                parameter
                for parameter in parameters
                if parameter.default is inspect.Parameter.empty
            )
            if len(required) != 1:
                continue
            try:
                instances.append(cls(1))
            except (TypeError, ValueError):
                continue
        return tuple(instances)

    @staticmethod
    def _stage_matrix_is_strictly_lower(matrix: Any) -> bool:
        return bool(matrix) and all(
            entry == 0
            for row_index, row in enumerate(matrix)
            for column_index, entry in enumerate(row)
            if column_index >= row_index
        )

    @staticmethod
    def _stage_matrix_has_implicit_coupling(matrix: Any) -> bool:
        return bool(matrix) and any(
            entry != 0
            for row_index, row in enumerate(matrix)
            for column_index, entry in enumerate(row)
            if column_index >= row_index
        )

    @staticmethod
    def _regions(
        schema: ParameterSpaceSchema,
    ) -> dict[str, DerivedParameterRegion]:
        return {region.name: region for region in schema.derived_regions}


class _LinearSolverCoverageRegionClaim(Claim[None]):
    """Claim: linear-solver selection is driven by schema coverage regions."""

    @property
    def description(self) -> str:
        return "algorithm_capabilities/linear_solver_coverage_regions"

    def check(self, _calibration: None) -> None:
        self._assert_no_local_disjointness_algebra()
        self._assert_no_descriptor_field_translation_tables()
        self._assert_no_legacy_coverage_region_names()
        self._assert_coverage_region_identity_is_owner_class()
        self._assert_no_coverage_region_priority_model()
        self._assert_no_coverage_region_status_model()
        self._assert_no_solver_local_point_query()
        self._assert_no_solver_error_string_dispatch()
        self._assert_no_linear_solver_field_text_identity()
        schema = linear_solver_parameter_schema()
        assert (
            set(SolveRelationField) | set(LinearSolverField) | set(DecompositionField)
            == schema.descriptor_fields
        )
        regions = linear_solver_coverage_regions()
        self._assert_final_solve_coverage_owners_are_linear_solvers(regions)
        self._assert_stationary_iterations_do_not_own_final_solve_regions(regions)
        self._assert_no_declared_coverage_literals(regions)
        for region in regions:
            schema.validate_coverage_region(region)
        self._assert_linear_solver_coverage_is_square_residual_solve(schema, regions)
        self._assert_direct_solver_coverage_uses_decomposition_certificate(regions)
        self._assert_final_solve_owners_are_schema_separated(regions)

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
    def _assert_linear_solver_coverage_is_square_residual_solve(
        cls,
        schema: ParameterSpaceSchema,
        regions: tuple[CoverageRegion, ...],
    ) -> None:
        least_squares = _SolveRelationSchemaClaim._linear_descriptor(
            objective_relation="least_squares"
        )
        overdetermined_least_squares = _SolveRelationSchemaClaim._linear_descriptor(
            dim_y=cls._two() + cls._one(),
            objective_relation="least_squares",
            symmetry_defect=cls._one(),
            coercivity_lower_bound=cls._zero(),
        )
        for descriptor in (least_squares, overdetermined_least_squares):
            schema.validate_descriptor(descriptor)
            assert schema.cell_status(descriptor, regions) == "uncovered"

    @staticmethod
    def _assert_final_solve_coverage_owners_are_linear_solvers(
        regions: tuple[CoverageRegion, ...],
    ) -> None:
        linear_solver = _resolve_dotted(
            "cosmic_foundry.computation.solvers.LinearSolver"
        )
        least_squares_solver = _resolve_dotted(
            "cosmic_foundry.computation.solvers.LeastSquaresSolver"
        )
        for region in regions:
            assert issubclass(region.owner, linear_solver)
            assert not issubclass(region.owner, least_squares_solver)
        for solver in _LEAST_SQUARES_SOLVERS:
            assert not issubclass(solver, linear_solver)
            assert not hasattr(solver, "linear_solver_coverage")

    @staticmethod
    def _assert_direct_solver_coverage_uses_decomposition_certificate(
        regions: tuple[CoverageRegion, ...],
    ) -> None:
        direct_solver = _resolve_dotted(
            "cosmic_foundry.computation.solvers.DirectSolver"
        )
        direct_regions = [
            region for region in regions if issubclass(region.owner, direct_solver)
        ]
        assert direct_regions
        for region in direct_regions:
            assert "linear_solver_coverage" not in region.owner.__dict__
            decomposition_type = region.owner.decomposition_type
            assert decomposition_type is not None
            assert "linear_solve_certificate" not in decomposition_type.__dict__
            certificate = decomposition_type.factorization_feasibility_certificate
            assert certificate
            for predicate in certificate:
                assert predicate.referenced_fields <= set(DecompositionField)
                assert isinstance(predicate, ComparisonPredicate)
                assert predicate in region.predicates

    @staticmethod
    def _assert_stationary_iterations_do_not_own_final_solve_regions(
        regions: tuple[CoverageRegion, ...],
    ) -> None:
        assert not any(
            issubclass(region.owner, StationaryIterationSolver) for region in regions
        )

    @staticmethod
    def _assert_final_solve_owners_are_schema_separated(
        regions: tuple[CoverageRegion, ...],
    ) -> None:
        assert len({region.owner for region in regions}) == len(regions)
        assert len(
            {_coverage_region_predicate_key(region) for region in regions}
        ) == len(regions)
        for index, left in enumerate(regions):
            for right in regions[index + 1 :]:
                assert predicate_sets_are_disjoint(left.predicates, right.predicates)

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
    def _assert_no_descriptor_field_translation_tables() -> None:
        module_name = "cosmic_foundry.computation.solvers.capabilities"
        module = importlib.import_module(module_name)
        source = _PACKAGE_ROOT / "computation" / "solvers" / "capabilities.py"
        tree = ast.parse(source.read_text())
        descriptor_field_types = (
            SolveRelationField,
            LinearSolverField,
            DecompositionField,
        )
        translations = []
        for node in tree.body:
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            evaluated = eval(
                compile(ast.Expression(node.value), str(source), "eval"),
                module.__dict__,
            )
            if (
                isinstance(evaluated, dict)
                and evaluated
                and all(
                    isinstance(key, descriptor_field_types)
                    and isinstance(value, descriptor_field_types)
                    for key, value in evaluated.items()
                )
            ):
                translations.append(ast.unparse(node.value))
        assert not translations, (
            "solver coverage must compose descriptor projections instead of "
            f"translating descriptor fields through dict literals: {translations}"
        )

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
        field_labels = {
            field.value
            for enum_type in (SolveRelationField, LinearSolverField)
            for field in enum_type
        }
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
        fields = cls._descriptor_axes(schema)
        coordinates = {
            field: DescriptorCoordinate(cls._axis_witness(axis))
            for field, axis in fields.items()
            if field in {schema_axis.field for schema_axis in schema.axes}
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

    @staticmethod
    def _descriptor_axes(
        schema: ParameterSpaceSchema,
    ) -> dict[DescriptorField, ParameterAxis]:
        auxiliary_schemas = (
            solve_relation_parameter_schema(),
            linear_solver_parameter_schema(),
            decomposition_parameter_schema(),
        )
        axes = {axis.field: axis for axis in schema.axes}
        for auxiliary_schema in auxiliary_schemas:
            for axis in auxiliary_schema.axes:
                if axis.field in schema.descriptor_fields:
                    axes.setdefault(axis.field, axis)
        return axes

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


_structure_atlas = importlib.import_module("tests.structure_atlas")
_CapabilityAtlasDocClaim = _structure_atlas._CapabilityAtlasDocClaim
_coverage_region_name = _structure_atlas._coverage_region_name
_coverage_region_predicate_key = _structure_atlas._coverage_region_predicate_key
_field_label = _structure_atlas._field_label

# ---------------------------------------------------------------------------
# Auto-discovery and registry
# ---------------------------------------------------------------------------

_MODULES = _discover_modules()
_ABCS = _discover_abcs(_MODULES)
_HIERARCHY_PAIRS = _discover_hierarchy_pairs(_ABCS)
_ITERATIVE_SOLVERS = _discover_concrete_iterative_solvers(_MODULES)
_MATRIX_FREE_ITERATIVE_SOLVERS = _discover_matrix_free_iterative_solvers(_MODULES)
_FACTORIZATIONS = _discover_concrete_factorizations(_MODULES)
_LEAST_SQUARES_SOLVERS = _discover_concrete_least_squares_solvers(_MODULES)
_TIME_INTEGRATORS = _discover_concrete_time_integrators(_MODULES)
_TEST_FILES = sorted(Path(__file__).parent.glob("test_*.py"))

_AlgorithmRequest = _resolve_dotted(
    "cosmic_foundry.computation.algorithm_capabilities.AlgorithmRequest"
)
_derivative_oracle_descriptor = _resolve_dotted(
    "cosmic_foundry.computation.time_integrators.capabilities."
    "derivative_oracle_descriptor"
)
_semilinear_map_descriptor = _resolve_dotted(
    "cosmic_foundry.computation.time_integrators.capabilities."
    "semilinear_map_descriptor"
)
_split_map_descriptor = _resolve_dotted(
    "cosmic_foundry.computation.time_integrators.capabilities.split_map_descriptor"
)
_hamiltonian_map_descriptor = _resolve_dotted(
    "cosmic_foundry.computation.time_integrators.capabilities."
    "hamiltonian_map_descriptor"
)
_composite_map_descriptor = _resolve_dotted(
    "cosmic_foundry.computation.time_integrators.capabilities."
    "composite_map_descriptor"
)
_rhs_evaluation_descriptor = _resolve_dotted(
    "cosmic_foundry.computation.time_integrators.capabilities."
    "rhs_evaluation_descriptor"
)
_rhs_history_descriptor = _resolve_dotted(
    "cosmic_foundry.computation.time_integrators.capabilities." "rhs_history_descriptor"
)
_nordsieck_history_descriptor = _resolve_dotted(
    "cosmic_foundry.computation.time_integrators.capabilities."
    "nordsieck_history_descriptor"
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
                "select_time_integrator",
                "time_integrator_step_linear_operator_descriptor",
                "time_integrator_step_solve_relation_descriptor",
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
                "AffineRHSProtocol",
                "BlackBoxRHS",
                "ComponentFlowRHS",
                "ComponentFlowProtocol",
                "CompositeRHS",
                "CompositeRHSProtocol",
                "FiniteDiffJacobianRHS",
                "HamiltonianRHS",
                "HamiltonianRHSProtocol",
                "JacobianRHS",
                "LinearReactionNetworkRHS",
                "ReactionNetworkRHS",
                "RHSProtocol",
                "SemilinearRHS",
                "SemilinearRHSProtocol",
                "SplitRHS",
                "SplitRHSProtocol",
                "UnitTransferRates",
                "UnitTransferTransitionSystemProtocol",
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
            _AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=4,
                descriptor=_rhs_evaluation_descriptor(),
            ),
            "RungeKuttaIntegrator",
        ),
        _CapabilityRequestExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=4,
                descriptor=_rhs_history_descriptor(),
            ),
            "ExplicitMultistepIntegrator",
        ),
        _CapabilityRequestExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=4,
                descriptor=_nordsieck_history_descriptor(),
            ),
            "MultistepIntegrator",
        ),
        _CapabilityRequestExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=2,
                descriptor=_derivative_oracle_descriptor(),
            ),
            "ImplicitRungeKuttaIntegrator",
        ),
        _CapabilityRequestExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=3,
                descriptor=_split_map_descriptor(),
            ),
            "AdditiveRungeKuttaIntegrator",
        ),
        _CapabilityRequestExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=4,
                descriptor=_semilinear_map_descriptor(),
            ),
            "LawsonRungeKuttaIntegrator",
        ),
        _CapabilityRequestExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=4,
                descriptor=_hamiltonian_map_descriptor(),
            ),
            "SymplecticCompositionIntegrator",
        ),
        _CapabilityRequestExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=4,
                descriptor=_composite_map_descriptor(2),
            ),
            "CompositionIntegrator",
        ),
        _CapabilityRequestExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"advance"}),
                order=2,
                descriptor=_derivative_oracle_descriptor(),
            ),
            "AdaptiveNordsieckController",
        ),
        _CapabilityRequestExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"advance"}),
                order=3,
                descriptor=_rhs_evaluation_descriptor(),
            ),
            "IntegrationDriver",
        ),
        _CapabilityRequestExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"advance"}),
                order=2,
                descriptor=_SolveRelationSchemaClaim._constraint_aware_descriptor(),
            ),
            "ConstraintAwareController",
        ),
    ),
    rejected_requests=(
        _CapabilityRejectionExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=3,
                descriptor=_hamiltonian_map_descriptor(),
            )
        ),
        _CapabilityRejectionExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=3,
                descriptor=_composite_map_descriptor(2),
            )
        ),
        _CapabilityRejectionExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"advance"}),
                order=2,
            )
        ),
        _CapabilityRejectionExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=2,
            )
        ),
        _CapabilityRejectionExpectation(
            _AlgorithmRequest(
                requested_properties=frozenset({"advance", "nordsieck"}),
                order=2,
            )
        ),
    ),
    descriptor_owned_capabilities=True,
    descriptor_request_property_limit=1,
    expected_class_modules={
        "AdaptiveNordsieckController": "adaptive_nordsieck",
        "AdditiveRungeKuttaIntegrator": "imex",
        "AlgorithmStructureContract": "algorithm_capabilities",
        "AutoIntegrator": "auto",
        "BlackBoxRHS": "integrator",
        "ComponentFlowProtocol": "splitting",
        "ComponentFlowRHS": "splitting",
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
        "LinearReactionNetworkRHS": "reaction_network",
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
        "UnitTransferTransitionSystemProtocol": "reaction_network",
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
                "LeastSquaresSolver",
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
        "least_squares_solver": frozenset({"DenseSVDLeastSquaresSolver"}),
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
        "DenseSVDLeastSquaresSolver": "least_squares_solver",
        "DenseSVDSolver": "dense_svd_solver",
        "DirectSolver": "direct_solver",
        "IterativeSolver": "iterative_solver",
        "KrylovSolver": "iterative_solver",
        "LeastSquaresSolver": "least_squares_solver",
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
        "DenseSVDLeastSquaresSolver": ("Dense", "Solver"),
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
                "FiniteStateTransitionSystem",
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
    _AbcInstantiationClaim(),
    _HierarchyClaim(),
    _ModuleAllClaim(),
    *[_IterativeSolverJitClaim(cls) for cls in _MATRIX_FREE_ITERATIVE_SOLVERS],
    *[_MaterializationGateClaim(cls) for cls in _MATRIX_FREE_ITERATIVE_SOLVERS],
    *[_FactorizationJitClaim(cls) for cls in _FACTORIZATIONS],
    _GenericBasesClaim(),
    _ManifoldIsolationClaim(),
    _ImportBoundaryClaim(),
    _TestFileStructureClaim(),
    _ArchitectureOwnershipClaim(_TIME_INTEGRATOR_OWNERSHIP),
    _ArchitectureOwnershipClaim(_LINEAR_SOLVER_OWNERSHIP),
    _ArchitectureOwnershipClaim(_DECOMPOSITION_OWNERSHIP),
    _ArchitectureOwnershipClaim(_DISCRETE_OPERATOR_OWNERSHIP),
    _ArchitectureOwnershipClaim(_GEOMETRY_OWNERSHIP),
    _AlgorithmSelectionAmbiguityClaim(),
    _AutoDiscoveryImportClaim(),
    _SelectorExpectationDerivationClaim(),
    _ParameterSpaceSchemaClaim(),
    _SolveRelationSchemaClaim(),
    _LinearOperatorDescriptorClaim(),
    _FiniteStateTransitionSystemClaim(),
    _TimeIntegratorSolveRelationClaim(),
    _LinearSolverCoverageLocalityClaim(),
    _LinearSolverCoverageRegionClaim(),
    _CapabilityAtlasDocClaim(),
]


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_structure(claim: Claim[None]) -> None:
    claim.check(None)
