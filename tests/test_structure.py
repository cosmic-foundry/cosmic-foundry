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
  _TestAxisConventionClaim  — module tests use correctness/convergence/performance
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import sympy

from cosmic_foundry.computation.backends.python_backend import PythonBackend
from cosmic_foundry.computation.decompositions.factorization import Factorization
from cosmic_foundry.computation.solvers.iterative_solver import IterativeSolver
from cosmic_foundry.computation.tensor import MaterializationError, Tensor
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.theory.continuous.manifold import Manifold
from cosmic_foundry.theory.discrete import (
    DiffusiveFlux,
    DirichletGhostCells,
    DivergenceFormDiscretization,
)
from cosmic_foundry.theory.foundation.indexed_set import IndexedSet
from tests.claims import Claim, assemble_linear_op

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


def _make_jit_op() -> Any:
    mesh = CartesianMesh(
        origin=(sympy.Rational(0),),
        spacing=(sympy.Rational(1, _JIT_N),),
        shape=(_JIT_N,),
    )
    flux = DiffusiveFlux(DiffusiveFlux.min_order, EuclideanManifold(1))
    return assemble_linear_op(
        DivergenceFormDiscretization(flux, DirichletGhostCells()), mesh
    )


_JIT_OP = _make_jit_op()
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
    _AutoDiscoveryImportClaim(),
]


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_structure(claim: Claim[None]) -> None:
    claim.check(None)
