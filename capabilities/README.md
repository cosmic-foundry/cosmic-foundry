# Capability Structure

This document defines how capabilities are conceptually organized and what
external grounding looks like at each level. It supersedes any implicit
layering in the epoch roadmap or existing capability specs.

## The core framing

A capability is a claim about an object or a map that an independent actor
could verify without access to the authors or the implementation history.
The kind of object or map determines what verification looks like.

### Objects

There are three kinds of objects in this codebase:

- **Mathematical objects** — both continuous (functions, operators, integrals,
  differential equations) and discrete (vectors, matrices, transforms,
  sequences). Discrete mathematics is first-class, not a subordinate of
  continuous mathematics.

- **Physical fields and equations** — statements about physical reality,
  expressed in the continuous world. The Navier-Stokes equations, Newton's
  law of gravitation, the Euler equations — these are continuous objects.
  Their correctness is independent of any discretization.

- **Computable representations** — finite arrays, sparse matrices, grid
  functions. These are what actually runs on hardware.

### Maps

Capabilities often live at the boundaries between object kinds, as maps:

- **Discretization maps** — from continuous objects to computable
  representations, parameterized by a resolution h. A discretization map
  has a correctness claim: convergence to the continuous object at a stated
  rate as h → 0.

- **Physical modeling maps** — from mathematical objects to physical ones.
  For example, the map from the inverse-square law (mathematics) to
  Newtonian gravitation (physics).

- **Numerical realization maps** — from a discretization to a specific
  computable algorithm. For example, the map from "solve a linear system"
  to "apply conjugate gradient."

This framing resolves the apparent layering problem. Linear algebra, the
DFT, and spherical harmonics are mathematical objects, not discretization
artifacts. A linear solver is a numerical realization map. A finite-difference
stencil is a discretization map. None of these fit cleanly into a simple
hierarchy — they are objects and maps at different levels, composable in
multiple ways.

## Verification by kind

The external grounding discipline differs by kind:

- **Mathematical objects and maps:** verified by symbolic computation, proof,
  or exact numerical evaluation. No convergence test needed — the claim is
  exact.

- **Physical equations:** verified against known analytical solutions of the
  continuous equation. The verification does not depend on the discretization.

- **Discretization maps:** verified by convergence to the continuous solution
  at the stated rate as h → 0. The continuous solution must itself be
  externally grounded (analytical or published), not engine-generated.

- **Numerical realization maps:** verified by agreement with the discretization
  they claim to realize, to within rounding error.

## The concrete anchor

The capability hierarchy for this project is grounded in a specific physical
scenario: a 3D double-degenerate white dwarf collision leading to a Type Ia
supernova candidate, in the pre-light-curve phase where nuclear astrophysics
dominates and radiation effects are secondary.

The top-level physical capabilities required are:

1. **Compressible fluid dynamics** — the carrier; evolves the fluid state
2. **Degenerate electron equation of state** — closes the fluid system
3. **Thermodynamic nuclear reactions** — modifies composition and energy
4. **Gravitational field** — forward (analytical/semi-analytical) and
   backward (Poisson solver with isolated boundary conditions)

Each of these depends on discretization maps and mathematical objects that
are specified and verified independently. The physical capabilities are
expressed in the continuous world; discretization is the necessary step to
make them computable, and its correctness is verified separately.

## What this means for the epoch roadmap

The epoch roadmap sequences delivery of these capabilities. The capability
structure document describes what each capability *is* and how it is
verified. When the two conflict, this document takes precedence for
capability definitions; the epoch roadmap takes precedence for sequencing.

Existing capability specs (C0001–C0009) predate this framing and should be
audited against it as part of M3.
