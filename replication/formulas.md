# Formula Register

A flat cross-reference of every physics formula the engine implements.
One entry per discrete formula. A *formula* here means a specific
equation (a stencil, a flux expression, a rate, an EOS fit) whose
implementation could silently encode a wrong answer — wrong sign,
missing term, wrong coefficient, wrong variant chosen from several
in the literature.

This register complements the capability specs in `capabilities/`.
A capability spec describes what a feature does and how to verify it;
this register maps from individual formulas to their authoritative
sources and their tests, at finer granularity than a capability.

## How to add an entry

Add a row when a PR introduces a formula that:

- has at least one published variant the implementation might
  silently choose differently (Riemann solver wave-speed estimate,
  flux limiter, gauge parameter), or
- contains non-obvious coefficients that must match a specific
  paper (PPM steepener, polynomial EOS fit, nuclear rate network), or
- has a sign or index convention that differs across sources.

Standard textbook operators with no free parameters (a centered
second-order Laplacian, a forward Euler step) qualify only if
something about the implementation is non-obvious — a missing
division by h², an unusual indexing convention, etc.

**Citation convention.** The **Source** field must use a DOI link
(`https://doi.org/<DOI>`) for journal papers and conference
proceedings. For books, use the DOI if one exists (most SIAM and
Springer titles have them); otherwise use the ISBN. Bare author-year
strings without a stable identifier are not acceptable — they rot
within a few years as PDFs move. If you cannot locate a DOI before
opening the PR, use an ADS permalink
(`https://ui.adsabs.harvard.edu/abs/<bibcode>`) as a fallback for
astrophysics papers; ADS bibcodes are stable indefinitely.

Reference the introducing PR number in the **PR** column.
Update the **Test** column if the validating test moves.

---

## Entries

### F-001 — 3-D second-order 7-point Laplacian stencil

| Field | Value |
|---|---|
| **Module** | `cosmic_foundry/kernels/__init__.py` — `seven_point_laplacian` example Op; `benchmarks/pointwise_roofline.py` |
| **Formula** | `(φᵢ₋₁,ⱼ,ₖ + φᵢ₊₁,ⱼ,ₖ + φᵢ,ⱼ₋₁,ₖ + φᵢ,ⱼ₊₁,ₖ + φᵢ,ⱼ,ₖ₋₁ + φᵢ,ⱼ,ₖ₊₁ − 6φᵢ,ⱼ,ₖ) / h²` |
| **Source** | Standard second-order finite difference; formula appears in virtually every numerical methods textbook and requires no single authoritative citation. |
| **External grounding** | Analytical: ∇²(x²+y²+z²) = 6 exactly. Test sets φ = x²+y²+z² and asserts the stencil returns 6.0 to float64 tolerance. No paper needed — the expected value is derived by hand in two lines. |
| **Non-obvious detail** | The kernel as implemented returns the **un-divided** stencil sum (numerator only). The caller supplies the grid spacing h and divides. In the Epoch 1 test h = 1 so the raw sum equals the Laplacian; downstream callers must not forget the 1/h² factor. |
| **Test** | `tests/test_kernels.py::test_dispatch_executes_laplacian_over_region` |
| **PR** | #53 |

---

## What to expect in future epochs

Papers listed here will become formal entries when the corresponding
code lands. DOIs are provided now so they are not looked up under
pressure during a PR review.

- **Epoch 4 (hydro):** Riemann solver variant selection and
  reconstruction coefficients.
  - Colella & Woodward (1984), "The Piecewise Parabolic Method (PPM)
    for gas-dynamical simulations" — PPM steepener coefficient C₂ = 1.25
    (Table 1) and monotonicity constraints.
    <https://doi.org/10.1016/0021-9991(84)90143-8>
  - Toro, Spruce & Speares (1994), "Restoration of the contact surface
    in the HLL-Riemann solver" — HLLC wave-speed structure.
    <https://doi.org/10.1007/BF00416935>

- **Epoch 6 (microphysics):** Rate network and EOS polynomial
  coefficients. Every rate in Aprox13/19 is a separate entry.
  - Timmes & Swesty (2000), "The accuracy, consistency, and speed of an
    electron-positron equation of state based on table interpolation of
    the Helmholtz free energy" — EOS polynomial fit coefficients,
    Tables 1–4.
    <https://doi.org/10.1086/313304>
  - Cyburt et al. (2010), "The JINA REACLIB database: its recent updates
    and impact on type I X-ray bursts" — rate parameterization form used
    by Aprox networks.
    <https://doi.org/10.1088/0067-0049/189/1/240>

- **Epoch 7 (MHD):** Constrained transport and multi-state Riemann solver.
  - Evans & Hawley (1988), "Simulation of magnetohydrodynamic flows:
    a constrained transport method" — CT electric-field averaging, §3.
    <https://ui.adsabs.harvard.edu/abs/1988ApJ...332..659E>
  - Miyoshi & Kusano (2005), "A multi-state HLL approximate Riemann
    solver for ideal magnetohydrodynamics" — HLLD wave structure, §3.1.
    <https://doi.org/10.1016/j.jcp.2005.02.017>

- **Epoch 8 (radiation):** Diffusion limiter and M1 closure.
  - Levermore & Pomraning (1981), "A flux-limited diffusion theory" —
    flux limiter λ(R), eq. 28.
    <https://ui.adsabs.harvard.edu/abs/1981ApJ...248..321L>

- **Epoch 9 (GR):** BSSN gauge parameters. Specific values for η and ξ
  in the Gamma-driver shift vary between standard references; the entry
  must name which presentation the implementation follows.
  - Baumgarte & Shapiro (1999), "Numerical integration of Einstein's
    field equations" — original BSSN decomposition.
    <https://doi.org/10.1103/PhysRevD.59.024007>
  - Nakamura, Oohara & Kojima (1987), "General relativistic collapse to
    black holes and gravitational waves from black holes" — NOK conformal
    decomposition predating BSSN.
    <https://doi.org/10.1143/PTPS.90.1>
