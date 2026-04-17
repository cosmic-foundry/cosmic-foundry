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

Reference the introducing PR number in the **PR** column.
Update the **Test** column if the validating test moves.
Mark the **Source** column with the paper's author-year shorthand
and equation number, or "standard" with a representative textbook.

---

## Entries

### F-001 — 3-D second-order 7-point Laplacian stencil

| Field | Value |
|---|---|
| **Module** | `cosmic_foundry/kernels/__init__.py` — `seven_point_laplacian` example Op; `benchmarks/pointwise_roofline.py` |
| **Formula** | `(φᵢ₋₁,ⱼ,ₖ + φᵢ₊₁,ⱼ,ₖ + φᵢ,ⱼ₋₁,ₖ + φᵢ,ⱼ₊₁,ₖ + φᵢ,ⱼ,ₖ₋₁ + φᵢ,ⱼ,ₖ₊₁ − 6φᵢ,ⱼ,ₖ) / h²` |
| **Source** | Standard second-order finite difference; see LeVeque (2007) *Finite Difference Methods*, §1.1 |
| **External grounding** | Analytical: ∇²(x²+y²+z²) = 6 exactly. Test sets φ = x²+y²+z² and asserts the stencil returns 6.0 to float64 tolerance. |
| **Non-obvious detail** | The kernel as implemented returns the **un-divided** stencil sum (numerator only). The caller supplies the grid spacing h and divides. In the Epoch 1 test h = 1 so the raw sum equals the Laplacian; downstream callers must not forget the 1/h² factor. |
| **Test** | `tests/test_kernels.py::test_dispatch_executes_laplacian_over_region` |
| **PR** | #53 |

---

## What to expect in future epochs

The entries worth tracking will cluster in:

- **Epoch 4 (hydro):** Riemann solver variant (HLL / HLLC / HLLD wave-speed
  estimate), reconstruction slope limiter (van Leer / MC / PPM with its
  contact-steepener coefficient C₂ = 1.25 from Colella & Woodward 1984 Table 1).
- **Epoch 6 (microphysics):** Every rate in the Aprox13/19 network is a
  polynomial of the form exp(a₀ + a₁/T₉ + a₂/T₉^{1/3} + …); each is a
  separate formula entry. EOS polynomial fit coefficients (Timmes & Swesty
  2000, Tables 1–4).
- **Epoch 7 (MHD):** Constrained-transport electric-field averaging
  (Evans & Hawley 1988 §3), HLLD multi-wave structure (Miyoshi & Kusano
  2005 §3.1).
- **Epoch 8 (radiation):** Flux-limited diffusion limiter λ(R) (Levermore &
  Pomraning 1981 eq. 28), M1 Eddington tensor closure.
- **Epoch 9 (GR):** BSSN conformal decomposition with gauge parameters
  (η, ξ in the Gamma-driver shift); specific values vary between the
  Alcubierre (2008) and Baumgarte & Shapiro (2010) presentations.
