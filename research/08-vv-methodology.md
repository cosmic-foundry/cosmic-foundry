# 8. Verification, Validation, and Specification Methodology

> Part of the [Cosmic Foundry research notes](index.md).

This section surveys the V&V literature across CFD, aerospace, nuclear, and
computational astrophysics. The research notes previously had no coverage of
V&V methodology — the gap this section fills. The key finding is that while
the field has well-developed verification techniques (especially MMS), the
specific idea of writing a formal capability specification *before* the
implementation — and treating that specification as the primary correctness
reference — has no established name in the literature and is only partially
approached in a few institutional contexts.

---

## 8.1 Foundational V&V Framework

The modern CFD V&V framework was established primarily by two groups:
Roache (independent consultant / Hermosa Publishers) in the 1990s, and
Oberkampf & Trucano at Sandia National Laboratories in the 2000s.

**Patrick J. Roache**, "Quantification of Uncertainty in Computational Fluid
Dynamics," *Annual Review of Fluid Mechanics*, 29:123–160, 1997.
<https://doi.org/10.1146/annurev.fluid.29.1.123>
Established the distinction between verification (did we solve the equations
right?) and validation (did we solve the right equations?), and introduced
the Grid Convergence Index as a standardized convergence reporting method.
This is the paper that gave the field its vocabulary.

**Patrick J. Roache**, *Verification and Validation in Computational Science
and Engineering*, Hermosa Publishers, 1998. ISBN 0-913478-08-3.
The first book-length treatment. Introduced MMS as a first-class code
verification tool.

**William L. Oberkampf and Timothy G. Trucano**, "Verification and Validation
in Computational Fluid Dynamics," *Progress in Aerospace Sciences*,
38:209–272, 2002.
<https://doi.org/10.1016/S0376-0421(02)00005-2>
The most-cited single paper in CFD V&V (~840 citations). Established rigorous
terminology, distinguished code verification from solution verification and
model validation from solution validation, and proposed a taxonomy of error
and uncertainty sources.

**William L. Oberkampf and Christopher J. Roy**, *Verification and Validation
in Scientific Computing*, Cambridge University Press, 2010.
ISBN 978-0-521-11360-1.
<https://doi.org/10.1017/CBO9780511760396>
The definitive textbook: 784 pages covering software engineering for
simulation, MMS, solution verification, model validation, design of
validation experiments, and uncertainty quantification.

---

## 8.2 Standards Documents

**AIAA G-077-1998**, *Guide for the Verification and Validation of
Computational Fluid Dynamics Simulations*, AIAA, 1998.
<https://arc.aiaa.org/doi/book/10.2514/4.472855>
The first consensus standards document. Defines terminology still in use
today and proposes procedures for each transition in the simulation pipeline.

**ASME VV-20-2009 (reaffirmed 2021)**, *Standard for Verification and
Validation in Computational Fluid Dynamics and Heat Transfer*, ASME.
<https://www.asme.org/codes-standards/find-codes-standards/v-v-20-standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer>
The first normative (not just advisory) standard. Introduces the view that
validation is not pass/fail but an assessment of model error at a specific
validation point. Required in aerospace and ASME-regulated industries.

**NASA-STD-7009B** (2024), *Standard for Models and Simulations*, NASA.
<https://standards.nasa.gov/standard/NASA/NASA-STD-7009>
The most operationally demanding regulatory standard. Mandates eight
credibility factors across V&V, operational quality, and supporting evidence
before simulation results may be used to support decisions.

---

## 8.3 Method of Manufactured Solutions (MMS)

MMS is now the de facto standard first step in code verification for any new
or refactored PDE solver. It verifies that a discretization map converges to
the correct continuous operator at the stated rate, using source terms
computed symbolically from a chosen "manufactured" solution.

**Kambiz Salari and Patrick Knupp**, *Code Verification by the Method of
Manufactured Solutions*, Sandia National Laboratories SAND2000-1444, 2000.
<https://www.osti.gov/biblio/759450>
The canonical reference document. Provides the complete mathematical
procedure, source-term derivation, and application to incompressible and
compressible Navier-Stokes.

**Patrick Knupp and Kambiz Salari**, *Verification of Computer Codes in
Computational Science and Engineering*, Chapman & Hall/CRC, 2003.
ISBN 978-1-58488-264-0.
The book-length treatment. Includes worked examples and discussion of how
MMS identifies coding errors that affect convergence order but not
isolated single-resolution tests.

**Christopher J. Roy**, "Verification of Euler/Navier-Stokes codes using the
method of manufactured solutions," *International Journal for Numerical
Methods in Fluids*, 44:599–620, 2004.
<https://doi.org/10.1002/fld.660>
Demonstrates MMS applied to full compressible flow solvers with detailed
worked examples.

**Limits of MMS.** MMS tests the discretization of interior operators but
does not test the physical model itself, stiff source terms, or problems with
discontinuities. Boundary condition MMS requires additional care. A code that
passes MMS on smooth manufactured data can still fail on physically
realizable initial conditions. Roy (2005) discusses this in detail.

**Christopher J. Roy**, "Review of Code and Solution Verification Procedures
for Computational Simulation," *Journal of Computational Physics*,
205(1):131–156, 2005.
<http://ftp.demec.ufpr.br/CFD/bibliografia/erros_numericos/Roy_2005.pdf>
Comprehensive survey of MMS, Richardson extrapolation, and order-of-accuracy
testing. Widely used as a self-contained technical reference.

---

## 8.4 Predictive Capability Maturity Model (PCMM)

The closest existing framework to Cosmic Foundry's capability-spec
discipline is the PCMM, developed at Sandia.

**William L. Oberkampf, Martin Pilch, and Timothy G. Trucano**, *Predictive
Capability Maturity Model for Computational Modeling and Simulation*, Sandia
National Laboratories SAND2007-5948, 2007.
<https://www.osti.gov/servlets/purl/976951/>
Defines a four-level maturity scale across six elements: geometric/
representation fidelity, physics model fidelity, code verification, solution
verification, model validation, and uncertainty quantification. Originally
developed for nuclear weapons stockpile assessment; now used more broadly.

**The PCMM used prospectively is the closest existing analog to a
capability specification.** The six elements define what evidence must exist
for a simulation capability to be credible at each maturity level. If the
PCMM rubric is filled in *before* the code is written — specifying what
evidence will be produced and at what level — it functions as a formal
capability specification. This use is not described in the literature but is
the natural extension.

---

## 8.5 Specification-First Approaches: What Exists

No paper in the literature is explicitly titled "write the capability spec
before the solver" or describes specification-first code development as a
named discipline. The closest approaches come from three directions:

**Requirements-before-V&V (Oberkampf, Trucano & Hirsch 2004).**
*Applied Mechanics Reviews*, 57(5):345–384, 2004.
<https://doi.org/10.1115/1.1767847>
Explicitly states that requirements for a computational-physics capability
must be defined first, that V&V activities must be designed against those
requirements, and that validation experiments must be matched to the
application domain of interest. This is philosophically specification-first,
framed as "V&V planning."

**ASC Implementation Plans (DOE/NNSA, annually).**
<https://www.osti.gov/biblio/2475271> (FY2025 example)
The NNSA Advanced Simulation and Computing program defines the physics
capabilities needed by the stockpile stewardship program, then maps those
to simulation development tasks. Requirements flow from application need to
code development, not the reverse. This is institutionalized
specification-first at the program level. These plans are the closest
existing operational analog to Cosmic Foundry's capability-first approach.

**Goal-oriented error estimation (Oden & Prudhomme 2001).**
*Computers and Mathematics with Applications*, 41(5–6):735–756, 2001.
<https://doi.org/10.1016/S0898122100003175>
Provides a mathematically rigorous path to verified error bounds on specific
output quantities using adjoint-based estimation. Specifying "compute drag
coefficient to 1% accuracy" before building the solver and then verifying
the code meets that spec is specification-first at the output-quantity level.
Rigorous but uncommon in practice outside finite-element structural
mechanics.

**The gap.** A 2015 survey of software use in astronomy found that 90% of
astronomers write their own code but only 8% received substantial software
training; no example of specification-first development was found in the
astrophysics community. NIST IR 8298 (2020) and a 2025 testing survey both
find that the dominant practice across scientific computing is ad hoc: code
is written first, then tested against available exact solutions, with no
prior specification of what the code is supposed to compute and how its
correctness will be demonstrated.

---

## 8.6 Code Projects with Systematic V&V

**FLASH / Flash-X (University of Chicago / DOE ASCI / Argonne).**
Calder et al. (2002), *ApJS* 143(1):201–229, is one of the first
astrophysics papers to apply the CFD V&V framework explicitly.
<https://doi.org/10.1086/342267>
The FLASH test suite covered known-answer problems, shock physics, and
comparison to laser-driven shock experiments. Dubey et al. (2014) documents
how the component-based architecture enabled ongoing verification across
20+ years of code evolution.
<https://doi.org/10.1177/1094342013505656>

**AMReX-Astro ecosystem (Castro, MAESTROeX, Nyx, Pele).**
<https://amrex-codes.github.io/regression_testing/>
The most systematic ongoing verification infrastructure currently active in
computational astrophysics. Nightly automated regression tests across the
full AMReX ecosystem; Castro's verification page explicitly documents
manufactured solution tests and known-answer problems. This is the current
state of the art in comp-astro V&V.

**Athena / Athena++.**
Stone et al. (2008), *ApJS* 178:137–177. <https://doi.org/10.1086/588755>
Documents a test suite with quantitative convergence measurements for 1D,
2D, and 3D hydro and MHD problems. The test suite was designed for
comparison with other codes and is built into the repository.

**Trilinos (Sandia National Laboratories).**
Heroux et al. (2005), *ACM TOMS* 31(3):397–423.
<https://doi.org/10.1145/1089014.1089021>
Each Trilinos package must meet ASC Software Quality Engineering standards:
unit tests, continuous integration, defined dependency contracts. The
TriBITS lifecycle model formalizes maturity levels from research prototype
to production library. Arguably the best example of software engineering
discipline applied to scientific computing infrastructure.

**NRC thermal-hydraulics codes (TRACE, RELAP).**
<https://www.nrc.gov/about-nrc/regulatory/research/safetycodes>
The most heavily regulated domain: codes cannot be used for nuclear safety
licensing without documented V&V. The NRC's CAMP international
collaboration provides structured validation across ~30 countries.

**SU2 (open-source aerospace CFD).**
<https://su2code.github.io/vandv/MMS_FVM_Navier_Stokes/>
Maintains a publicly documented MMS verification suite for compressible
Navier-Stokes, updated with each code release.

---

## 8.7 Reproducibility in Scientific Computing

**Ivie and Thain**, "Reproducibility in Scientific Computing," *ACM
Computing Surveys*, 51(3):63, 2018.
<https://doi.org/10.1145/3186266>
Surveys reproducibility barriers across scientific computing. Finds that most
computational experiments are described only informally in papers and that
the code producing the results is rarely available.

**Non-determinism in HPC.** Modern HPC systems produce results that differ
across runs even with identical source code and input, because parallel
floating-point arithmetic is non-associative. This is a hard reproducibility
floor below which bit-reproducibility cannot be guaranteed without explicit
design effort. See Hoefler and Belli (*SC15*, 2015).

**DOE Correctness for Scientific Computing (CS2) Initiative, 2025.**
<https://computing.llnl.gov/projects/formal-methods-correctness>
A 2025 DOE/NSF joint program funding research into formal methods and
correctness for scientific computing codes. Signals that the gap between
software engineering rigor and simulation practice is recognized at the
funding-agency level.

---

## 8.8 Implications for Cosmic Foundry

**What the field has that we should use:**

- **MMS** is the right verification technique for discretization maps.
  When we implement any PDE discretization, MMS is the primary external
  grounding method: choose a smooth manufactured solution, compute the
  source term symbolically (SymPy), verify second-order (or higher)
  convergence of the solver to that solution. This is the operationalization
  of "verify the discretization map against the continuous operator."

- **The PCMM** provides a useful checklist of what evidence must exist for
  a capability to be credible. Used prospectively — filled in before the
  code is written — it functions as a capability specification template. The
  six elements (representation fidelity, physics model fidelity, code
  verification, solution verification, model validation, UQ) map onto the
  three-track roadmap structure.

- **Goal-oriented error estimation** is the rigorous path to specifying
  "compute quantity X to accuracy ε" and then verifying the code meets that
  spec. Worth incorporating once physics capabilities are implemented.

**What the field lacks that we are building:**

The literature has no established name or practice for what we are doing:
writing a capability specification before implementation, with the external
grounding source identified in advance, as a first-class development
discipline. The closest things are the ASC Implementation Plans
(institutional, classified context) and the PCMM used prospectively.

This means we are not re-implementing something well-understood — we are
filling a gap. The three-track roadmap (capabilities, implementation,
verification) and the object/map framing in `capabilities/README.md` have
no direct precedent in the published literature, though they are
philosophically consistent with Oberkampf et al.'s requirements-before-V&V
principle and with the PCMM maturity framework.

**The absence from astrophysics is striking.** The comp-astro community has
not applied CFD V&V methodology systematically. FLASH and AMReX-Astro are
the best examples, but even these built the code first and the test suite
second. No astrophysics code project appears to have written formal
capability specifications before implementation.
