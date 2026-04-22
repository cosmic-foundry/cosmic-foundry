# Cosmic Foundry — Roadmap

Items here are long-horizon: the motivation is clear but the design is not yet
specifiable. When an item becomes fully specified it moves to STATUS.md.

---

## Physical constants ingestion (CODATA)

The engine will need physical constants (G, c, ħ, k_B, …) throughout the
physics epochs. The authoritative machine-readable source is NIST CODATA
(public domain), available as an ASCII table at
`https://physics.nist.gov/cuu/Constants/Table/allascii.txt`.

Design questions to settle before implementation:
- Where does the constants module live (`foundation/`? `computation/`?); it
  must respect the symbolic-reasoning import boundary
- How are constants exposed — as SymPy symbols with known numerical values,
  as plain Python floats, or both?
- How is the CODATA revision pinned and updated (hash, version tag)?
- WGS 84 / GPS-specific defined constants (μ, Ω_E, GPS semi-major axis) have
  no machine-readable API; the ingestion discipline for PDF-sourced defined
  constants needs a separate decision
