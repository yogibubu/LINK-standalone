# PPCS2 all-ORCA feasibility examples

These inputs exercise the LINK linear multi-backend assembler with

```text
DLPNO-CCSD(T)-F12/cc-pVDZ-F12
+ ae-MP2/cc-pwCVTZ
- fc-MP2/cc-pwCVTZ
```

All three terms are evaluated by ORCA at the same Cartesian geometry. LINK
forms numerical derivatives of the assembled energy along the active SONIC
coordinates and uses SONIC for the geometry updates. `TightPNO` is the default
local-correlation threshold.

Each feasibility case is first optimized with ORCA at
`B3LYP-D4/def2-TZVP`. An exact ORCA Hessian at the same level is then
transformed to the PPCS2 SONIC chart and used to initialize the composite
optimization. PPCS2 displaced points are launched as independent serial ORCA
jobs and parallelized across SONIC points, avoiding MPI session-directory
collisions on workstation installations.

Run one example from an activated nano-MATRIX source tree:

```bash
matrix link optimize-input examples/ppcs2/acetylene.inp \
  --run-dir scratch/ppcs2-acetylene
```

The examples are deliberately limited to acetylene and formaldehyde. They are
proofs of feasibility, not a benchmark campaign.

The bounded two-case runner is:

```bash
python examples/ppcs2/run_feasibility.py \
  --manifest examples/ppcs2/feasibility_cases.json \
  --run-dir scratch/ppcs2-feasibility
```
