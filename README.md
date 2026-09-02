# LINK standalone

This repository is the standalone distribution of LINK (v0.1.0rc8).  It contains the
runtime components required for multi-backend geometry optimization:

- ORACLE for molecular perception and topology;
- SMITH for symmetry-oriented nonredundant internal coordinates;
- LINK for geometry optimization, derivative handling, transition-state
  workflows, and composite surfaces.

The distribution is independent of the nano-MATRIX monorepo.  The internal
Python package names retain their stable `matrix_*` namespace so that the
released implementation and its provenance remain unambiguous; no MATRIX
runtime installation is required.

Electronic-structure programs are optional backend providers.  Install the
base package for coordinate construction and optimization, then install the
optional backend dependencies required by the selected quantum-chemical
program.  The backend executable itself remains an external installation.

The package is released under the BSD 3-Clause license.  This bundle was
assembled from nano-MATRIX commit `54123b2702d84177c73a7c1a813695c678945a51`;
the standalone source tree is intended to be versioned and tagged separately.

## Dependencies

The base installation includes the internal runtime layers for ORACLE, SMITH,
SONIC, and LINK, together with NumPy, SciPy, NetworkX, Pillow, Autograd, and
Numba.  ASE and `qc-iodata` are optional dependencies for backend adapters.
The executable quantum-chemical programs remain external and are not bundled.

## Installation

```bash
python -m pip install .
```

The release contains no campaign scratch data or manuscript build products.
The source tree is intended to be versioned independently and tested with the
same coordinate-rank and backend-contract checks used for the manuscript.

```bash
python -m pytest
```
