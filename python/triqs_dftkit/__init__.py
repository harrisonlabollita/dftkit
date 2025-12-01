################################################################################
#
# TRIQS: a Toolbox for Research in Interacting Quantum Systems
#
# Copyright (C) 2016-2018, N. Wentzell
# Copyright (C) 2018-2019, The Simons Foundation
#   author: N. Wentzell
#
# TRIQS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# TRIQS. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

r"""
triqs_dftkit: DFT Converters for TRIQS

This package provides converters from various DFT codes to the HDF5 format
used by TRIQS for DFT+DMFT calculations.

Available submodules:
- elk: Elk converter
- hk: General H(k) converter
- qe: Quantum Espresso converter
- vasp: VASP converter
- wannier90: Wannier90 converter
- wien2k: Wien2k converter

Each submodule provides a Converter class that can be imported as:
    from triqs_dftkit.vasp import Converter
"""

__all__ = []
