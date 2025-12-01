
################################################################################
#
# TRIQS: a Toolbox for Research in Interacting Quantum Systems
#
# Copyright (C) 2011 by M. Aichhorn, L. Pourovskii, V. Vildosola
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


"""
Collinear SrVO3 tests: two correlated sites (V t2g and V eg) and uncorrelated
orbitals (O p), two impurities
"""

import os
from triqs_dftkit.wannier90 import Converter
from triqs.utility import h5diff
from triqs.utility import mpi

subfolder = 'w90_convert/'

# Test 1: Wannier basis (bloch_basis=False)
seedname = subfolder+'SrVO3_col'
converter = Converter(seedname=seedname, hdf_filename=seedname+'_wannierbasis.out.h5',
                               rot_mat_type='hloc_diag', bloch_basis=False)
converter.convert_dft_input()

if mpi.is_master_node():
    h5diff.h5diff(seedname+'_wannierbasis.out.h5', seedname+'_wannierbasis.ref.h5')

# Test 2: Bloch basis (bloch_basis=True)
# Need to temporarily move OUTCAR and LOCPROJ files for converter to find them
if mpi.is_master_node():
    os.rename(seedname + '.OUTCAR', subfolder + 'OUTCAR')
    os.rename(seedname + '.LOCPROJ', subfolder + 'LOCPROJ')
mpi.barrier()

try:
    converter = Converter(seedname=seedname, hdf_filename=seedname+'_blochbasis.out.h5',
                                   rot_mat_type='hloc_diag', bloch_basis=True)
    converter.convert_dft_input()
finally:
    if mpi.is_master_node():
        os.rename(subfolder + 'OUTCAR', seedname + '.OUTCAR')
        os.rename(subfolder + 'LOCPROJ', seedname + '.LOCPROJ')

if mpi.is_master_node():
    h5diff.h5diff(seedname+'_blochbasis.out.h5', seedname+'_blochbasis.ref.h5')
