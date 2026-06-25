r"""
Tests for triqs_dftkit.vasp.Driver.band_energy_and_write_charge_update.

The writer stores the DMFT charge-density correction in the layout VASP's
ADD_GAMMA_FROM_FILE reads:
  - band_window/0          : (n_k_ibz, 2) band ranges
  - deltaN/{up,down}/<ik>  : collinear / spin-averaged (SP=0, SO=0)
  - deltaN/ud/<ik>         : spin-orbit (SO=1), single spinor channel

Strategy: build the expected correction independently in numpy, write it into a
reference vaspgamma.h5 in the same layout, run the writer, and compare the two
archives with triqs.utility.h5diff (which recurses groups/lists and compares
arrays via assert_arrays_are_close). h5diff therefore checks in one shot:
the group keys (up/down vs ud), the IBZ list length, band_window and every
deltaN block. The scalar band-energy correction is checked separately.
"""
import os
import unittest
import tempfile
import shutil
import numpy as np
from h5 import HDFArchive
from triqs.utility import h5diff
from triqs_dftkit.vasp.driver import Driver, MPIHandler


def _hermitian(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return a + a.conj().T


def _assert_h5_equal(f1, f2, precision=1.e-10):
    h5diff.failures = []          # module-global accumulator is never reset by h5diff itself
    h5diff.h5diff(f1, f2, precision)


def _write_seedname_h5(path, SO, SP, n_orbitals, fermi, hopping, band_window, bz_weights, n_k_ibz):
    with HDFArchive(path, 'w') as ar:
        ar.create_group('dft_input')
        ar['dft_input']['SO'] = SO
        ar['dft_input']['SP'] = SP
        ar['dft_input']['n_orbitals'] = n_orbitals          # (n_k, n_spin_blocks)
        ar['dft_input']['hopping'] = hopping                # (n_k, n_spin_blocks, nb, nb)
        ar['dft_input']['bz_weights'] = bz_weights          # (n_k,)
        ar.create_group('dft_misc_input')
        ar['dft_misc_input']['dft_fermi_weights'] = fermi   # (n_k, n_spin_blocks, nb)
        ar['dft_misc_input']['band_window'] = [band_window]  # list with one (n_k, 2) array
        ar['dft_misc_input']['n_k_ibz'] = n_k_ibz


def _write_expected_vaspgamma(path, band_window_ibz, deltaN_blocks, channels):
    with HDFArchive(path, 'w') as e:
        e['band_window'] = [band_window_ibz]
        e.create_group('deltaN')
        for ch in channels:
            e['deltaN'][ch] = deltaN_blocks


class TestGammaWriter(unittest.TestCase):

    def setUp(self):
        self._cwd = os.getcwd()
        self._tmp = tempfile.mkdtemp()
        os.chdir(self._tmp)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _driver(self):
        return Driver(seedname='seed', plo_cfg='unused', mpi_handler=MPIHandler())

    # shared small system: 2 k-points, max 3 bands, varying bands per k
    def _system(self):
        n_k, nb_max = 2, 3
        nb_per_k = [3, 2]
        bz = np.array([0.4, 0.6])
        band_window = np.array([[1, 3], [1, 2]], dtype=int)
        fermi = np.zeros((n_k, 1, nb_max))
        fermi[0, 0, :3] = [1.0, 0.3, 0.0]
        fermi[1, 0, :2] = [1.0, 0.2]
        hopping = np.zeros((n_k, 1, nb_max, nb_max), dtype=complex)
        hopping[0, 0, :3, :3] = _hermitian(3, 1)
        hopping[1, 0, :2, :2] = _hermitian(2, 2)
        return n_k, nb_max, nb_per_k, bz, band_window, fermi, hopping

    def test_collinear_writes_up_down_and_is_correct(self):
        n_k, nb_max, nb_per_k, bz, band_window, fermi, hopping = self._system()

        N_k = np.zeros((n_k, 2, nb_max, nb_max), dtype=complex)
        M = {}
        for ik, nb in enumerate(nb_per_k):
            for s in range(2):
                M[(ik, s)] = _hermitian(nb, 10 * ik + s + 3)
                N_k[ik, s, :nb, :nb] = M[(ik, s)]

        # independent reference: subtract DFT occupations on the diagonal, average spins
        bec_ref, delta_ref = 0.0, []
        for ik, nb in enumerate(nb_per_k):
            ncorr = []
            for s in range(2):
                c = M[(ik, s)].copy()
                c[np.diag_indices(nb)] -= fermi[ik, 0, :nb]
                ncorr.append(c)
                bec_ref += bz[ik] * np.trace(c @ hopping[ik, 0, :nb, :nb]).real
            delta_ref.append(0.5 * (ncorr[0] + ncorr[1]))

        _write_seedname_h5('seed.h5', 0, 0, np.array([[3], [2]]),
                           fermi, hopping, band_window, bz, n_k)
        _write_expected_vaspgamma('expected.h5', band_window, delta_ref, channels=['up', 'down'])

        bec = self._driver().band_energy_and_write_charge_update(N_k.copy())

        self.assertAlmostEqual(bec, bec_ref, places=10)
        _assert_h5_equal('vaspgamma.h5', 'expected.h5')

    def test_soc_writes_ud_and_is_correct(self):
        n_k, nb_max, nb_per_k, bz, band_window, fermi, hopping = self._system()

        N_k = np.zeros((n_k, 1, nb_max, nb_max), dtype=complex)   # single spinor channel
        M = {}
        for ik, nb in enumerate(nb_per_k):
            M[ik] = _hermitian(nb, 20 + ik)
            N_k[ik, 0, :nb, :nb] = M[ik]

        bec_ref, delta_ref = 0.0, []
        for ik, nb in enumerate(nb_per_k):
            c = M[ik].copy()
            c[np.diag_indices(nb)] -= fermi[ik, 0, :nb]
            delta_ref.append(c)                                   # no spin average for SOC
            bec_ref += bz[ik] * np.trace(c @ hopping[ik, 0, :nb, :nb]).real

        _write_seedname_h5('seed.h5', 1, 0, np.array([[3], [2]]),
                           fermi, hopping, band_window, bz, n_k)
        _write_expected_vaspgamma('expected.h5', band_window, delta_ref, channels=['ud'])

        bec = self._driver().band_energy_and_write_charge_update(N_k.copy())

        self.assertAlmostEqual(bec, bec_ref, places=10)
        _assert_h5_equal('vaspgamma.h5', 'expected.h5')

    def test_ibz_slicing(self):
        # n_k_ibz < n_k: only the first n_k_ibz blocks are written
        n_k, nb_max, nb_per_k, bz, band_window, fermi, hopping = self._system()
        N_k = np.zeros((n_k, 1, nb_max, nb_max), dtype=complex)
        # with N_k = 0 the correction is just minus the DFT occupations on the diagonal
        nb0 = nb_per_k[0]
        c0 = np.zeros((nb0, nb0), dtype=complex)
        c0[np.diag_indices(nb0)] -= fermi[0, 0, :nb0]
        delta_ref = [c0]   # only ik=0
        _write_seedname_h5('seed.h5', 1, 0, np.array([[3], [2]]),
                           fermi, hopping, band_window, bz, n_k_ibz=1)
        _write_expected_vaspgamma('expected.h5', band_window[:1], delta_ref, channels=['ud'])
        self._driver().band_energy_and_write_charge_update(N_k.copy())
        _assert_h5_equal('vaspgamma.h5', 'expected.h5')


if __name__ == '__main__':
    unittest.main()
