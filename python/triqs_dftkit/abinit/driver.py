import triqs.utility.mpi as mpi
import numpy as np
from h5 import HDFArchive
from ..wannier90.converter import Converter

class Driver(object):

    def __init__(self, seedname): self.seedname = seedname

    def run_initial_stage(self, **kwargs):
        Converter(self.seedname, **kwargs).convert_dft_input()
        return 

    def run_update_stage(self, **kwargs):
        pass

    def band_energy_and_write_charge_update(self, N_k):
        n_k = N_k.shape[1]
        fermi_weights, n_bands_per_k, band_window, Hk, bz_weights = None, None, None, None, None
        spin_to_data_index = [0,0]

        if mpi.is_master_node():
            with HDFArchive(f"{self.seedname}.h5", "r") as ar:
                fermi_weights = ar['dft_misc_input']['dft_fermi_weights']
                band_window = ar['dft_misc_input']['band_window']
                n_bands_per_k = ar['dft_input']['n_orbitals']
                Hk            = ar['dft_input']['hopping']
                bz_weights    = ar['dft_input']['bz_weights']
                spin_to_data_index = [0,1] if ar['dft_input']['SP'] else [0,0]

        fermi_weights = mpi.bcast(fermi_weights)
        band_window   = mpi.bcast(band_window)
        n_bands_per_k = mpi.bcast(n_bands_per_k)
        Hk            = mpi.bcast(Hk)
        bz_weights    = mpi.bcast(bz_weights)
        spin_to_data_index = mpi.bcast(spin_to_data_index)
        mpi.barrier()

        density_matrix_dft = [ [ fermi_weights[ik,idx,:].astype(complex) for ik in range(n_k) ] for idx in spin_to_data_index]
        band_energy_correction = 0.0
        # subtract off DFT density matrix and compute band energy correction
        for ik in range(n_k):
            for spin, idx in enumerate(spin_to_data_index):
                nb = n_bands_per_k[ik, idx]
                diag_indices = np.diag_indices(nb)
                N_k[spin, ik][diag_indices] -= density_matrix_dft[spin][ik][:nb]
                band_energy_correction += np.dot(N_k[spin][ik], Hk[ik,idx,:nb,:nb]).trace().real * bz_weights[ik]

        if mpi.is_master_node():
            kpts_to_write = np.arange(n_k)
            with open("abiout.delta_N", "w") as f:
                f.write(f" {n_k} - 1 ! Number of k-points, default number of bands\n")
                for index, ik in enumerate(kpts_to_write):
                    ib1, ib2 = band_window[0][ik,0], band_window[0][ik,1]
                    f.write(f" {index+1} {ib1} {ib2}\n")
                    for inu in range(n_bands_per_k[ik, 0]):
                        for imu in range(n_bands_per_k[ik, 0]):
                            valre = (N_k[0][ik][inu, imu].real + N_k[1][ik][inu, imu].real) /2.0
                            valim = (N_k[0][ik][inu, imu].imag + N_k[1][ik][inu, imu].imag) /2.0
                            f.write(f" {valre:.14f} {valim:.14f}\n")
                        f.write("\n")

        return band_energy_correction
