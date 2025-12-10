import triqs.utility.mpi as mpi
from h5 import HDFArchive
from .converter import Converter

class Driver(object):

    def __init__(self, seedname): self.seedname = seedname

    def run_initial_stage(self):
        Converter(self.seedname).convert_dft_input()
        return 

    def run_update_stage(self, N_k, Ecorr, **kwargs):
        beta = kwargs.pop('beta'); mu   = kwargs.pop('mu')

        write_charge_correction(N_k, Ecorr, beta, mu)

        return 

    def write_charge_correction(N_k, Ecorr, beta, mu):
        energy_unit = 13.605698 # eV to Ry
        if not mpi.is_master_node(): return
        n_k = Nk.shape[0]
        with HDFArchive(f"{self.seedname}.h5", 'r') as ar:
            n_bands_per_k = ar['dft_input']['n_orbitals']
            SO            = ar['dft_input']['SO']
            SP            = ar['dft_input']['SP']
        def write_spin_block(file, Nksp, isp):
            file.write("%.14f\n" % (mu / energy_unit) )
            file.write("%.14f\n" % (beta * energy_unit) )
            for ik in range(n_k):
                file.write("%s\n" % n_bands_per_k[ik,isp])
                mat = Nksp[ik]
                for inu in range(n_bands_per_k[ik,isp]):
                    for imu in range(n_bands_per_k[ik,isp]):
                        file.write(f"{mat[inu,imu].real:.14f} {mat[inu,imu].imag:.14f} ")
                    file.write("\n")
                file.write("\n")
            file.write("%.16f\n" % Ecorr.real)
        if SP == 0:
            Nk_avg = 0.5 * (N_k[:, 0] + N_k[:, 1])
            with open(f"{self.seedname}.qdmft", "w") as f: write_spin_block(f, Nk_avg, 0)
        else:
            spin_to_data_idx = [0,1] if SO == 0 else [0,0]
            with open(f"{self.seedname}.qdmftup", "w") as fup, open(f"{self.seedname}.qdmftdn", "w") as fdn:
                for f, isp in zip([fup, fdn], spin_to_data_idx): write_spin_block(f, N_k[:, isp], isp)
        return
