"""
Quantum ESPRESSO driver for TRIQS+DFT workflow automation.

This module provides a driver class for automating Quantum ESPRESSO (QE) and
Wannier90 workflows in the context of DMFT calculations with TRIQS/modest
"""
import os, shlex, subprocess
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from h5 import HDFArchive
import triqs.utility.mpi as mpi

from ..wannier90.converter import Converter


class DFTWorkflowError(Exception):
    """Exception raised for errors in the DFT workflow."""
    pass


# Default environment variables to preserve for subprocess execution
_DEFAULT_ENV_VARS = [
    'PATH',
    'LD_LIBRARY_PATH',
    'SHELL',
    'PWD',
    'HOME',
    'OMP_NUM_THREADS',
    'OMPI_MCA_btl_vader_single_copy_mechanism'
]

@dataclass
class MPIHandler:
    """
    Handles MPI configuration and execution environment for parallel DFT calculations.
    
    This class encapsulates MPI-related configuration and provides utilities for
    running parallel executables with proper environment variable handling.
    
    Attributes
    ----------
    mpi_exec : str
        The MPI executor command (e.g., "mpirun", "mpiexec", "srun").
    n_cores : int
        Number of MPI processes/cores to use for parallel execution.
    
    Examples
    --------
    >>> mpi_handler = MPIHandler(mpi_exec="mpirun -np 4", n_cores=4)
    >>> env = mpi_handler.get_env_vars()
    """

    mpi_exec : str = "mpirun"
    n_cores  : int = 1

    def __post_init__(self):
        """Validate MPI configuration."""
        if self.n_cores <= 0:
            raise ValueError(f"n_cores must be positive, got {self.n_cores}")
        if not self.mpi_exec.strip():
            raise ValueError("mpi_exec cannot be empty")

    def __repr__(self): 
        return f"MPIHandler(mpi_exec={self.mpi_exec}, n_cores={self.n_cores})"

    __str__ = __repr__

    def get_env_vars(self):
        """
        Retrieve essential environment variables for subprocess execution.
        
        Collects a subset of environment variables that are typically needed
        for proper execution of MPI-enabled DFT codes.
        
        Returns
        -------
        dict[str, str]
            Dictionary of environment variable names and their values.
            Only includes variables that are set in the current environment.
            
        Note
        ----
        The following variables are collected if present:
        - PATH: System executable paths
        - LD_LIBRARY_PATH: Shared library paths
        - SHELL: User shell
        - PWD: Present working directory
        - HOME: User home directory
        - OMP_NUM_THREADS: OpenMP thread count
        - OMPI_MCA_btl_vader_single_copy_mechanism: OpenMPI optimization setting
        """
        env_vars = {}
        for var_name in _DEFAULT_ENV_VARS:
            var = os.getenv(var_name)
            if var: env_vars[var_name] = var
        return env_vars

    def report(self, *args):
        """
        Report a message (only from master MPI node).
        
        Parameters
        ----------
        *args : Any
            Arguments to pass to mpi.report for printing.
        """
        return mpi.report(*args)

    def is_master_node(self) -> bool:
        """
        Check if this is the master MPI node.
        
        Returns
        -------
        bool
            True if this is the master node, False otherwise.
        """
        return mpi.is_master_node()

class Driver:
    """
    Driver for orchestrating Quantum ESPRESSO and Wannier90 calculations.
    
    This class automates the execution of a typical DFT workflow consisting of:
    1. Self-consistent field (SCF) calculations
    2. Non-self-consistent field (NSCF) calculations
    3. Wannier90 preprocessing and execution
    4. pw2wannier90 interface for projections
    
    The driver is designed to work within DMFT self-consistency loops where
    DFT calculations may need to be updated with charge density modifications.
    
    Attributes
    ----------
    seedname : str
        Base name for all input/output files (e.g., "system" for "system.scf.in").
    mpi_handler : MPIHandler
        MPI configuration handler for parallel execution.
    """

    def __init__(self, seedname : str, mpi_handler : MPIHandler) -> None:
        """
        Initialize the Quantum ESPRESSO driver.
        
        Parameters
        ----------
        seedname : str
            Base name for all input/output files.
        mpi_handler : MPIHandler
            MPI configuration for parallel execution.
        """
        self.seedname = seedname
        self.mpi_handler = mpi_handler

    def __repr__(self): return f"QuantumEspressoDriver(seedname= {self.seedname}, mpi_handler= {repr(self.mpi_handler)})"

    __str__ = __repr__

    def _run_qe_step(self, executable : str, step_name : str, k_parallel : bool = False) -> int:
        """
        Execute a Quantum ESPRESSO calculation step.
        
        This is an internal method that handles the execution of QE executables
        (pw.x, pw2wannier90.x, etc.) with proper I/O redirection and error handling.
        
        Parameters
        ----------
        executable : str
            Name of the QE executable to run (e.g., "pw.x", "pw2wannier90.x").
        step_name : str
            Calculation step identifier used for file naming (e.g., "scf", "nscf").
        k_parallel : bool, optional
            If True, enables k-point parallelization with -nk flag. Default is False.
            
        Returns
        -------
        int
            Return code from the subprocess execution (0 for success).
            
        Note
        ----
        Expected file naming convention:
        - Input: {seedname}.{step_name}.in
        - Output: {seedname}.{step_name}.out
        - Error: {seedname}.{step_name}.err
        
        The calculation is only executed on the master MPI node to avoid
        duplicate execution in MPI environments.
        
        Raises
        ------
        FileNotFoundError
            If the required input file does not exist.
        DFTWorkflowError
            If the executable returns a non-zero exit code.
        """

        if not self.mpi_handler.is_master_node(): return 0

        # validate input file existence
        infile  = f"{self.seedname}.{step_name}.in"
        if not os.path.exists(infile): raise FileNotFoundError(f"Required input file not found: {infile}")

        outfile = f"{self.seedname}.{step_name}.out"
        errfile = f"{self.seedname}.{step_name}.err"

        with open(infile, "r") as inp, open(outfile, "w") as out, open(errfile, "w") as err:

            command = [*shlex.split(self.mpi_handler.mpi_exec), executable]

            #if k_parallel: command.extend(["-nk", str(self.mpi_handler.n_cores)])

            self.mpi_handler.report(f"[{datetime.now()}] running {' '.join(command)} < {infile} > {outfile}")

            result = subprocess.run(
                    command,
                    stdin=inp,
                    stdout=out,
                    stderr=err,
                    env=self.mpi_handler.get_env_vars(),
                    text=True,
                    )

            if result.returncode != 0:
                error_msg = f"{executable} ({step_name}) failed with code {result.returncode}"
                self.mpi_handler.report(error_msg)
                raise DFTWorkflowError(error_msg)

            return result.returncode

    def _run_w90_step(self, executable : str) -> int:
        """
        Execute a Wannier90 calculation step.
        
        This is an internal method that handles the execution of Wannier90
        executables with proper command-line argument formatting.
        
        Parameters
        ----------
        executable : str
            Wannier90 executable command with flags (e.g., "wannier90.x -pp").
            
        Returns
        -------
        int
            Return code from the subprocess execution (0 for success).
            
        Note
        ----
        The seedname is automatically appended as the final command-line argument.
        The calculation is only executed on the master MPI node.
        
        Raises
        ------
        DFTWorkflowError
            If the executable returns a non-zero exit code.
        """
        if not self.mpi_handler.is_master_node(): return 0

        command = [*shlex.split(executable), self.seedname]

        self.mpi_handler.report(f"[{datetime.now()}] running {' '.join(command)}")

        try: subprocess.check_call(command, shell=False, env=self.mpi_handler.get_env_vars())
        except subprocess.CalledProcessError as e:
            error_msg = f"Wannier90 step failed: {executable}"
            self.mpi_handler.report(error_msg)
            raise DFTWorkflowError(error_msg) from e
        return 0

    def run_scf(self) -> int:
        """
        Run a self-consistent field (SCF) calculation.
        
        Returns
        -------
        int
            Return code (0 for success).
        """
        return self._run_qe_step("pw.x", "scf", k_parallel=True)

    def run_mod_scf(self) -> int:
        """
        Run a modified SCF calculation with updated charge density.
        
        This is typically used in DMFT self-consistency loops where the
        charge density has been modified based on the lattice self-energy.
        
        Returns
        -------
        int
            Return code (0 for success).
        """
        return self._run_qe_step("pw.x", "mod_scf", k_parallel=True)

    def run_nscf(self) -> int:
        """
        Run a non-self-consistent field (NSCF) calculation.
        
        This step computes band structures on a denser k-point grid using
        the converged charge density from SCF.
        
        Returns
        -------
        int
            Return code (0 for success).
        """
        return self._run_qe_step("pw.x", "nscf", k_parallel=True)

    def run_pw2wannier90(self) -> int:
        """
        Run the pw2wannier90 interface.
        
        This step computes overlaps and projections needed for Wannier90
        from the QE wavefunction data.
        
        Returns
        -------
        int
            Return code (0 for success).
        """
        return self._run_qe_step("pw2wannier90.x", "pw2wannier90")

    def run_wannier90_pp(self) -> int:
        """
        Run Wannier90 preprocessing step.
        
        This generates the required k-point list and other auxiliary files
        for the subsequent QE and Wannier90 calculations.
        
        Returns
        -------
        int
            Return code (0 for success).
        """
        return self._run_w90_step("wannier90.x -pp") 

    def run_wannier90(self) -> int:
        """
        Run the Wannier90 wannierization.
        
        This step performs the actual Wannier function construction and
        generates the Wannier Hamiltonian.
        
        Returns
        -------
        int
            Return code (0 for success).
        """
        return self._run_w90_step("wannier90.x")

    def band_energy_and_write_charge_update(self, N_k):

        n_spin, n_k, n_max_bands, _ = N_k.shape

        fermi_weights, n_bands_per_k, Hk, bz_weights = None, None, None, None
        spin_to_data_index = [0,0]
        if mpi.is_master_node():
            with HDFArchive(f"{self.seedname}.h5", "r") as ar:
                fermi_weights = ar['dft_misc_input']['dft_fermi_weights']

                n_bands_per_k = ar['dft_input']['n_orbitals']
                Hk            = ar['dft_input']['hopping']
                bz_weights    = ar['dft_input']['bz_weights']
                spin_to_data_index = [0,0] if ar['dft_input']['SO'] == 0 else [0,1]

        fermi_weights = mpi.bcast(fermi_weights)
        n_bands_per_k = mpi.bcast(n_bands_per_k)
        Hk            = mpi.bcast(Hk)
        bz_weights    = mpi.bcast(bz_weights)
        spin_to_index = mpi.bcast(spin_to_data_index)
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
        mpi.report(f"\nBand Energy Correction= {band_energy_correction} eV\n")
        mpi.report(f"{n_k} -1 ! number of k-points, default number of bands\n")
        delta_Nk = np.zeros( (n_k, n_max_bands, n_max_bands), dtype=complex)
        for ik in range(n_k):
            for inu in range(n_bands_per_k[ik,0]):
                for imu in range(n_bands_per_k[ik,0]):
                    valre = (N_k[0][ik][inu, imu].real + N_k[1][ik][inu, imu].real) /2.0
                    valim = (N_k[0][ik][inu, imu].imag + N_k[1][ik][inu, imu].imag) /2.0
                    delta_Nk[ik, inu, imu] = valre + 1j*valim

        if mpi.is_master_node():
            subgrp = 'dft_update'
            with HDFArchive(f"{self.seedname}.h5", "a") as ar:
                if subgrp not in ar: ar.create_group(subgrp)
                ar[subgrp]["delta_N"] = delta_Nk

        return band_energy_correction

    def read_dft_energy(self):
        energy_unit = 13.605693123 # eV
        dft_energy = 0.0 
        if mpi.is_master_node():

            read_scf = False if os.path.isfile(f"{self.seedname}.mod_scf.out") else True

            if read_scf:
                with open(f"{self.seedname}.scf.out", "r") as f: lines = f.readlines()
                for line in lines:
                    if '!' in line:
                        dft_energy = float(line.split()[-2]) * energy_unit
                        break
            else:
                with open(f"{self.seedname}.mod_scf.out", "r") as f: lines = f.readlines()
                for line in lines:
                    if "(sum(wg*et))" in line:
                        print("\nReading band energy from the mod_scf calculation\n")
                        band_energy_modscf = float(line.split()[-2]) * energy_unit
                        print(f"The mod_scf band energy is: {band_energy_modscf} eV")
                    if "total energy" in line:
                        print("\nReading total energy from the mod_scf calculation\n")
                        dft_energy = float(line.split()[-2]) * energy_unit
                        print(f"The uncorrected DFT energy is: {dft_energy} eV")
                dft_energy -= band_energy_modscf
                print(f"The DFT energy without kinetic part is: {dft_energy} eV")

                with open(f"{self.seedname}.nscf.out", "r") as f: lines = f.readlines()
                for line in lines:
                    if "The nscf band energy" in line:
                        print("\nReading band energy from the nscf calculation\n")
                        band_energy_nscf = float(line.split()[-2]) * energy_unit
                        dft_energy += band_energy_nscf
                        print(f"The nscf band energy is: {band_energy_nscf} eV")
                        print(f"The corrected DFT energy is: {dft_energy} eV")
                        break

        dft_energy = mpi.bcast(dft_energy);
        return dft_energy

    def run_initial_stage(self, **kwargs) -> int:
        steps = [  ("SCF calculation", self.run_scf),
                   ("NSCF calculation", self.run_nscf),
                   ("Wannier90 preprocessing", self.run_wannier90_pp),
                   ("pw2wannier90 interface", self.run_pw2wannier90),
                   ("Wannier90 wannierization", self.run_wannier90),
                   ("TRIQS QE Converter", Converter(seedname=self.seedname, bloch_basis=True, **kwargs).convert_dft_input),
        ]

        status = 0
        for i, (step_name, step_func) in enumerate(steps, 1): status = step_func()
        return status

    def run_update_stage(self, N_k, Eint_m_dc, **kwargs):

        dft_energy = self.read_dft_energy()

        band_energy_correction = self.band_energy_and_write_charge_update(N_k)

        mpi.report(f"! DFT + DMFT Total Energy: {dft_energy + band_energy_correction + Eint_m_dc}")

        steps = [  ("Mod SCF calculation", self.run_mod_scf),
                   ("NSCF calculation", self.run_nscf),
                   ("Wannier90 preprocessing", self.run_wannier90_pp),
                   ("pw2wannier90 interface", self.run_pw2wannier90),
                   ("Wannier90 wannierization", self.run_wannier90),
                   ("TRIQS QE Converter", Converter(seedname=self.seedname, bloch_basis=True, **kwargs).convert_dft_input),
        ]
        status = 0
        for i, (step_name, step_func) in enumerate(steps, 1): status = step_func()
        return status
