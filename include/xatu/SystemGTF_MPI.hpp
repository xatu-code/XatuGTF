#pragma once
#include <mpi.h>
#include "xatu/ConfigurationCRYSTAL_MPI.hpp"
#include "xatu/System.hpp"

#ifndef constants
#define ANG2AU 1.889726126  
#endif

namespace xatu { 

/**
 * The SystemGTF class gathers all R-matrices and implements their transformation to k-space, referred to as FT.
 * It also handles the calculation of single-particle energies and wave-function coefficients.  
 * Exclusive to the GAUSSIAN mode
 */
class SystemGTF_MPI : public System {

    protected:
        // List of direct lattice vectors {Ri} in atomic units associated to the stored Hamiltonian matrices, stored by columns: (3,ncells).
        // The Hamiltonian (and Overlap, if reading from CRYSTAL) matrices are stored in the same order as these vectors
        arma::mat RlistAU_;
        // List of k-points in fractional coordinates employed in the BSE Hamiltonian that define the exciton basis, stored by (3D) columns
        arma::mat kpointsBSE_;
        // Number of k-points in kpointsBSE
        uint32_t nkBSE_;
        // Number of different |k-k'| in kpointsBSE (each will define a chunk, and all the set of chunks will be distributed among MPI processes)
        uint32_t nAbsk_;
        // List of k-points in fractional coordinates employed in the internal summations to compute the k-dependent matrix Pi (in the AUX basis), stored by (3D) columns
        arma::mat kpointsPol_;
        // Number of k-points in kpoints_Pi
        uint32_t nkPol_;
        // R-matrix elements of 2-center Coulomb integrals in the AUX basis, above a predefined tolerance
        std::vector<double> Coulomb2CValues_;
        // (mu, mu', R) indices corresponding to each element in Coulomb2CValues_
        std::vector<std::array<uint32_t,3>> Coulomb2CIndices_;
        // Number of 2-center Coulomb matrix elements that are stored in Coulomb2CValues
        uint64_t nvaluesC2c_;
        // List of R-vectors in fractional coordinates for which the 2-center Coulomb matrix elements were computed, in the same ordering and stored by columns
        arma::mat RlistFrac_C2c_;
        // Number of unit cells for which the 2-center Coulomb matrix elements are stored 
        uint ncellsC2c_;
        // R-matrix elements of 2-center metric integrals in the AUX basis, above a predefined tolerance
        std::vector<double> metric2CValues_;
        // (mu, mu', R) indices corresponding to each element in metric2CValues_
        std::vector<std::array<uint32_t,3>> metric2CIndices_;
        // Number of 2-center metric matrix elements that are stored in metric2CValues
        uint64_t nvaluesM2c_;
        // List of R-vectors in fractional coordinates for which the 2-center metric matrix elements have values above tolerance, in the original order and stored by columns
        arma::mat RlistFrac_M2c_;
        // Number of vectors in RlistFrac_M2c
        uint ncellsM2c_;
        // R,R'-matrix elements of 3-center metric integrals in the mixed SCF and AUX basis, above a predefined tolerance
        std::vector<double> metric3CValues_;
        // (P, mu, mu', R, R') indices corresponding to each element in metric3CValues_
        std::vector<std::array<uint32_t,5>> metric3CIndices_;
        // Number of 3-center metric matrix elements that are stored in metric3CValues
        uint64_t nvaluesM3c_;
        // List of R-vectors in fractional coordinates for which the 3-center metric matrix elements have values above tolerance, in the original order and stored by columns
        arma::mat RlistFrac_M3c_;
        // Number of vectors in RlistFrac_M3c
        uint ncellsM3c_;
        // Matrix dimension (or number of orbitals) in the AUX basis
        uint32_t norbitals_AUX_; 
        // Boolean which indicates whether the single-particle Hamiltonian includes SOC
        bool SOC_FLAG_;
        // Boolean which indicates whether the single-particle Hamiltonian includes magnetic terms. Irrelevant under SOC
        bool MAGNETIC_FLAG_;
    
    public:
        const arma::mat& RlistAU = RlistAU_;
        const arma::mat& kpointsBSE = kpointsBSE_;
        const uint32_t& nkBSE       = nkBSE_;
        const uint32_t& nAbsk       = nAbsk_;
        const arma::mat& kpointsPol = kpointsPol_;
        const uint32_t& nkPol       = nkPol_;
        const std::vector<double>& Coulomb2CValues = Coulomb2CValues_;
        const std::vector<std::array<uint32_t,3>>& Coulomb2CIndices = Coulomb2CIndices_;
        const uint64_t& nvaluesC2c = nvaluesC2c_;
        const arma::mat& RlistFrac_C2c = RlistFrac_C2c_;
        const uint& ncellsC2c = ncellsC2c_;
        const std::vector<double>& metric2CValues = metric2CValues_;
        const std::vector<std::array<uint32_t,3>>& metric2CIndices = metric2CIndices_;
        const uint64_t& nvaluesM2c = nvaluesM2c_;
        const arma::mat& RlistFrac_M2c = RlistFrac_M2c_;
        const uint& ncellsM2c = ncellsM2c_;
        const std::vector<double>& metric3CValues = metric3CValues_;
        const std::vector<std::array<uint32_t,5>>& metric3CIndices = metric3CIndices_;
        const uint64_t& nvaluesM3c = nvaluesM3c_;
        const arma::mat& RlistFrac_M3c = RlistFrac_M3c_;
        const uint& ncellsM3c = ncellsM3c_;
        const uint32_t& norbitals_AUX = norbitals_AUX_;
        const bool& SOC_FLAG = SOC_FLAG_;
        const bool& MAGNETIC_FLAG = MAGNETIC_FLAG_;

    protected:
        SystemGTF_MPI() = default; 
    public:
        SystemGTF_MPI(const ConfigurationCRYSTAL_MPI&, const int procMPI_rank, const int procMPI_size, const uint metric, const bool is_for_Dk, const std::string& intName = "", const std::string& int_dir = "Results/1-Integrals/", const bool loadInt = true);

    protected:
        // Initialize either the Hamiltonian or the alpha & beta pointers of System, depending on the spin casuistic
        void initializeHamiltonian(const ConfigurationCRYSTAL_MPI& CRYSTALconfig); 
        // Loads the previously computed matrix elements (above a predefined tolerance) and corresponding indices of 2-center Coulomb 
        // integrals in the AUX basis, and the corresponding list of R-vectors
        void loadCoulomb2C(const int procMPI_rank, const int procMPI_size, const bool is_for_Dk = false, const std::string& intName = "", const std::string& integrals_directory = "Results/1-Integrals/");
        // Loads the previously computed matrix elements (above a predefined tolerance) and corresponding indices of 2-center Ewald 
        // integrals in the AUX basis, and the corresponding list of R-vectors within the supercell
        void loadEwald2C(const int procMPI_rank, const int procMPI_size, const bool is_for_Dk = false, const std::string& intName = "", const std::string& integrals_directory = "Results/1-Integrals/");
        // Loads the previously computed matrix elements (above a predefined tolerance) and corresponding indices of 2-center metric 
        // integrals in the AUX basis, and the corresponding list of R-vectors
        void loadMetric2C(const int procMPI_rank, const int procMPI_size, const uint metric, const std::string& intName = "", const std::string& integrals_directory = "Results/1-Integrals/");
        // Loads the previously computed matrix elements (above a predefined tolerance) and corresponding indices of 3-center metric
        // integrals in the mixed SCF and AUX basis, and the corresponding list of R-vectors
        void loadMetric3C(const int procMPI_rank, const int procMPI_size, const uint metric, const std::string& intName = "", const std::string& integrals_directory = "Results/1-Integrals/");

    public: 
        // Initialize the k-point grids for the exciton basis and for the BZ-integration of the auxiliary polarizability tensor Pi
        void initializekGrids(const std::vector<int32_t>& nkiBSE, const std::vector<int32_t>& nkiPol);
        // Compute the single-particle energies and wave-function coefficients at a given k-points
        void solveBands(const arma::colvec& k, arma::colvec& eigval, arma::cx_mat& eigvec) override;
        // Group the entries of the (k,k') matrix (with k,k' = k_0,..,k_nkBSE) with the same k'' = abs(k-k') (= k_i, for some i)
        std::vector<std::vector<std::array<uint32_t,3>>> generatekentries(const std::vector<int32_t>& nkiBSE, const arma::ucolvec& selected_chunks);

};

}