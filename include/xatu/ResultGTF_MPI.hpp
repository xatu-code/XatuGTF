#pragma once
#include <mpi.h>
#include "xatu/ExcitonGTF_MPI.hpp"

namespace xatu { 

/**
 * The ResultGTF_MPI class contains common routines needed for post-processing of the BSE solutions, or for related single-particle calculations.
 * Exclusive to the GAUSSIAN mode
 */
class ResultGTF_MPI : public ExcitonGTF_MPI {

    protected:
        // Dipole integrals in the SCF basis, in atomic units. The 3 components are, in order, (x,y,z); at least one of which is above a certain tolerance
        std::vector<std::array<double,3>> dipoleValues_;
        // (mu, mu', R) indices corresponding to each element in dipoleValues_
        std::vector<std::array<uint32_t,3>> dipoleIndices_;
        // Number of dipole matrix elements (triplets) that are stored in dipoleValues
        uint64_t nvaluesDip_;
        // List of R-vectors in fractional coordinates for which the dipole matrix elements were computed, in the same ordering and stored by columns
        arma::mat RlistFrac_dip_;
        // Number of unit cells for which the dipole matrix elements are stored 
        uint ncellsDip_;
        // Eigenvalues of the BSE Hamiltonian, stored in atomic units. Contains all the values, irrespective of the number of stored eigenvectors
        arma::colvec Eexc_;
        // Eigenvectors of the BSE Hamiltonian, stored by columns. Contains only the first nA states (given in the constructor)
        arma::cx_mat Aexc_;
        
    public:
        const std::vector<std::array<double,3>>& dipoleValues = dipoleValues_;
        const std::vector<std::array<uint32_t,3>>& dipoleIndices = dipoleIndices_;
        const uint64_t& nvaluesDip = nvaluesDip_;
        const arma::mat& RlistFrac_dip = RlistFrac_dip_;
        const uint& ncellsDip = ncellsDip_;
        const arma::colvec& Eexc = Eexc_;
        const arma::cx_mat& Aexc = Aexc_;

    protected:
        ResultGTF_MPI() = default; 
    public:
        ResultGTF_MPI(const ConfigurationExciton_MPI&, const ConfigurationCRYSTAL_MPI&, const int procMPI_rank, const int procMPI_size, const uint32_t nA, const std::string& intName = "", const std::string& excName = "BSE", const std::string& int_dir = "Results/1-Integrals/", const std::string& exc_dir = "Results/3-Excitons/");
        ResultGTF_MPI(const ConfigurationCRYSTAL_MPI&, const int procMPI_rank, const int procMPI_size, const std::string& intName = "", const std::string& int_dir = "Results/1-Integrals/");

    protected:
        // Loads the previously computed matrix elements and corresponding indices of dipole integrals (x,y,z) in the SCF basis, 
        // and the corresponding list of R-vectors. All the relevant quantities are stored as attributes
        void loadDipole(const int procMPI_rank, const int procMPI_size, const std::string& intName = "", const std::string& int_dir = "Results/1-Integrals/");
        // Loads the previously computed excitonic energies and (the first nA) wavefunction coefficients, and store them as attributes
        void loadExcitons(const int procMPI_rank, const int procMPI_size, const uint32_t nA, const std::string& excName = "BSE", const std::string& exc_dir = "");

    public:
        // Method to compute the velocity matrix elements in a k-point. The 3 spatial components are computed, and in atomic units
        arma::cx_mat velocities_vc(const arma::colvec& k, arma::colvec& sp_energies, const double scissor = 0.);

};

}