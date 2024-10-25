#pragma once
#include "xatu/ConfigurationExciton_MPI.hpp"
#include "xatu/SystemGTF_MPI.hpp"

#ifndef constants
#define EV2HARTREE 0.03674932218  
#endif

namespace xatu {

/**
 * The ExcitonGTF contains the exciton attributes not related to the k-grid, and the final methods to construct and diagonalize
 * the BSE Hamiltonian. Exclusive to the GAUSSIAN mode
 */
class ExcitonGTF_MPI : public virtual SystemGTF_MPI {

    protected:
        // Number of valence and conduction single-particle bands included in the exciton basis
        uint nvbands_, ncbands_;
        // List of bands included in the exciton basis, where 0 is the highest valence band (conduction >=1, valence <=0)
        arma::ivec bands_;
        // List of valence bands that form the exciton, with 0 being the lowest single-particle occupied band
        arma::ucolvec valenceBands_;
        // List of conduction bands that form the exciton, with the same enumeration criterion as valenceBands (highestValenceBand the highest valence band)
        arma::ucolvec conductionBands_;
        // Total list of bands used to build the exciton (union of valenceBands and conductionBands, with the same enumeration criterion)
        arma::ucolvec bandListBSE_;
        // Total number of single-particle bands in the exciton basis
        uint nbandsBSE_;
        // Dimension (number of rows or columns) of the (k,k')-blocks in the BSE Hamiltonian
        uint dimkblock_;
        // Number of elements in the lower triangle (including main diagonal) of each (k,k')-block
        uint dimkblock_triang_;
        // 3-component column vector with the center-of-mass momentum (Q quantum number) of the exciton
        arma::colvec Q_;
        // Small regularization parameter to circumvent potential (quasi-) linear dependencies in the calculation of the inverse matrix.
        // See Supporting Information of J. Chem. Theory Comput. 2024, 20, 2202−2208 ; where default value of alpha = 0.01 is suggested
        double alpha_;
        // Scissor cut to correct the bandgap, in Hartree units
        double scissor_;
        // Dimension of the BSE Hamiltonian, or equivalently of the basis for electron-hole pairs used to build excitons
        uint32_t dimBSE_;

    public:
        const uint& nvbands = nvbands_;
        const uint& ncbands = ncbands_;
        const arma::ivec& bands = bands_;
        const arma::ucolvec& valenceBands = valenceBands_;
        const arma::ucolvec& conductionBands = conductionBands_;
        const arma::ucolvec& bandListBSE = bandListBSE_;
        const uint& nbandsBSE = nbandsBSE_;
        const uint& dimkblock = dimkblock_;
        const uint& dimkblock_triang = dimkblock_triang_;
        const arma::colvec& Q = Q_;
        const double& alpha = alpha_;
        const double& scissor = scissor_;
        const uint32_t& dimBSE = dimBSE_;

    protected:
        ExcitonGTF_MPI() = default; 
    public:
        ExcitonGTF_MPI(const ConfigurationExciton_MPI&, const ConfigurationCRYSTAL_MPI&, const int procMPI_rank, const int procMPI_size, const uint metric, const bool is_for_Dk = false, const std::string& intName = "", const std::string& int_dir = "Results/1-Integrals/", const bool loadInt = true);

    protected:
        // Initialize the attributes related to the single-particle bands involved in the exciton basis
        void initializeExcitonBands(const ConfigurationExciton_MPI&);

    public:
        // Computes and returns the matrix product J^(1/2)_{k} * [M + alpha*I]^(-1)_{k}, where J_{k} is the FT of the 2-center Coulomb
        // matrices and M_{k} is (assuming the overlap metric) the FT of the 2-center overlap matrices, both in the AUX basis
        arma::cx_mat computeJMproduct(const arma::colvec& k);
        // Computes and returns the auxiliary irreducible polarizability tensor Pi(k) in the AUX basis, assuming that the bands are centro-symmetric
        arma::cx_mat computePik(const arma::colvec& k, const arma::cx_mat& JMprod);
        
        // CURRENTLY NOT NEEDED
        arma::cx_cube computevPmk1nk2(const arma::colvec& k1, const arma::colvec& k2, const arma::cx_mat& JMprod, const arma::ucolvec& mlist, const arma::ucolvec& nlist);
        arma::cx_mat computevPmk1nk2_row(const arma::colvec& k1, const arma::colvec& k2, const arma::cx_mat& JMprod, const arma::ucolvec& mlist, const arma::ucolvec& nlist);
        // arma::cx_mat computevPmk1nk2_row(const arma::colvec& k1, const arma::colvec& k2, const arma::cx_mat& JMprod);


};

}