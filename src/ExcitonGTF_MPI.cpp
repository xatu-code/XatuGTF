#include "xatu/ExcitonGTF_MPI.hpp"

namespace xatu {

/**
 * Configuration constructor from a ConfigurationExciton and a ConfigurationCRYSTAL object.
 * @param ExcitonConfig ConfigurationExciton object obtained from an exciton file.
 * @param CRYSTALconfig ConfigurationCRYSTAL object obtained from an .outp file.
 * @param metric 0 for the overlap metric, 1 for the attenuated Coulomb metric.
 * @param is_for_Dk If true, the integrals and lattice vectores will be stored with the extension .E2cDk (or .C2cDk) instead of the usual .E2c (.C2c)
 * @param intName Name of the files where the integral matrix elements are stored (C2Mat_intName.C2c, o2Mat_intName.o2c, o3Mat_intName.o3c).
 * @param int_dir Location where the integrals and list of Bravais vectors files are stored.
 * @param loadInt True (false) if the metric and Coulomb/Ewald integrals are (not) to be loaded. If false, the previous 2 arguments are irrelevant. 
*/
ExcitonGTF_MPI::ExcitonGTF_MPI(const ConfigurationExciton_MPI& ExcitonConfig, const ConfigurationCRYSTAL_MPI& CRYSTALconfig, const int procMPI_rank, const int procMPI_size, const uint metric, const bool is_for_Dk, const std::string& intName, const std::string& int_dir, const bool loadInt) 
    : SystemGTF_MPI(CRYSTALconfig, procMPI_rank, procMPI_size, metric, is_for_Dk, intName, int_dir, loadInt){

    initializeExcitonBands(ExcitonConfig);
    this->Q_       = ExcitonConfig.excitonInfoGTF.Q;
    this->alpha_   = ExcitonConfig.excitonInfoGTF.alpha;
    this->scissor_ = (ExcitonConfig.excitonInfoGTF.scissor) * EV2HARTREE;

    std::vector<int32_t> nkiBSE = ExcitonConfig.excitonInfoGTF.nki;
    std::vector<int32_t> nkiPol = ExcitonConfig.excitonInfoGTF.nkiPol;
    initializekGrids(nkiBSE, nkiPol);
    
    this->dimkblock_ = static_cast<uint>(nvbands_ * ncbands_);
    this->dimBSE_ = dimkblock_ * nkBSE_;
    this->dimkblock_triang_ = (dimkblock*(dimkblock + 1))/2;

}

/**
 * Initialize the attributes related to the single-particle bands involved in the exciton basis.
 * @param ExcitonConfig ConfigurationExciton object obtained from an exciton file.
 * @return void.
*/
void ExcitonGTF_MPI::initializeExcitonBands(const ConfigurationExciton_MPI& ExcitonConfig){

    if(ExcitonConfig.excitonInfoGTF.bands.empty()){
        this->nvbands_ = static_cast<uint>(ExcitonConfig.excitonInfoGTF.nvbands);
        this->ncbands_ = static_cast<uint>(ExcitonConfig.excitonInfoGTF.ncbands);
        this->bands_   = arma::regspace<arma::ivec>(- static_cast<int>(nvbands) + 1, ncbands);
    } else {
        this->bands_   = ExcitonConfig.excitonInfoGTF.bands;
        arma::uvec vbands = arma::find(bands_ <= 0);
        this->nvbands_ = vbands.n_elem;
        this->ncbands_ = bands_.n_elem - nvbands_;
    }

    arma::ucolvec valenceBands(nvbands_);
    arma::ucolvec conductionBands(ncbands_);
    int vcountr = 0;
    int ccountr = 0;
    for(int32_t b : bands_){
        if(b <= 0){
            valenceBands(vcountr) = static_cast<uint32_t>(b + highestValenceBand_);
            vcountr++;
        } else {
            conductionBands(ccountr) = static_cast<uint32_t>(b + highestValenceBand_);
            ccountr++;
        }
    }    
    this->valenceBands_    = valenceBands;
    this->conductionBands_ = conductionBands;
    this->bandListBSE_     = arma::conv_to<arma::uvec>::from(arma::join_vert(valenceBands_, conductionBands_));
    this->nbandsBSE_       = bandListBSE_.n_elem;

}

/**
 * Computes and returns the matrix product J^(1/2)_{k} * [M + alpha*I]^(-1)_{k}, where J_{k} is the FT of the 2-center Coulomb
 * matrices and M_{k} is the FT of the 2-center metric matrices, both in the AUX basis.
 * @param k k-point in fractional coordinates where the FT's of both J(R) and M(k) are computed.
 * @return arma::cx_mat J^(1/2)_{k} * [M + alpha*I]^(-1)_{k}, which has dimensions (norbitals_AUX,norbitals_AUX).
*/
arma::cx_mat ExcitonGTF_MPI::computeJMproduct(const arma::colvec& k){

    // 2-center metric (metric) matrix
    arma::cx_mat Mk = arma::zeros<arma::cx_mat>(norbitals_AUX_, norbitals_AUX_);
    arma::cx_rowvec eiKRvec_M = arma::exp( imag*TWOPI*(k.t()*RlistFrac_M2c_) );
    for(uint64_t i = 0; i < nvaluesM2c_; i++){
        Mk(metric2CIndices_[i][0], metric2CIndices_[i][1]) += metric2CValues_[i] * eiKRvec_M(metric2CIndices_[i][2]);
    }
    Mk.diag() *= 0.5;  //complete the upper triangle due to hermiticity
    Mk += Mk.t();
    arma::cx_mat Mk_inv;
    bool Mk_inv_bool = arma::inv_sympd(Mk_inv, Mk + alpha_*arma::eye(norbitals_AUX_,norbitals_AUX_) );
    if(!Mk_inv_bool){ //this will trigger if Mk has spurious negative or (almost) zero eigenvalues
        std::cerr << "WARNING! k-dependent 2-center metric matrix in the AUX basis is not positive definite. Consider increasing the alpha parameter or the number of R-vectors in its series, or reducing the AUX basis" << std::endl;
        Mk_inv_bool = arma::inv(Mk_inv, Mk + alpha_*arma::eye(norbitals_AUX_,norbitals_AUX_) ); //try general inverse algorithm, otherwise stop execution because Mk is singular
        if(!Mk_inv_bool){
            throw std::logic_error("SystemGTF::JMproduct: k-dependent 2-center metric matrix in the AUX basis appears to be singular. Consider reducing the AUX basis");
        }
    }

    // 2-center Coulomb matrix 
    arma::cx_mat Jk = arma::zeros<arma::cx_mat>(norbitals_AUX_, norbitals_AUX_);
    arma::cx_rowvec eiKRvec_J = arma::exp( imag*TWOPI*(k.t()*RlistFrac_C2c_) );
    for(uint64_t i = 0; i < nvaluesC2c_; i++){
        Jk(Coulomb2CIndices_[i][0], Coulomb2CIndices_[i][1]) += Coulomb2CValues_[i] * eiKRvec_J(Coulomb2CIndices_[i][2]);
    }
    Jk.diag() *= 0.5;  //complete the upper triangle due to hermiticity
    Jk += Jk.t();
    if(std::abs(arma::sum(eiKRvec_J)) > 1e-6){
        std::cerr << "WARNING! The list of R-vectors for the 2-center Coulomb integrals may be inadequate" << std::endl;
    }
    arma::colvec Jkeigvals;
    arma::cx_mat Jkeigvecs;
    eig_sym(Jkeigvals, Jkeigvecs, Jk);
    Jkeigvals.transform( [](double val) { return (val <= 0.0) ? 0.0 : std::sqrt(val); } ); //remove possible spurious negative eigenvalues, and take the square root
    arma::cx_mat Jk_sqrt = Jkeigvecs*arma::diagmat(Jkeigvals)*(Jkeigvecs.t());

    return (Jk_sqrt * Mk_inv);

}

/**
 * Computes and returns the auxiliary irreducible polarizability tensor Pi(k) in the AUX basis, assuming that the bands are centro-symmetric.
 * @param k k-point in fractional coordinates where the FT's of both J(R) and M(k) are computed.
 * @return JMprod J^(1/2)_{k} * [M + alpha*I]^(-1)_{k}, as given by computeJMproduct(k).
 * @return arma::cx_mat Pi(k) in the AUX basis.
*/
arma::cx_mat ExcitonGTF_MPI::computePik(const arma::colvec& k, const arma::cx_mat& JMprod){

#pragma omp declare reduction( add_cxmat : arma::cx_mat : omp_out += omp_in ) initializer( omp_priv = omp_orig )   

    arma::cx_mat Pik = arma::zeros<arma::cx_mat>(norbitals_AUX_,norbitals_AUX_);
    
    #pragma omp parallel for reduction(add_cxmat : Pik)
    for(uint32_t kind = 0; kind < nkPol_; kind++){
        // Compute v_{P}^{mk1,nk2} and the single-particle energies
        arma::cx_cube vPmk1nk2;
        arma::colvec k1 = kpointsPol_.col(kind);     // this is \tilde{k} in my paper's notation 
        arma::colvec k2 = kpointsPol_.col(kind) + k;
        arma::colvec eigval1, eigval2;
        { 
            arma::cx_mat eigvec1, eigvec2;
            solveBands(k1, eigval1, eigvec1);
            solveBands(k2, eigval2, eigvec2);
        
            arma::cx_mat eiKR2mat = arma::zeros<arma::cx_mat>(ncellsM3c_ + 1,ncellsM3c_ + 1);
            for(uint i1 = 0; i1 <= ncellsM3c_; i1++){
                double k1R1 = arma::dot(k1,RlistFrac_M3c_.col(i1));
                for(uint i2 = 0; i2 <= ncellsM3c_; i2++){
                    eiKR2mat(i1,i2) = std::exp( imag*TWOPI*(arma::dot(k2, RlistFrac_M3c_.col(i2)) - k1R1) );
                }
            }

            arma::cx_cube vPmuk1nuk2_pre = arma::zeros<arma::cx_cube>(norbitals_,norbitals_,norbitals_AUX_);
            for(uint64_t i = 0; i < nvaluesM3c_; i++){
                vPmuk1nuk2_pre(metric3CIndices_[i][1], metric3CIndices_[i][2], metric3CIndices_[i][0]) += metric3CValues_[i] * eiKR2mat(metric3CIndices_[i][3], metric3CIndices_[i][4]);
            }
        
            vPmk1nk2 = arma::zeros<arma::cx_cube>(norbitals_,norbitals_,norbitals_AUX_);
            arma::cx_mat vPmuk1nuk2_slice;
            for(uint32_t sl_outer = 0; sl_outer < norbitals_AUX_; sl_outer++){
                vPmuk1nuk2_slice = arma::zeros<arma::cx_mat>(norbitals_,norbitals_);
                for(uint32_t sl_inner = 0; sl_inner < norbitals_AUX_; sl_inner++){
                    vPmuk1nuk2_slice += vPmuk1nuk2_pre.slice(sl_inner) * JMprod(sl_outer, sl_inner);
                }
                vPmk1nk2.slice(sl_outer) = eigvec1.t() * vPmuk1nuk2_slice * eigvec2;
            }
        }
        // Complete the calculation of the kind-th \tilde{k}-summand of Pi_{R,R'}(k)
        arma::cx_mat invEnergDiff(filling_, norbitals_ - filling_);
        eigval2.subvec(filling_, norbitals_ - 1) += scissor_;    // apply scissor correction (by default 0) to conduction states
        for(int v = 0; v < filling_; v++){
            for(int32_t c = filling_; c < static_cast<int32_t>(norbitals_); c++){
             invEnergDiff(v, c - filling_) = 1./(eigval1(v) - eigval2(c));
            }
        }
        
        arma::cx_mat vsubmat1, vsubmat2;
        for(uint32_t P1 = 0; P1 < norbitals_AUX_; P1++){
            vsubmat1 = vPmk1nk2(arma::span(0, highestValenceBand_), arma::span(filling_, norbitals_-1), arma::span(P1));
            vsubmat1 %= invEnergDiff;
            for(uint32_t P2 = 0; P2 <= P1; P2++){ // compute only the lowest triangle, as the matrix Pi_{P,P'}(k) is hermitian
                vsubmat2 = vPmk1nk2(arma::span(0, highestValenceBand_), arma::span(filling_, norbitals_-1), arma::span(P2));
                Pik(P1,P2) += arma::accu( vsubmat1 % arma::conj(vsubmat2) );
            }
        }   
        
    }
    double spin_factor = (!SOC_FLAG && !MAGNETIC_FLAG)? 2. : 1.;
    Pik *= 2.*spin_factor/(double)nkPol_;      // normalization factor
    Pik.diag() *= 0.5;  // reconstruct the upper triangle due to hermicity
    Pik += Pik.t();

    return Pik;
    
}

/**
 * Embedded in computePik
*/
arma::cx_cube ExcitonGTF_MPI::computevPmk1nk2(const arma::colvec& k1, const arma::colvec& k2, const arma::cx_mat& JMprod, const arma::ucolvec& mlist, const arma::ucolvec& nlist){

    arma::colvec eigval1, eigval2;
    arma::cx_mat eigvec1, eigvec2;
    solveBands(k1, eigval1, eigvec1);
    solveBands(k2, eigval2, eigvec2);
    eigvec1 = eigvec1.cols(mlist);
    eigvec2 = eigvec2.cols(nlist);
    
    arma::cx_mat eiKR2mat = arma::zeros<arma::cx_mat>(ncellsM3c_ + 1,ncellsM3c_ + 1);
    for(uint i1 = 0; i1 <= ncellsM3c_; i1++){
        double k1R1 = arma::dot(k1,RlistFrac_M3c_.col(i1));
        for(uint i2 = 0; i2 <= ncellsM3c_; i2++){
            eiKR2mat(i1,i2) = std::exp( imag*TWOPI*(arma::dot(k2, RlistFrac_M3c_.col(i2)) - k1R1) );
        }
    }
    // for(uint32_t i = 0; i <= ncellsM3c_*ncellsM3c_; i++){
    //     uint32_t i1 = i % ncellsM3c_;
    //     uint32_t i2 = i / ncellsM3c_;
    //     eiKR2mat(i1,i2) = std::exp( imag*TWOPI*( arma::dot(k2, RlistFrac_M3c_.col(i2)) - arma::dot(k1, RlistFrac_M3c_.col(i1)) ) );
    // }

    arma::cx_cube vPmuk1nuk2_pre = arma::zeros<arma::cx_cube>(norbitals_,norbitals_,norbitals_AUX_);
    for(uint64_t i = 0; i < nvaluesM3c_; i++){
        vPmuk1nuk2_pre(metric3CIndices_[i][1], metric3CIndices_[i][2], metric3CIndices_[i][0]) += metric3CValues_[i] * eiKR2mat(metric3CIndices_[i][3], metric3CIndices_[i][4]);
    }
    
    arma::cx_cube vPmk1nk2 = arma::zeros<arma::cx_cube>(mlist.n_elem, nlist.n_elem, norbitals_AUX_);
    arma::cx_mat vPmuk1nuk2_slice;
    for(uint32_t sl_outer = 0; sl_outer < norbitals_AUX_; sl_outer++){
        vPmuk1nuk2_slice = arma::zeros<arma::cx_mat>(norbitals_,norbitals_);
        for(uint32_t sl_inner = 0; sl_inner < norbitals_AUX_; sl_inner++){
            vPmuk1nuk2_slice += vPmuk1nuk2_pre.slice(sl_inner) * JMprod(sl_outer, sl_inner);
        }
        vPmk1nk2.slice(sl_outer) = eigvec1.t() * vPmuk1nuk2_slice * eigvec2;
    }

    return vPmk1nk2;

}


/** CURRENTLY NOT NEEDED
 * k1, k2 are in fractional coordinates!
*/
arma::cx_mat ExcitonGTF_MPI::computevPmk1nk2_row(const arma::colvec& k1, const arma::colvec& k2, const arma::cx_mat& JMprod, const arma::ucolvec& mlist, const arma::ucolvec& nlist){

    arma::colvec eigval1, eigval2;
    arma::cx_mat eigvec1, eigvec2;
    solveBands(k1, eigval1, eigvec1);
    solveBands(k2, eigval2, eigvec2);
    eigvec1 = eigvec1.cols(mlist);
    eigvec2 = eigvec2.cols(nlist);
    
    arma::cx_mat eiKR2mat = arma::zeros<arma::cx_mat>(ncellsM3c_ + 1,ncellsM3c_ + 1);
    for(uint i1 = 0; i1 <= ncellsM3c_; i1++){
        double k1R1 = arma::dot(k1,RlistFrac_M3c_.col(i1));
        for(uint i2 = 0; i2 <= ncellsM3c_; i2++){
            eiKR2mat(i1,i2) = std::exp( imag*TWOPI*(arma::dot(k2, RlistFrac_M3c_.col(i2)) - k1R1) );
        }
    }

    arma::cx_mat vPmuk1nuk2_pre = arma::zeros<arma::cx_mat>(norbitals_AUX_, norbitals_*norbitals_);
    for(uint64_t i = 0; i < nvaluesM3c_; i++){
        vPmuk1nuk2_pre(metric3CIndices_[i][0], metric3CIndices_[i][1] + norbitals_*metric3CIndices_[i][2]) += metric3CValues_[i] * eiKR2mat(metric3CIndices_[i][3], metric3CIndices_[i][4]);
    }
    vPmuk1nuk2_pre = JMprod * vPmuk1nuk2_pre;
    return (vPmuk1nuk2_pre * arma::kron(eigvec2, arma::conj(eigvec1)));

}

/** CURRENTLY NOT NEEDED
 * k1, k2 are in fractional coordinates!
*/
// arma::cx_mat ExcitonGTF_MPI::computevPmk1nk2_row(const arma::colvec& k1, const arma::colvec& k2, const arma::cx_mat& JMprod){

//     arma::colvec eigval1, eigval2;
//     arma::cx_mat eigvec1, eigvec2;
//     solveBands(k1, eigval1, eigvec1);
//     solveBands(k2, eigval2, eigvec2);
//     eigvec1 = eigvec1.cols(bandListBSE_);
//     eigvec2 = eigvec2.cols(bandListBSE_);
    
//     arma::cx_mat eiKR2mat = arma::zeros<arma::cx_mat>(ncellsM3c_ + 1,ncellsM3c_ + 1);
//     for(uint i1 = 0; i1 <= ncellsM3c_; i1++){
//         double k1R1 = arma::dot(k1,RlistFrac_M3c_.col(i1));
//         for(uint i2 = 0; i2 <= ncellsM3c_; i2++){
//             eiKR2mat(i1,i2) = std::exp( imag*TWOPI*(arma::dot(k2, RlistFrac_M3c_.col(i2)) - k1R1) );
//         }
//     }

//     arma::cx_mat vPmuk1nuk2_pre = arma::zeros<arma::cx_mat>(norbitals_AUX_, norbitals_*norbitals_);
//     for(uint64_t i = 0; i < nvaluesM3c_; i++){
//         vPmuk1nuk2_pre(metric3CIndices_[i][0], metric3CIndices_[i][1] + norbitals_*metric3CIndices_[i][2]) += metric3CValues_[i] * eiKR2mat(metric3CIndices_[i][3], metric3CIndices_[i][4]);
//     }
//     vPmuk1nuk2_pre = JMprod * vPmuk1nuk2_pre;
//     return (vPmuk1nuk2_pre * arma::kron(eigvec2, arma::conj(eigvec1)));

// }


}