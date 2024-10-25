#include "xatu/ResultGTF_MPI.hpp"

namespace xatu {

/**
 * Configuration constructor from a ConfigurationExciton and a ConfigurationCRYSTAL object. For post-BSE calculations.
 * @param ExcitonConfig ConfigurationExciton object obtained from an exciton file.0
 * @param CRYSTALconfig ConfigurationCRYSTAL object obtained from the .outp file.
 * @param nA Number of BSE eigenvectors to be stored, in order of ascending energy.
 * @param intName Name of the files where the dipole integrals are stored (dipoleMat_intName.dip).
 * @param excName Name of the files where the exciton energies (excName.energ) and eigenvectors (excName.eigvec) are stored.
 * @param int_dir Location where the dipole integrals and list of Bravais vectors files are stored.
 * @param exc_dir Location where the exciton energies and eigenvectors files are stored.
*/
ResultGTF_MPI::ResultGTF_MPI(const ConfigurationExciton_MPI& ExcitonConfig, const ConfigurationCRYSTAL_MPI& CRYSTALconfig, const int procMPI_rank, const int procMPI_size, 
    const uint32_t nA, const std::string& intName, const std::string& excName, const std::string& int_dir, const std::string& exc_dir) 
        : SystemGTF_MPI(CRYSTALconfig, procMPI_rank, procMPI_size, 0, false, intName, int_dir, false), ExcitonGTF_MPI(ExcitonConfig, CRYSTALconfig, procMPI_rank, procMPI_size, 0, false, intName, int_dir, false){

    loadExcitons(procMPI_rank, procMPI_size, nA, excName, exc_dir);
    loadDipole(procMPI_rank, procMPI_size, intName, int_dir);
 
}

/**
 * Configuration constructor from a ConfigurationCRYSTAL object. For single-particle calculations.
 * @param CRYSTALconfig ConfigurationCRYSTAL object obtained from the .outp file.
 * @param intName Name of the files where the dipole integrals are stored (dipoleMat_intName.dip).
 * @param int_dir Location where the dipole integrals and list of Bravais vectors files are stored.
*/
ResultGTF_MPI::ResultGTF_MPI(const ConfigurationCRYSTAL_MPI& CRYSTALconfig, const int procMPI_rank, const int procMPI_size, const std::string& intName, const std::string& int_dir) 
        : SystemGTF_MPI(CRYSTALconfig, procMPI_rank, procMPI_size, 0, false, intName, int_dir, false){

    loadDipole(procMPI_rank, procMPI_size, intName, int_dir);

    // Include all valence and conduction bands for single-particle calculations
    this->nvbands_ = filling;
    this->ncbands_ = norbitals - filling;
    this->bands_   = arma::regspace<arma::ivec>(- static_cast<int>(nvbands) + 1, ncbands);
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
 
}

/**
 * Method to load the previously computed matrix elements and corresponding indices of dipole integrals (x,y,z) in the SCF basis, 
 * and the corresponding list of R-vectors. 
 * @param intName Name of the files where the dipole integrals are stored (dipoleMat_intName.dip).
 * @param integrals_directory Location where the dipole integrals and list of Bravais vectors files are stored.
 * @return void.
*/
void ResultGTF_MPI::loadDipole(const int procMPI_rank, const int procMPI_size, const std::string& intName, const std::string& integrals_directory){

    std::string dip_file = integrals_directory + "dipoleMat_" + intName + ".dip";
    uint64_t nvaluesDip;
    if(procMPI_rank == 0){
        if(dip_file.empty()){
            throw std::invalid_argument("ResultGTF::loadDipole: file must not be empty");
        }
        std::ifstream dip_ifstream;
        dip_ifstream.open(dip_file.c_str());
        if(!dip_ifstream.is_open()){
            throw std::invalid_argument("ResultGTF::loadDipole: file does not exist");
        }
        std::string line;
        uint32_t norbs_SCF;
        std::getline(dip_ifstream, line);
        std::getline(dip_ifstream, line);
        std::getline(dip_ifstream, line);
        std::getline(dip_ifstream, line);
        std::getline(dip_ifstream, line);
        std::istringstream iss0(line);
        iss0 >> nvaluesDip;
        std::getline(dip_ifstream, line);
        std::istringstream iss1(line);
        iss1 >> norbs_SCF;
        if(norbs_SCF != norbitals_){
            throw std::logic_error("Error loading the dipole matrix elements: the SCF basis set is not the same as for the CRYSTAL H and S matrices");
        }
        
        dip_ifstream.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);
    MPI_Bcast (&nvaluesDip, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    this->nvaluesDip_ = nvaluesDip;
    
    std::vector<double> dipoleValues_vec(3*nvaluesDip_);
    std::vector<uint32_t> dipoleIndices_vec(3*nvaluesDip_);

    if(procMPI_rank == 0){
        std::ifstream dip_ifstream;
        dip_ifstream.open(dip_file.c_str());
        std::string line;
        std::getline(dip_ifstream, line);
        std::getline(dip_ifstream, line);
        std::getline(dip_ifstream, line);
        std::getline(dip_ifstream, line);
        std::getline(dip_ifstream, line);
        std::getline(dip_ifstream, line);
        double valueX, valueY, valueZ;
        uint32_t muind1, muind2, Rind;
        for(uint64_t i = 0; i < nvaluesDip_; i++){
            std::getline(dip_ifstream, line);
            std::istringstream iss(line);
            iss >> valueX >> valueY >> valueZ >> muind1 >> muind2 >> Rind;
    
            dipoleValues_vec[i]                  = valueX;
            dipoleValues_vec[i + nvaluesDip_]    = valueY;
            dipoleValues_vec[i + 2*nvaluesDip_]  = valueZ;
            dipoleIndices_vec[i]                 = muind1;
            dipoleIndices_vec[i + nvaluesDip_]   = muind2;
            dipoleIndices_vec[i + 2*nvaluesDip_] = Rind;
        }
        dip_ifstream.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);

    MPI_Bcast (&dipoleValues_vec[0], 3*nvaluesDip_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (&dipoleIndices_vec[0], 3*nvaluesDip_, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    std::vector<std::array<double,3>> dipoleValues(nvaluesDip_);
    std::vector<std::array<uint32_t,3>> dipoleIndices(nvaluesDip_);
    for(uint64_t i = 0; i < nvaluesDip_; i++){
        dipoleValues[i] = {dipoleValues_vec[i], dipoleValues_vec[i + nvaluesDip_], dipoleValues_vec[i + 2*nvaluesDip_]};
        dipoleIndices[i] = {dipoleIndices_vec[i], dipoleIndices_vec[i + nvaluesDip_], dipoleIndices_vec[i + 2*nvaluesDip_]};
    }
    this->dipoleValues_  = dipoleValues;
    this->dipoleIndices_ = dipoleIndices;

    for(int r = 0; r < procMPI_size; r++){
        if(procMPI_rank == r){
            RlistFrac_dip_.load(integrals_directory + "RlistFrac_" + intName + ".dip", arma::arma_ascii);
            this->ncellsDip_ = RlistFrac_dip_.n_cols;
            if(r == procMPI_size - 1){
                std::cout << "Dipole integrals loaded from file" << std::endl;
            }
        }
        MPI_Barrier (MPI_COMM_WORLD);
    }

}

/**
 * Method to load the previously computed excitonic energies and (the first nA) wavefunction coefficients.
 * @param nA Number of BSE eigenvectors to be stored, in order of ascending energy.
 * @param excName Name of the files where the exciton energies (excName.energ) and eigenvectors (excName.eigvec) are stored.
 * @param exc_dir Location where the exciton energies and eigenvectors files are stored.
 * @return void.
*/
void ResultGTF_MPI::loadExcitons(const int procMPI_rank, const int procMPI_size, const uint32_t nA, const std::string& excName, const std::string& exc_dir){

    for(int r = 0; r < procMPI_size; r++){
        if(procMPI_rank == r){
            Eexc_.load(exc_dir + excName + ".energ", arma::arma_ascii);
            Eexc_ *= EV2HARTREE;   // convert from eV to Hartree, noting that the exciton energies are always stored in the former units
            Aexc_.load(exc_dir + excName + ".eigvec", arma::arma_ascii);
            if(procMPI_rank == 0){
                if(Aexc_.n_cols < nA){
                    throw std::invalid_argument("ERROR: more exciton states were requested than are stored");
                }
            }
            if(Aexc_.n_cols > nA){
                Aexc_.shed_cols(nA, Aexc_.n_cols - 1);  // take only the first nA states
            }
            if(r == procMPI_size - 1){
                std::cout << "Exciton energies and wavefunctions loaded from file" << std::endl;
            }
        }
        MPI_Barrier (MPI_COMM_WORLD);
    }

}

/**
 * Method to compute the velocity matrix elements (v,c) in a k-point. The 3 spatial components are computed, and in atomic units.
 * The phase criterion of "sum of coefficients be real" is imposed, see Journal of Chemical Theory and Computation, 19, 9416, 2023.
 * @param k k-point in fractional coordinates. Number of elements = dimensionality of the lattice.
 * @param sp_energies Real column vector that will store the single-particle energies. Dimension with magnetism (no SOC): (2*norbitals).
 * @param scissor Rigid upwards traslation of the conduction bands, in ATOMIC UNITS (Hartree). By default it is 0.
 * @return cx_mat The components of the velocity matrix elements, stacked horizontally as (v_x,v_y,v_z).
*/
arma::cx_mat ResultGTF_MPI::velocities_vc(const arma::colvec& k, arma::colvec& sp_energies, const double scissor){

    arma::colvec kAU = Gbasis_*k/ANG2AU;
    if(!MAGNETIC_FLAG_ || SOC_FLAG_){
        // Gradient of Hamiltonian and overlap 
        arma::cx_mat Hk = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);
        arma::cx_mat Sk = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);
        arma::cx_mat nabla_Hk_X = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);  // d_{kx}H
        arma::cx_mat nabla_Hk_Y = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);  // d_{ky}H
        arma::cx_mat nabla_Hk_Z = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);  // d_{kz}H
        arma::cx_mat nabla_Sk_X = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);  // d_{kx}S
        arma::cx_mat nabla_Sk_Y = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);  // d_{ky}S
        arma::cx_mat nabla_Sk_Z = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);  // d_{kz}S
        for(int i = 0; i < ncells; i++){
            arma::colvec RlistAUi = RlistAU_.col(i);
            std::complex<double> exp_ikR = std::exp(imag*arma::dot(kAU, RlistAUi));
            arma::cx_mat Hki = exp_ikR * ((*ptr_hamiltonianMatrices).slice(i));
            arma::cx_mat Ski = exp_ikR * ((*ptr_overlapMatrices).slice(i));
            Hk += Hki;
            Sk += Ski;
            nabla_Hk_X += RlistAUi(0) * Hki;  //global factor i included at the end
            nabla_Hk_Y += RlistAUi(1) * Hki;  //global factor i included at the end
            nabla_Hk_Z += RlistAUi(2) * Hki;  //global factor i included at the end
            nabla_Sk_X += RlistAUi(0) * Ski;  //global factor i included at the end
            nabla_Sk_Y += RlistAUi(1) * Ski;  //global factor i included at the end
            nabla_Sk_Z += RlistAUi(2) * Ski;  //global factor i included at the end
        }
        Hk.diag() *= 0.5;
        Hk += Hk.t();
        Sk.diag() *= 0.5;
        Sk += Sk.t();
        nabla_Hk_X -= (trimatl(nabla_Hk_X, -1)).t();
        nabla_Hk_Y -= (trimatl(nabla_Hk_Y, -1)).t();
        nabla_Hk_Z -= (trimatl(nabla_Hk_Z, -1)).t();
        nabla_Sk_X -= (trimatl(nabla_Sk_X, -1)).t();
        nabla_Sk_Y -= (trimatl(nabla_Sk_Y, -1)).t();
        nabla_Sk_Z -= (trimatl(nabla_Sk_Z, -1)).t();

        arma::vec eigval_S;
        arma::cx_mat eigvec_S;
        arma::eig_sym(eigval_S, eigvec_S, Sk);
        eigval_S = 1./arma::sqrt(arma::abs(eigval_S));
        arma::cx_mat X = eigvec_S*arma::diagmat(eigval_S);     //canonical orthogonalization (Szabo - eq 3.169)
        Hk = X.t()*Hk*X;
        Hk = 0.5*(Hk + Hk.t());

        arma::cx_mat sp_eigenvectors;          //single-particle eigenvectors
        arma::eig_sym(sp_energies, sp_eigenvectors, Hk);
        sp_energies += arma::join_vert(arma::zeros<arma::colvec>(filling), scissor*arma::ones<arma::colvec>(norbitals-filling)); //apply scissor correction
        sp_eigenvectors = X*sp_eigenvectors;
        sp_eigenvectors %= arma::repmat( arma::exp( -imag*arma::arg( arma::sum(sp_eigenvectors) ) ), norbitals_, 1); //make the sum of coefficients real for each eigenvector

        arma::mat Ev_rows = arma::repmat(sp_energies.subvec(valenceBands(0), valenceBands.back()), 1, norbitals_);
        arma::mat Ec_cols = (arma::repmat(sp_energies.subvec(conductionBands(0), conductionBands.back()), 1, nvbands)).t();
        arma::cx_mat sp_eigenvectors_1 = (sp_eigenvectors.cols(valenceBands)).t();
        arma::cx_mat sp_eigenvectors_2 = sp_eigenvectors.cols(conductionBands);
        arma::cx_mat Ev_sp_eigenvectors_1 = Ev_rows % sp_eigenvectors_1;

        arma::cx_mat v_pre_X = (sp_eigenvectors_1*nabla_Hk_X - Ev_sp_eigenvectors_1*nabla_Sk_X)*sp_eigenvectors_2;
        arma::cx_mat v_pre_Y = (sp_eigenvectors_1*nabla_Hk_Y - Ev_sp_eigenvectors_1*nabla_Sk_Y)*sp_eigenvectors_2;
        arma::cx_mat v_pre_Z = (sp_eigenvectors_1*nabla_Hk_Z - Ev_sp_eigenvectors_1*nabla_Sk_Z)*sp_eigenvectors_2;

        // Dipole k-matrix (employs a different list of R-vectors in general)
        arma::cx_mat rk_X = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);        // r_{x}(k)
        arma::cx_mat rk_Y = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);        // r_{y}(k)
        arma::cx_mat rk_Z = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);        // r_{z}(k)
        arma::cx_rowvec eiKRvec_r = arma::exp( imag*TWOPI*(k.t()*RlistFrac_dip_) );
        for(uint64_t i = 0; i < nvaluesDip_; i++){
            std::complex<double> exp_ikR = eiKRvec_r(dipoleIndices_[i][2]);
            rk_X(dipoleIndices_[i][0], dipoleIndices_[i][1]) += dipoleValues_[i][0] * exp_ikR;
            rk_Y(dipoleIndices_[i][0], dipoleIndices_[i][1]) += dipoleValues_[i][1] * exp_ikR;
            rk_Z(dipoleIndices_[i][0], dipoleIndices_[i][1]) += dipoleValues_[i][2] * exp_ikR;
        }
        arma::mat Evc_mat = Ev_rows.cols(conductionBands) - Ec_cols;
        v_pre_X += ( (sp_eigenvectors_1*rk_X*sp_eigenvectors_2) % Evc_mat );
        v_pre_Y += ( (sp_eigenvectors_1*rk_Y*sp_eigenvectors_2) % Evc_mat );
        v_pre_Z += ( (sp_eigenvectors_1*rk_Z*sp_eigenvectors_2) % Evc_mat );
        
        arma::cx_mat vmats = arma::join_horiz(v_pre_X, v_pre_Y, v_pre_Z);
        
        return imag * vmats;
    }
    else { // CRYSTALconfig.MAGNETIC_FLAG && !CRYSTALconfig.SOC_FLAG (magnetism without SOC)
        throw std::invalid_argument("ResultGTF: velocities not implemented yet for the magnetic case w/o SOC");
    }

}


}