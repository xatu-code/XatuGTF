#include "xatu/SystemGTF_MPI.hpp"

namespace xatu {

/**
 * Configuration constructor from a ConfigurationCRYSTAL object. It does not initialize the k-grids, which are delegated to ExcitonGTF.
 * @param CRYSTALconfig ConfigurationCRYSTAL object obtained from the .outp file.
 * @param metric 0 for the overlap metric, 1 for the attenuated Coulomb metric.
 * @param is_for_Dk If true, the integrals and lattice vectores will be stored with the extension .E2cDk (or .C2cDk) instead of the usual .E2c (.C2c)
 * @param intName Name of the files where the integral matrix elements are stored (C2Mat_intName.C2c, o2Mat_intName.o2c, o3Mat_intName.o3c).
 * @param int_dir Location where the integrals and list of Bravais vectors files are stored.
 * @param loadInt True (false) if the metric and Coulomb/Ewald integrals are (not) to be loaded. If false, the previous 2 arguments are irrelevant.
*/
SystemGTF_MPI::SystemGTF_MPI(const ConfigurationCRYSTAL_MPI& CRYSTALconfig, const int procMPI_rank, const int procMPI_size, const uint metric, const bool is_for_Dk, const std::string& intName, const std::string& int_dir, const bool loadInt) 
    : System(CRYSTALconfig){

    this->RlistAU_       = ANG2AU*(CRYSTALconfig.Rlist); //convert from Angstrom to atomic units
    this->SOC_FLAG_      = CRYSTALconfig.SOC_FLAG;
    this->MAGNETIC_FLAG_ = CRYSTALconfig.MAGNETIC_FLAG;
    initializeHamiltonian(CRYSTALconfig);

    if(loadInt){
        if(ndim == 1){
            loadCoulomb2C(procMPI_rank, procMPI_size, is_for_Dk, intName, int_dir);
        } 
        else if (ndim >= 2){
            loadEwald2C(procMPI_rank, procMPI_size, is_for_Dk, intName, int_dir);
        }
        loadMetric2C(procMPI_rank, procMPI_size, metric, intName, int_dir);
        loadMetric3C(procMPI_rank, procMPI_size, metric, intName, int_dir);
    }
 
}

/**
 * Initialize either the Hamiltonian or the alpha & beta pointers of System, depending on the spin casuistic.
 * The unused pointer attributes are set to nullptr.
 * @param CRYSTALconfig ConfigurationCRYSTAL object.
 * @return void.
*/
void SystemGTF_MPI::initializeHamiltonian(const ConfigurationCRYSTAL_MPI& CRYSTALconfig){
    
    if(!MAGNETIC_FLAG || SOC_FLAG){
		this->ptr_hamiltonianMatrices = &CRYSTALconfig.hamiltonianMatrices;
		this->ptr_alphaMatrices       = nullptr;
		this->ptr_betaMatrices        = nullptr;
	} 
	else { // CRYSTALconfig.MAGNETIC_FLAG && !CRYSTALconfig.SOC_FLAG (magnetism without SOC)
		this->ptr_hamiltonianMatrices = nullptr;
		this->ptr_alphaMatrices       = &CRYSTALconfig.alphaMatrices;
		this->ptr_betaMatrices        = &CRYSTALconfig.betaMatrices;
	}

}

/**
 * Initialize the k-point grids for the exciton basis and for the BZ-integration of the auxiliary polarizability tensor Pi.
 * It also sets the number of |k-k'| chunks that will be distributed among MPI processes.
 * @param nkiBSE Vector with the number of k-points in each Gi direction for the BSE.
 * @param nkiPol Vector with the number of k-points in each Gi direction for the auxiliary polarizability matrix Pi.
 * @return void.
*/
void SystemGTF_MPI::initializekGrids(const std::vector<int32_t>& nkiBSE, const std::vector<int32_t>& nkiPol){

    int32_t n1 = 0;
    int32_t n2 = 0;
    int32_t n3 = 0;
    unify_ni(nkiBSE,n1,n2,n3);
    std::vector<int32_t> nkiBSE_unified = {n1,n2,n3};
    int32_t n1Pol = 0;
    int32_t n2Pol = 0;
    int32_t n3Pol = 0;
    unify_ni(nkiPol,n1Pol,n2Pol,n3Pol);
    std::vector<int32_t> nkiPol_unified = {n1Pol,n2Pol,n3Pol};

    this->kpointsBSE_ = generateBSEgrid(nkiBSE_unified, true);
    this->nkBSE_      = kpointsBSE.n_cols;

    int nkchunks_pre = 1;
    for(int d = 0; d < ndim; d++){
        nkchunks_pre *= 2 - (nkiBSE_unified[d] % 2);
    }
    this->nAbsk_ = (nkBSE_ + nkchunks_pre)/2;

    this->kpointsPol_ = generateBSEgrid(nkiPol_unified, true);
    this->nkPol_      = kpointsPol.n_cols;

}

/**
 * Method to load the previously computed matrix elements (above a predefined tolerance) and corresponding indices of 2-center Coulomb 
 * integrals in the AUX basis, and the corresponding list of R-vectors.
 * @param is_for_Dk If true, the integrals and lattice vectores will be stored with the extension .C2cDk instead of the usual .C2c 
 * @param intName Name of the file where the 2-center Coulomb matrix elements are stored (C2Mat_intName.C2c).
 * @param integrals_directory Directory where the file with the 2-center Coulomb integrals is stored.
 * @return void.
*/
void SystemGTF_MPI::loadCoulomb2C(const int procMPI_rank, const int procMPI_size, const bool is_for_Dk, const std::string& intName, const std::string& integrals_directory){

    std::string C2_file = is_for_Dk? integrals_directory + "C2Mat_" + intName + ".C2cDk" : integrals_directory + "C2Mat_" + intName + ".C2c";
    uint64_t nvaluesC2c;
    uint32_t norbitals_AUX;
    if(procMPI_rank == 0){
        if(C2_file.empty()){
            throw std::invalid_argument("SystemGTF::loadCoulomb2C: file must not be empty");
        }
        std::ifstream C2_ifstream;
        C2_ifstream.open(C2_file.c_str());
        if(!C2_ifstream.is_open()){
            throw std::invalid_argument("SystemGTF::loadCoulomb2C: file does not exist");
        }
        std::string line;
        std::getline(C2_ifstream, line);
        std::getline(C2_ifstream, line);
        std::getline(C2_ifstream, line);
        std::getline(C2_ifstream, line);
        std::getline(C2_ifstream, line);
        std::istringstream iss0(line);
        iss0 >> nvaluesC2c;
        std::getline(C2_ifstream, line);
        std::istringstream iss1(line);
        iss1 >> norbitals_AUX;

        C2_ifstream.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);
    MPI_Bcast (&nvaluesC2c, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    this->nvaluesC2c_ = nvaluesC2c;
    
    MPI_Bcast (&norbitals_AUX, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    this->norbitals_AUX_ = norbitals_AUX;
    std::vector<double> Coulomb2CValues(nvaluesC2c_);
    std::vector<uint32_t> Coulomb2CIndices_vec(3*nvaluesC2c_);

    if(procMPI_rank == 0){
        std::ifstream C2_ifstream;
        C2_ifstream.open(C2_file.c_str());
        std::string line;
        std::getline(C2_ifstream, line);
        std::getline(C2_ifstream, line);
        std::getline(C2_ifstream, line);
        std::getline(C2_ifstream, line);
        std::getline(C2_ifstream, line);
        std::getline(C2_ifstream, line);
        double value;
        uint32_t muind1, muind2, Rind;
        for(uint64_t i = 0; i < nvaluesC2c_; i++){
            std::getline(C2_ifstream, line);
            std::istringstream iss(line);
            iss >> value >> muind1 >> muind2 >> Rind;
    
            Coulomb2CValues[i]  = value;
            Coulomb2CIndices_vec[i] = muind1;
            Coulomb2CIndices_vec[i + nvaluesC2c_] = muind2;
            Coulomb2CIndices_vec[i + 2*nvaluesC2c_] = Rind;
        }
        C2_ifstream.close();
    }

    MPI_Barrier (MPI_COMM_WORLD);
    MPI_Bcast (&Coulomb2CValues[0], nvaluesC2c_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (&Coulomb2CIndices_vec[0], 3*nvaluesC2c_, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    std::vector<std::array<uint32_t,3>> Coulomb2CIndices(nvaluesC2c_);
    for(uint64_t i = 0; i < nvaluesC2c_; i++){
        Coulomb2CIndices[i] = {Coulomb2CIndices_vec[i], Coulomb2CIndices_vec[i + nvaluesC2c_], Coulomb2CIndices_vec[i + 2*nvaluesC2c_]};
    }
    this->Coulomb2CValues_  = Coulomb2CValues;
    this->Coulomb2CIndices_ = Coulomb2CIndices;

    arma::mat RlistFrac_C2c;
    for(int r = 0; r < procMPI_size; r++){
        if(procMPI_rank == r){
            std::string RC2_file = is_for_Dk? integrals_directory + "RlistFrac_" + intName + ".C2cDk" : integrals_directory + "RlistFrac_" + intName + ".C2c";
            RlistFrac_C2c.load(RC2_file, arma::arma_ascii);
            this->RlistFrac_C2c_ = RlistFrac_C2c;
            this->ncellsC2c_ = RlistFrac_C2c.n_cols;
            if(r == procMPI_size - 1){
                std::cout << "2-center Coulomb matrices loaded from file" << std::endl;
            }
        }
        MPI_Barrier (MPI_COMM_WORLD);
    }

}

/**
 * Method to load the previously computed matrix elements (above a predefined tolerance) and corresponding indices of 2-center Ewald
 * integrals in the AUX basis, and the corresponding list of R-vectors within the supercell.
 * @param is_for_Dk If true, the integrals and lattice vectores will be stored with the extension .E2cDk instead of the usual .E2c 
 * @param intName Name of the file where the 2-center Ewald matrix elements are stored (E2Mat_intName.E2c).
 * @param integrals_directory Directory where the file with the 2-center Ewald integrals is stored.
 * @return void.
*/
void SystemGTF_MPI::loadEwald2C(const int procMPI_rank, const int procMPI_size, const bool is_for_Dk, const std::string& intName, const std::string& integrals_directory){

    std::string E2_file = is_for_Dk? integrals_directory + "E2Mat_" + intName + ".E2cDk" : integrals_directory + "E2Mat_" + intName + ".E2c";
    uint64_t nvaluesC2c;
    uint32_t norbitals_AUX;
    if(procMPI_rank == 0){
        if(E2_file.empty()){
            throw std::invalid_argument("SystemGTF::loadEwald2C: file must not be empty");
        }
        std::ifstream E2_ifstream;
        E2_ifstream.open(E2_file.c_str());
        if(!E2_ifstream.is_open()){
            throw std::invalid_argument("SystemGTF::loadEwald2C: file does not exist");
        }
        std::string line;
        std::getline(E2_ifstream, line);
        std::getline(E2_ifstream, line);
        std::getline(E2_ifstream, line);
        std::getline(E2_ifstream, line);
        std::getline(E2_ifstream, line);
        std::getline(E2_ifstream, line);
        std::istringstream iss0(line);
        iss0 >> nvaluesC2c;
        std::getline(E2_ifstream, line);
        std::istringstream iss1(line);
        iss1 >> norbitals_AUX;
        
        E2_ifstream.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);
    MPI_Bcast (&nvaluesC2c, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    this->nvaluesC2c_ = nvaluesC2c;
    
    MPI_Bcast (&norbitals_AUX, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    this->norbitals_AUX_ = norbitals_AUX;
    std::vector<double> Ewald2CValues(nvaluesC2c_);
    std::vector<uint32_t> Ewald2CIndices_vec(3*nvaluesC2c_);

    if(procMPI_rank == 0){
        std::ifstream E2_ifstream;
        E2_ifstream.open(E2_file.c_str());
        std::string line;
        std::getline(E2_ifstream, line);
        std::getline(E2_ifstream, line);
        std::getline(E2_ifstream, line);
        std::getline(E2_ifstream, line);
        std::getline(E2_ifstream, line);
        std::getline(E2_ifstream, line);
        std::getline(E2_ifstream, line);
        double value;
        uint32_t muind1, muind2, Rind;
        for(uint64_t i = 0; i < nvaluesC2c_; i++){
            std::getline(E2_ifstream, line);
            std::istringstream iss(line);
            iss >> value >> muind1 >> muind2 >> Rind;
    
            Ewald2CValues[i]  = value;
            Ewald2CIndices_vec[i] = muind1;
            Ewald2CIndices_vec[i + nvaluesC2c_] = muind2;
            Ewald2CIndices_vec[i + 2*nvaluesC2c_] = Rind;
        }
        E2_ifstream.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);

    MPI_Bcast (&Ewald2CValues[0], nvaluesC2c_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (&Ewald2CIndices_vec[0], 3*nvaluesC2c_, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    std::vector<std::array<uint32_t,3>> Ewald2CIndices(nvaluesC2c_);
    for(uint64_t i = 0; i < nvaluesC2c_; i++){
        Ewald2CIndices[i] = {Ewald2CIndices_vec[i], Ewald2CIndices_vec[i + nvaluesC2c_], Ewald2CIndices_vec[i + 2*nvaluesC2c_]};
    }
    this->Coulomb2CValues_  = Ewald2CValues;
    this->Coulomb2CIndices_ = Ewald2CIndices;

    arma::mat RlistFrac_E2c;
    for(int r = 0; r < procMPI_size; r++){
        if(procMPI_rank == r){
            std::string RE2_file = is_for_Dk? integrals_directory + "RlistFrac_" + intName + ".E2cDk" : integrals_directory + "RlistFrac_" + intName + ".E2c";
            RlistFrac_E2c.load(RE2_file, arma::arma_ascii);
            this->RlistFrac_C2c_ = RlistFrac_E2c;
            this->ncellsC2c_ = RlistFrac_E2c.n_cols;
            if(r == procMPI_size - 1){
                std::cout << "2-center Ewald matrices loaded from file" << std::endl;
            }
        }
        MPI_Barrier (MPI_COMM_WORLD);
    }


}

/**
 * Method to load the previously computed matrix elements (above a predefined tolerance) and corresponding indices of 2-center metric 
 * integrals in the AUX basis, and the corresponding list of R-vectors.
 * @param metric 0 for the overlap metric, 1 for the attenuated Coulomb metric.
 * @param intName Name of the file where the 2-center metric matrix elements are stored (o2Mat_intName.o2c).
 * @param integrals_directory Directory where the file with the 2-center metric integrals is stored.
 * @return void.
*/
void SystemGTF_MPI::loadMetric2C(const int procMPI_rank, const int procMPI_size, const uint metric, const std::string& intName, const std::string& integrals_directory){

    std::string M2_file, RlistM2_file;
    if(metric == 0){       // Overlap metric
        M2_file = integrals_directory + "o2Mat_" + intName + ".o2c";
        RlistM2_file = integrals_directory + "RlistFrac_" + intName + ".o2c";
    }
    else if(metric == 1){  // Attenuated Coulomb metric
        M2_file = integrals_directory + "att0C2Mat_" + intName + ".att0C2c";
        RlistM2_file = integrals_directory + "RlistFrac_" + intName + ".att0C2c";
    }
    
    uint64_t nvaluesM2c;
    if(procMPI_rank == 0){
        if(M2_file.empty()){
            throw std::invalid_argument("SystemGTF::loadMetric2C: file must not be empty");
        }
        std::ifstream M2_ifstream;
        M2_ifstream.open(M2_file.c_str());
        if(!M2_ifstream.is_open()){
            throw std::invalid_argument("SystemGTF::loadMetric2C: file does not exist");
        }
        std::string line;
        uint32_t norbs_AUX;
        std::getline(M2_ifstream, line);
        std::getline(M2_ifstream, line);
        std::getline(M2_ifstream, line);
        std::getline(M2_ifstream, line);
        std::getline(M2_ifstream, line);
        std::istringstream iss0(line);
        iss0 >> nvaluesM2c;
        std::getline(M2_ifstream, line);
        std::istringstream iss1(line);
        iss1 >> norbs_AUX;
        if(norbs_AUX != norbitals_AUX_){
            throw std::logic_error("Error loading the integral matrix elements: 2-center Coulomb and metric were computed in different AUX basis sets");
        }
        
        M2_ifstream.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);
    MPI_Bcast (&nvaluesM2c, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    this->nvaluesM2c_ = nvaluesM2c;
    
    std::vector<double> metric2CValues(nvaluesM2c_);
    std::vector<uint32_t> metric2CIndices_vec(3*nvaluesM2c_);
    uint32_t RindMax;

    if(procMPI_rank == 0){
        std::ifstream M2_ifstream;
        M2_ifstream.open(M2_file.c_str());
        std::string line;
        std::getline(M2_ifstream, line);
        std::getline(M2_ifstream, line);
        std::getline(M2_ifstream, line);
        std::getline(M2_ifstream, line);
        std::getline(M2_ifstream, line);
        std::getline(M2_ifstream, line);
        double value;
        uint32_t muind1, muind2, Rind;
        for(uint64_t i = 0; i < nvaluesM2c_; i++){
            std::getline(M2_ifstream, line);
            std::istringstream iss(line);
            iss >> value >> muind1 >> muind2 >> Rind;
    
            metric2CValues[i]  = value;
            metric2CIndices_vec[i] = muind1;
            metric2CIndices_vec[i + nvaluesM2c_] = muind2;
            metric2CIndices_vec[i + 2*nvaluesM2c_] = Rind;
        }
        M2_ifstream.close();
        RindMax = *std::max_element(metric2CIndices_vec.begin() + 2*nvaluesM2c_, metric2CIndices_vec.end());
    }
    MPI_Barrier (MPI_COMM_WORLD);

    MPI_Bcast (&RindMax, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast (&metric2CValues[0], nvaluesM2c_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (&metric2CIndices_vec[0], 3*nvaluesM2c_, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    std::vector<std::array<uint32_t,3>> metric2CIndices(nvaluesM2c_);
    for(uint64_t i = 0; i < nvaluesM2c_; i++){
        metric2CIndices[i] = {metric2CIndices_vec[i], metric2CIndices_vec[i + nvaluesM2c_], metric2CIndices_vec[i + 2*nvaluesM2c_]};
    }
    this->metric2CValues_  = metric2CValues;
    this->metric2CIndices_ = metric2CIndices;
    this->ncellsM2c_ = RindMax;

    arma::mat RlistFrac_M2c;
    for(int r = 0; r < procMPI_size; r++){
        if(procMPI_rank == r){
            RlistFrac_M2c.load(RlistM2_file, arma::arma_ascii);
            this->RlistFrac_M2c_ = RlistFrac_M2c.cols(0, RindMax);
            if(r == procMPI_size - 1){
                std::cout << "2-center metric matrices loaded from file. Maximum R index: " << RindMax << std::endl;
            }
        }
        MPI_Barrier (MPI_COMM_WORLD);
    }

}

/**
 * Method to load the previously computed matrix elements (above a predefined tolerance) and corresponding indices of 3-center metric 
 * integrals in the mixed SCF and AUX basis, and the corresponding list of R-vectors.
 * @param intName Name of the file where the 3-center metric matrix elements are stored (o3Mat_intName.o3c).
 * @param integrals_directory Directory where the file with the 3-center metric integrals is stored.
 * @return void.
*/
void SystemGTF_MPI::loadMetric3C(const int procMPI_rank, const int procMPI_size, const uint metric, const std::string& intName, const std::string& integrals_directory){

    std::string M3_file, RlistM3_file;
    if(metric == 0){       // Overlap metric
        M3_file = integrals_directory + "o3Mat_" + intName + ".o3c";
        RlistM3_file = integrals_directory + "RlistFrac_" + intName + ".o3c";
    }
    else if(metric == 1){  // Attenuated Coulomb metric
        M3_file = integrals_directory + "att0C3Mat_" + intName + ".att0C3c";
        RlistM3_file = integrals_directory + "RlistFrac_" + intName + ".att0C3c";
    }
        
    uint64_t nvaluesM3c;
    if(procMPI_rank == 0){
        if(M3_file.empty()){
            throw std::invalid_argument("SystemGTF::loadMetric3C: file must not be empty");
        }
        std::ifstream M3_ifstream;
        M3_ifstream.open(M3_file.c_str());
        if(!M3_ifstream.is_open()){
            throw std::invalid_argument("SystemGTF::loadMetric3C: file does not exist");
        }
        std::string line;
        uint32_t norbs_AUX, norbs_SCF;
        std::getline(M3_ifstream, line);
        std::getline(M3_ifstream, line);
        std::getline(M3_ifstream, line);
        std::getline(M3_ifstream, line);
        std::getline(M3_ifstream, line);
        std::istringstream iss0(line);
        iss0 >> nvaluesM3c;
        std::getline(M3_ifstream, line);
        std::istringstream iss1(line);
        iss1 >> norbs_AUX >> norbs_SCF;
        if(norbs_AUX != norbitals_AUX_){
            throw std::logic_error("Error loading the integral matrix elements: 2-center Coulomb and 3-center metric were computed in different AUX basis sets");
        }
        if(norbs_SCF != norbitals_){
            throw std::logic_error("Error loading the integral matrix elements: 3-center metric were computed in a different SCF basis set than the Hamiltonian from the .outp");
        }
        
        M3_ifstream.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);
    MPI_Bcast (&nvaluesM3c, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    this->nvaluesM3c_ = nvaluesM3c;
    
    std::vector<double> metric3CValues(nvaluesM3c_);
    std::vector<uint32_t> metric3CIndices_vec(5*nvaluesM3c_);
    uint32_t RindMax;

    if(procMPI_rank == 0){
        std::ifstream M3_ifstream;
        M3_ifstream.open(M3_file.c_str());
        std::string line;
        std::getline(M3_ifstream, line);
        std::getline(M3_ifstream, line);
        std::getline(M3_ifstream, line);
        std::getline(M3_ifstream, line);
        std::getline(M3_ifstream, line);
        std::getline(M3_ifstream, line);
        double value;
        uint32_t Pind, muind1, muind2, Rind1, Rind2;
        for(uint64_t i = 0; i < nvaluesM3c_; i++){
            std::getline(M3_ifstream, line);
            std::istringstream iss(line);
            iss >> value >> Pind >> muind1 >> muind2 >> Rind1 >> Rind2;
    
            metric3CValues[i] = value;
            metric3CIndices_vec[i] = Pind;
            metric3CIndices_vec[i + nvaluesM3c_] = muind1;
            metric3CIndices_vec[i + 2*nvaluesM3c_] = muind2;
            metric3CIndices_vec[i + 3*nvaluesM3c_] = Rind1;
            metric3CIndices_vec[i + 4*nvaluesM3c_] = Rind2;
        }
        M3_ifstream.close();
        RindMax = *std::max_element(metric3CIndices_vec.begin() + 3*nvaluesM3c_, metric3CIndices_vec.end());
    }
    MPI_Barrier (MPI_COMM_WORLD);

    MPI_Bcast (&RindMax, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast (&metric3CValues[0], nvaluesM3c_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (&metric3CIndices_vec[0], 5*nvaluesM3c_, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    std::vector<std::array<uint32_t,5>> metric3CIndices(nvaluesM3c_);
    for(uint64_t i = 0; i < nvaluesM3c_; i++){
        metric3CIndices[i] = {metric3CIndices_vec[i], metric3CIndices_vec[i + nvaluesM3c_], metric3CIndices_vec[i + 2*nvaluesM3c_], metric3CIndices_vec[i + 3*nvaluesM3c_], metric3CIndices_vec[i + 4*nvaluesM3c_]};
    }
    this->metric3CValues_  = metric3CValues;
    this->metric3CIndices_ = metric3CIndices;
    this->ncellsM3c_ = RindMax;

    arma::mat RlistFrac_M3c;
    for(int r = 0; r < procMPI_size; r++){
        if(procMPI_rank == r){
            RlistFrac_M3c.load(RlistM3_file, arma::arma_ascii);
            this->RlistFrac_M3c_ = RlistFrac_M3c.cols(0, RindMax);
            if(r == procMPI_size - 1){
                std::cout << "3-center metric matrices loaded from file. Maximum R index: " << RindMax << std::endl;
            }
        }
        MPI_Barrier (MPI_COMM_WORLD);
    }

}

/**
 * Compute the single-particle energies and wave-function coefficients at a given k-point. In the event of magnetism 
 * without SOC, the UP and DOWN energies and coefficients are all returned in eigval and eigvec (which have thus double the
 * dimension of the spinless case). eigval is sorted in ascending order, and the columns of eigvec are sorted accordingly.
 * The phase criterion of "sum of coefficients be real" is imposed, see Journal of Chemical Theory and Computation, 19, 9416, 2023.
 * @details Canonical orthogonalization is employed to circumvent the generalized eigenvalue problem (arma::eig_pair is avoided).
 * @param k k-point k-point in fractional coordinates where the eigenvalues and eigenvectors are computed.
 * @param eigval Real column vector that will store the energies. Dimension with magnetism (no SOC): (2*norbitals)
 * @param eigvec Complex matrix that will store the eigenvectors by columns. Dimension with magnetism (no SOC): (norbitals,2*norbitals)
 * @return void.
*/
void SystemGTF_MPI::solveBands(const arma::colvec& k, arma::colvec& eigval, arma::cx_mat& eigvec) {

    arma::colvec kAU = Gbasis_*k/ANG2AU;
    if(!MAGNETIC_FLAG_ || SOC_FLAG_){
        arma::cx_mat Hk = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);
        arma::cx_mat Sk = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);
        for(int i = 0; i < ncells; i++){
            std::complex<double> exp_ikR = std::exp(imag*arma::dot(kAU, RlistAU_.col(i)));
            Hk += exp_ikR * ((*ptr_hamiltonianMatrices).slice(i));
            Sk += exp_ikR * ((*ptr_overlapMatrices).slice(i));
        }
        Hk.diag() *= 0.5;
        Hk += Hk.t();
        Sk.diag() *= 0.5;
        Sk += Sk.t();

        arma::vec eigval_S;
        arma::cx_mat eigvec_S;
        arma::eig_sym(eigval_S, eigvec_S, Sk);
        eigval_S = 1./arma::sqrt(arma::abs(eigval_S));
        arma::cx_mat X = eigvec_S*arma::diagmat(eigval_S);                  //canonical orthogonalization (Szabo - eq 3.169)
        Hk = X.t()*Hk*X;
        // arma::cx_mat X = eigvec_S*arma::diagmat(eigval_S)*eigvec_S.t();  //symmetric orthogonalization (Szabo - eq 3.167)
        // Hk = X*Hk*X;   

        Hk = 0.5*(Hk + Hk.t());

        arma::eig_sym(eigval, eigvec, Hk);
        eigvec = X*eigvec;
        // arma::uvec ind_E {arma::sort_index(eigval)};   //the eigenvalues are already sorted in ascending order with eig_sym
        // eigval = eigval(ind_E);                        //sort the eigenvalues in ascending order
        // eigvec = eigvec.cols(ind_E);                   //sort the eigenvectors accordingly
    } 

    else { // CRYSTALconfig.MAGNETIC_FLAG && !CRYSTALconfig.SOC_FLAG (magnetism without SOC)
        arma::colvec eigvalUP, eigvalDOWN;
        arma::cx_mat eigvecUP, eigvecDOWN;
        arma::cx_mat HkUP = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);
        arma::cx_mat HkDOWN = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);
        arma::cx_mat Sk = arma::zeros<arma::cx_mat>(norbitals_, norbitals_);
        std::complex<double> exp_ikR;
        for(int i = 0; i < ncells; i++){
            exp_ikR = std::exp(imag*arma::dot(kAU, RlistAU_.col(i)));
            HkUP   += exp_ikR * ((*ptr_alphaMatrices).slice(i));
            HkDOWN += exp_ikR * ((*ptr_betaMatrices).slice(i));
            Sk     += exp_ikR * ((*ptr_overlapMatrices).slice(i));
        }
        HkUP.diag() *= 0.5;
        HkUP += HkUP.t();
        HkDOWN.diag() *= 0.5;
        HkDOWN += HkDOWN.t();
        Sk.diag() *= 0.5;
        Sk += Sk.t();

        arma::vec eigval_S;
        arma::cx_mat eigvec_S;
        arma::eig_sym(eigval_S, eigvec_S, Sk);
        eigval_S = 1./arma::sqrt(arma::abs(eigval_S));
        arma::cx_mat X = eigvec_S*arma::diagmat(eigval_S);                  //canonical orthogonalization (Szabo - eq 3.169)
        HkUP = X.t()*HkUP*X;
        HkDOWN = X.t()*HkDOWN*X;
        // arma::cx_mat X = eigvec_S*arma::diagmat(eigval_S)*eigvec_S.t();  //symmetric orthogonalization (Szabo - eq 3.167)
        // HkUP = X*HkUP*X;   
        // HkDOWN = X*HkDOWN*X;    

        HkUP   = 0.5*(HkUP + HkUP.t());
        HkDOWN = 0.5*(HkDOWN + HkDOWN.t());                                   

        arma::eig_sym(eigvalUP, eigvecUP, HkUP);
        eigvecUP = X*eigvecUP;
        arma::eig_sym(eigvalDOWN, eigvecDOWN, HkDOWN);
        eigvecDOWN = X*eigvecDOWN;
        eigval = arma::join_vert(eigvalUP, eigvalDOWN);  //join the UP and DOWN energies vertically
        eigvec = arma::join_horiz(eigvecUP, eigvecDOWN); //join the UP and DOWN eigenvectors horizontally
        arma::uvec ind_E {arma::sort_index(eigval)};     //the eigenvalues are already sorted in ascending order with eig_sym, but the union must be sorted
        eigval = eigval(ind_E);                          //sort the eigenvalues in ascending order
        eigvec = eigvec.cols(ind_E);                     //sort the eigenvectors accordingly
    }

    //Make the sum of coefficients real for each eigenvector
    eigvec %= arma::repmat( arma::exp( -imag*arma::arg( arma::sum(eigvec) ) ), norbitals_, 1); 

}

/**
 * Group the entries of the (k,k') matrix (with k,k' = k_0,..,k_nkBSE) with the same k'' = abs(k-k') (= k_i, for some i=0,..,0.5*(nkBSE+2^sum_{d=1}^{ndim}\iseven(nki[d])) ). 
 * The returned array has 3 indices: the first one spans the k'' = abs(k-k'), the second spans the different (k,k') pairs yielding the corresponding k'', 
 * and the third has 3 components and determines the k and k' (respectively) indices in the (k,k') matrix, and whether k-k'=k_i (value 0) or -k_i (value 1) noting
 * that k_i is determined by being the first among (k_i,-k_i) in the BSE grid. Only the values for the first index contained in 
 * selected_chunks are returned (this is so that each MPI process gets only its corresponding chunks).   
 * @param nkiBSE Vector with the number of k-points in each Gi direction for the BSE. Read from the exciton file.
 * @param selected_chunks Vector with the indices of the abs(k-k')-chunks that will be returned.
 * @return std::vector<std::vector<std::array<uint32_t,2>>> The kentries_chunks list with 3 indices described above.
 */
std::vector<std::vector<std::array<uint32_t,3>>> SystemGTF_MPI::generatekentries(const std::vector<int32_t>& nkiBSE, const arma::ucolvec& selected_chunks){

    double tol = 0.00001; // threshold tolerance to determine equalities between k vector (double) components
    arma::mat combs = (generateCombinations(nkiBSE, false)).t();
    arma::colvec shrink_vec(ndim);
    for(int d = 0; d < ndim; d++){
        shrink_vec(d) = (double)nkiBSE[d];
    }
    combs /= arma::repmat(shrink_vec,1,nkBSE_);

    // Determine the index of k = 0 within the list of combinations
    arma::colvec ki_frac0 = combs.col(0);
    arma::colvec kdiff0 = ki_frac0 - combs.col(0);
    kdiff0 -= arma::floor(kdiff0);
    uint32_t kfind0 = 0;
    arma::colvec kfind_vec0 = combs.col(kfind0);
    while(!arma::approx_equal(kdiff0, kfind_vec0, "absdiff", tol)){
        kfind0++;
        kfind_vec0 = combs.col(kfind0);
    }
    
    // Generate list with the opposite of each k-point, stored in kOppList
    std::vector<uint32_t> kOppList(nkBSE_);
    arma::colvec ki_frac = combs.col(kfind0);  
    for(uint32_t j = 0; j < nkBSE_; j++){
        arma::colvec kdiff = ki_frac - combs.col(j);
        kdiff -= arma::floor(kdiff);

        uint32_t kfind = 0;
        arma::colvec kfind_vec = combs.col(kfind);
        while(!arma::approx_equal(kdiff, kfind_vec, "absdiff", tol)){
            kfind++;
            kfind_vec = combs.col(kfind);
        }
        kOppList[j] = kfind;
    }
    
    // Generate list with the index of a single k-point in the (k,-k) pair, for each of these existing pairs
    std::map<uint32_t,uint32_t> kindSingle;
    uint32_t countr = 0;
    for(uint32_t kind = 0; kind < nkBSE_; kind++){
        if(kOppList[kind] >= kind){
            kindSingle[kind] = countr;
            countr++;
        }
    }
    uint32_t nkSingles = countr;
    
    // Determine the index of each k-k' (only lower triangle in {k,k'} matrix) and assign it to the corresponding first index of kentries_chunks
    std::vector<std::vector<std::array<uint32_t,3>>> kentries_chunks(nkSingles);
    for(uint32_t i = 0; i < nkSingles; i++){
        kentries_chunks[i].reserve(nkBSE_);
    }
    for(uint32_t i = 0; i < nkBSE_; i++){
        arma::colvec kdiff, kfind_vec, kfind_vecOpp;
        arma::colvec ki_frac = combs.col(i);
        for(uint32_t j = 0; j <= i; j++){
            kdiff = ki_frac - combs.col(j);
            kdiff -= arma::floor(kdiff);

            uint32_t kfind = 0;
            kfind_vec    = combs.col(kfind);
            kfind_vecOpp = combs.col(kOppList[kfind]);
            bool is_k_plus  = arma::approx_equal(kdiff, kfind_vec,    "absdiff", tol);
            bool is_k_minus = arma::approx_equal(kdiff, kfind_vecOpp, "absdiff", tol);
            while(!is_k_plus && !is_k_minus){   
                kfind++;
                kfind_vec    = combs.col(kfind);
                kfind_vecOpp = combs.col(kOppList[kfind]);
                is_k_plus    = arma::approx_equal(kdiff, kfind_vec,    "absdiff", tol);
                is_k_minus   = arma::approx_equal(kdiff, kfind_vecOpp, "absdiff", tol);
            }
            uint32_t k_plus_or_minus = is_k_plus? 0 : 1;
            kentries_chunks[kindSingle.at(kfind)].push_back({i,j, k_plus_or_minus});
        }
    }

    // Select chunks of kentries_chunks (i.e. the values of its first index) which will be returned by the present method
    uint32_t Nselected = selected_chunks.n_elem;
    std::vector<std::vector<std::array<uint32_t,3>>> kentries_chunks_selected(Nselected);
    for(uint32_t n = 0; n < Nselected; n++){
        kentries_chunks_selected[n] = kentries_chunks[selected_chunks(n)];
    }

    return kentries_chunks_selected;

}


}