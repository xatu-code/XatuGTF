#include "xatu/Lattice.hpp"

namespace xatu {

/**
 * Default constructor to initialize Lattice attributes from a ConfigurationSystem object.
 * @param SystemConfig ConfigurationSystem object. 
 */
Lattice::Lattice(const ConfigurationSystem& SystemConfig){

    this->ndim_   = SystemConfig.ndim;	
	this->Rbasis_ = SystemConfig.Rbasis;
	computeUnitCellVolume();
    calculateGbasis();

}

/**
 * Rearrange vector ni, coming from the exciton input (flexible), assigning a value for each spatial component, duplicating values if necessary
 * @param ni Vector with the number of points in each direction (to be applied to Ri or Gi). If some component is missing, the first one is 
 * employed for them. Then, the first ndim components are sequentially assigned to n1,n2,n3.
 * @param n1,n2,n3 Contain the resulting values from ni, only the first ndim are nonzero.
 * @return void.
*/
void Lattice::unify_ni(const std::vector<int32_t>& ni, int32_t& n1, int32_t& n2, int32_t& n3){

    n1 = ni[0];
    if(n1 <= 0){
        throw std::invalid_argument("ERROR unify_ni: number of points per periodic direction must be positive");
    }
    n2 = 0;
    n3 = 0;
    if(ndim >= 2){
        if(ni.size() >= 2 ){
            n2 = ni[1];
        } else { 
            n2 = n1;
        }
        if(n2 <= 0){
            throw std::invalid_argument("ERROR unify_ni: number of points per periodic direction must be positive");
        }
        if(ndim == 3){
            if(ni.size() == 3){
                n3 = ni[2];
            } else {
                n3 = n1;
            }
            if(n3 <= 0){
                throw std::invalid_argument("ERROR unify_ni: number of points per periodic direction must be positive");
            }
        }
    }
    
}

/**
 * Method to generate a kronecker-like list of integer combinations, to be used with direct or reciprocal lattice vectors.
 * Each row contains the ndim coefficients of a different point. Matrix dimension: (n_{1}*..*n_{ndim}, ndim)
 * @param ni Vector with the number of points in each direction (Ri or Gi). Flexible input: only the first ndim components are taken 
 * into account, and if some component is missing the first one is employed for them
 * @param centered If true, the combinations are centered at zero. If false, all combinations have positive coefficients.
 * @return arma::mat List of cell combinations.
*/
arma::mat Lattice::generateCombinations(const std::vector<int32_t>& ni, const bool centered){

    int32_t n1 = 0;
    int32_t n2 = 0;
    int32_t n3 = 0;
    unify_ni(ni,n1,n2,n3);
    arma::icolvec nvec = {1,n1,n2,n3};
    nvec = nvec.subvec(0, ndim);
    nvec.insert_rows(ndim + 1, 1);
    nvec(ndim + 1) = 1;   // nvec = {1,n_{1},..,n_{ndim},1}

    int64_t ncombinations = arma::prod(nvec);
    arma::mat combinations(ncombinations, ndim);

    for(int d = 0; d < ndim; d++){
        int dshift = centered ? (int)nvec(d+1)/2 : 0;
        arma::colvec dvalues = arma::regspace<arma::colvec>(0, nvec(d+1) - 1) - dshift;
        arma::colvec com_aux = arma::repelem( dvalues, arma::prod( nvec.subvec(0,d) ), 1 ); 
        combinations.col(d)  = arma::repmat( com_aux , arma::prod( nvec.subvec(d+2, nvec.n_elem - 1) ), 1 );
    }

    return combinations;

}

/* --------------------------- Reciprocal Lattice methods --------------------------- */

/**
 * Compute the reciprocal lattice vectors {G_1,..,G_ndim} and return them by columns in arma::mat (3,ndim). 
 * The units are preserved from those of the input Bravais vectors, which are by default in Angstrom.
 * @return void.
 */
void Lattice::calculateGbasis(){
    
    arma::mat Gbasis = arma::zeros<arma::mat>(3,ndim);
    arma::colvec R1 = Rbasis_.col(0);
    if(ndim == 1){
        Gbasis(0,0) = TWOPI/R1(0);
    } 
    else if(ndim == 2){
        arma::colvec R2 = Rbasis_.col(1);
        arma::mat Rot2D = {{0,-1,0},{1,0,0},{0,0,1}};
        arma::mat RotFac = (TWOPI/arma::dot(R1,Rot2D*R2))*Rot2D;
        Gbasis.col(0) = RotFac*R2;
        Gbasis.col(1) = -RotFac*R1;
    }
    else if(ndim == 3){
        arma::colvec R2 = Rbasis_.col(1);
        arma::colvec R3 = Rbasis_.col(2);
        double volFac = TWOPI/std::abs(arma::det(Rbasis_));
        Gbasis.col(0) = volFac*arma::cross(R2,R3);
        Gbasis.col(1) = volFac*arma::cross(R3,R1);
        Gbasis.col(2) = volFac*arma::cross(R1,R2);
    }
    else {
        throw std::invalid_argument("ERROR calculateGbasis: lattice dimensionality is not 1, 2 or 3.");
    }

    this->Gbasis_ = Gbasis;

}

/**
 * Compute a Monkhorst-Pack grid in the interval [0 G1)x...x[0 Gn_dim), and return the k-points by columns. The dimensions of the matrix
 * are (3,nk) if fractionalCoords = false, and (ndim,nk) if fractionalCoords = true. 
 * @param nki Vector with the number of points in each Gi direction. Only the first ndim components are taken into account.
 * @param containsGamma True (false) for a grid containing Gamma (displaced by half the corresponding step in each Gi, respectively).
 * @param fractionalCoords True (false) to return the k-points in fractional coordinates (Angstrom^-1, respectively).
 * @return arma::mat List of k-points in Angstrom^-1 or atomic units, stored by columns.
 */
arma::mat Lattice::gridMonkhorstPack(const std::vector<int32_t>& nki, const bool containsGamma, const bool fractionalCoords){

    arma::mat combs = ( generateCombinations(nki, false) ).t();
    arma::colvec shrink_vec(ndim);
    for(int d = 0; d < ndim; d++){
        shrink_vec(d) = (double)nki[d];
    }

    uint32_t nk = combs.n_cols;
    if(!containsGamma){
        combs += 0.5; 
    }
    combs /= arma::repmat(shrink_vec,1,nk);

    arma::mat Klist = fractionalCoords? combs : Gbasis*combs;
    return Klist;

}

/**
* Method to create the matrix of the first nG (at least) 3-component reciprocal lattice vectors, stored by columns and ordered by ascending norm.
* @details The number of returned vectors is at least nG because full stars are given. The units are preserved from those of the input 
* reciprocal basis vectors, which are by default in Angstrom^-1. It is used for the reciprocal lattice term in the Ewald potential.
* @param nG Minimum number of reciprocal lattice vectors that will be listed.
* @param combs List of fractional coordinates, centered at 0. It is an output of this method.
* @return arma::mat (3, nG' >= nG) matrix with the aforementioned reciprocal lattice vectors by columns.
*/
arma::mat Lattice::generateGlist(const uint32_t nG, arma::mat& combs, const int procMPI_rank){

    arma::rowvec norms_Gi = arma::sqrt(arma::sum(Gbasis_ % Gbasis_, 0));
    // Automatic correction accounting for possibly large differences of norms in the reciprocal vectors
    uint normRatio = std::ceil(0.5*arma::max(norms_Gi) / arma::min(norms_Gi)); 
    // Conservative estimate to make sure that none of the first n vectors is left out
    int32_t GindmaxAux = std::ceil(3*normRatio*std::pow(nG,1./(double)ndim));
	GindmaxAux += 1 - (GindmaxAux % 2);

	combs = ( generateCombinations({GindmaxAux}, true) ).t();
    arma::mat generated_Glist = Gbasis*combs;
    arma::rowvec generated_norms = arma::sqrt( arma::sum(generated_Glist % generated_Glist ,0) );

    arma::urowvec indices = (arma::sort_index(generated_norms)).t();
    generated_norms = arma::sort(generated_norms);
    generated_Glist = generated_Glist.cols(indices); // Order the lattice vectors (columns of generated_Glist) according to the norms 
    combs = combs.cols(indices);
    
    double requested_norm = generated_norms(nG - 1);
    uint32_t countr = nG;
    double current_norm = generated_norms(countr);
    if(procMPI_rank == 0){
        std::cout << "Requested reciprocal lattice vectors maximum norm: " << requested_norm << " Angstrom^-1" << std::endl;
    }

    while(current_norm - requested_norm < 1e-3){ // Complete the current star of direct lattice vectors
        countr++;
        current_norm = generated_norms(countr);
    }
    combs = combs.cols(0, countr - 1);
    return generated_Glist.cols(0, countr - 1);

}

/**
* Method to create the matrix of the first nG (at least) 3-component reciprocal vectors in a supercell defined by scale factors in scalei.
* @details The number of returned vectors is at least nG because full stars are given. The units are preserved from those of the input 
* reciprocal vectors, which are by default in Angstrom^-1. It is used in the reciprocal lattice term of the Ewald potential.
* @param nG Minimum number of supercell reciprocal vectors that will be listed.
* @param scalei Vector where each component is the scaling factor for the corresponding original (unit cell) reciprocal basis vectors Gi.
* @return arma::mat (3, nG' >= nG) matrix with the aforementioned supercell reciprocal vectors by columns.
*/
arma::mat Lattice::generateGlist_supercell(const uint32_t nG, const std::vector<int32_t>& scalei, const int procMPI_rank){

    int32_t n1 = 0;
    int32_t n2 = 0;
    int32_t n3 = 0;
    unify_ni(scalei,n1,n2,n3);
    arma::mat Gbasis_supercell = Gbasis;
    Gbasis_supercell.col(0) /= n1;
    if(ndim >= 2){
        Gbasis_supercell.col(1) /= n2;
        if(ndim == 3){
            Gbasis_supercell.col(2) /= n3;
        }
    }
    arma::rowvec norms_Gi = arma::sqrt(arma::sum(Gbasis_supercell % Gbasis_supercell, 0));
    // Automatic correction accounting for possibly large differences of norms in the supercell reciprocal vectors
    uint normRatio = std::ceil(0.5*arma::max(norms_Gi) / arma::min(norms_Gi)); 
    // Conservative estimate to make sure that none of the first n vectors is left out
    int32_t GindmaxAux = std::ceil(3*normRatio*std::pow(nG,1./(double)ndim));
	GindmaxAux += 1 - (GindmaxAux % 2);

	arma::mat combs = ( generateCombinations({GindmaxAux}, true) ).t();
    arma::mat generated_Glist = Gbasis_supercell*combs;
    arma::rowvec generated_norms = arma::sqrt( arma::sum(generated_Glist % generated_Glist ,0) );

    arma::urowvec indices = (arma::sort_index(generated_norms)).t();
    generated_norms = arma::sort(generated_norms);
    generated_Glist = generated_Glist.cols(indices); // Order the reciprocal vectors (columns of generated_Glist) according to the norms 
    
    double requested_norm = generated_norms(nG - 1);
    uint32_t countr = nG;
    double current_norm = generated_norms(countr);
    if(procMPI_rank == 0){
        std::cout << "Requested supercell reciprocal vectors maximum norm: " << requested_norm << " Angstrom^-1" << std::endl;
    }

    while(current_norm - requested_norm < 1e-3){ // Complete the current star of supercell reciprocal vectors
        countr++;
        current_norm = generated_norms(countr);
    }
    
    return generated_Glist.cols(0, countr - 1);

}

/**
* Analogous to generateGlist_supercell, but only one of each (G_{n},-G_{n}) pair is included, and G = 0 is excluded. 
* @details nG applies to this truncated list of vectors (not to the original without truncation).
* @param nG Minimum number of supercell reciprocal vectors that will be listed.
* @param scalei Vector where each component is the scaling factor for the corresponding original (unit cell) reciprocal basis vectors Gi.
* @return arma::mat (3, nG' >= nG) matrix with the aforementioned supercell reciprocal vectors by columns.
*/
arma::mat Lattice::generateGlist_supercell_half(const uint32_t nG, const std::vector<int32_t>& scalei, const int procMPI_rank){

    #pragma omp declare reduction (merge_uint64 : std::vector<uint64_t> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) 

    int32_t n1 = 0;
    int32_t n2 = 0;
    int32_t n3 = 0;
    unify_ni(scalei,n1,n2,n3);
    arma::mat Gbasis_supercell = Gbasis;
    Gbasis_supercell.col(0) /= n1;
    if(ndim >= 2){
        Gbasis_supercell.col(1) /= n2;
        if(ndim == 3){
            Gbasis_supercell.col(2) /= n3;
        }
    }
    arma::rowvec norms_Gi = arma::sqrt(arma::sum(Gbasis_supercell % Gbasis_supercell, 0));
    // Automatic correction accounting for possibly large differences of norms in the supercell reciprocal vectors
    uint normRatio = std::ceil(0.5*arma::max(norms_Gi) / arma::min(norms_Gi)); 
    // Conservative estimate to make sure that none of the first n vectors is left out (higher usual due to the restriction on selected vectors)
    int32_t GindmaxAux = std::ceil(5*normRatio*std::pow(nG,1./(double)ndim));
	GindmaxAux += 1 - (GindmaxAux % 2);

	arma::mat combs = ( generateCombinations({GindmaxAux}, true) ).t();
    // Remove G = 0 and opposite G_{n} 
    std::map<uint32_t,uint32_t> combs_opp = generateRlistOpposite(combs);
    std::vector<uint64_t> inds_to_remove;
    inds_to_remove.reserve(combs.n_cols);
    #pragma omp parallel for reduction(merge_uint64: inds_to_remove)  
    for(uint64_t ncombinations = 0; ncombinations < combs.n_cols; ncombinations++){
        if(combs_opp.at(ncombinations) <= ncombinations){
            inds_to_remove.push_back(ncombinations);
        }
    }
    arma::ucolvec inds_to_remove_arma = arma::conv_to<arma::ucolvec>::from(inds_to_remove);
    combs.shed_cols(inds_to_remove_arma);

    // Proceed as usual 
    arma::mat generated_Glist = Gbasis_supercell*combs;
    arma::rowvec generated_norms = arma::sqrt( arma::sum(generated_Glist % generated_Glist ,0) );

    arma::urowvec indices = (arma::sort_index(generated_norms)).t();
    generated_norms = arma::sort(generated_norms);
    generated_Glist = generated_Glist.cols(indices); // Order the reciprocal vectors (columns of generated_Glist) according to the norms 
    
    double requested_norm = generated_norms(nG - 1);
    uint32_t countr = nG;
    double current_norm = generated_norms(countr);
    if(procMPI_rank == 0){
        std::cout << "Requested supercell reciprocal vectors maximum norm: " << requested_norm << " Angstrom^-1" << std::endl;
    }

    while(current_norm - requested_norm < 1e-3){ // Complete the current star of supercell reciprocal vectors
        countr++;
        current_norm = generated_norms(countr);
    }
    
    return generated_Glist.cols(0, countr - 1);

}

/* --------------------------- Direct Lattice methods --------------------------- */

/**
 * Method to compute the unit cell volume, area or length (depending on the lattice dimensionality).
 * The units are preserved from those of the input Bravais vectors, which are by default in Angstrom.
 * @return void
 */
void Lattice::computeUnitCellVolume(){

	arma::mat Rbasis_red = Rbasis_.submat(0, 0, ndim - 1, ndim - 1);
	this->unitCellVolume_ = std::abs( arma::det( Rbasis_red ) );

}


/**
* Method to create the matrix of the first nR (at least) 3-component Bravais vectors, stored by columns and ordered by ascending norm.
* @details The number of returned vectors is at least nR because full stars are given. The units are preserved from those of the input 
* Bravais vectors, which are by default in Angstrom. It basically substitutes Rlist for the integrals when more R-vectors are requested 
* than contained in the .outp.
* @param nR Minimum number of direct lattice vectors that will be listed.
* @param combs List of fractional coordinates, centered at 0. It is an output of this method.
* @param IntegralType String which will be printed along with norm(nR) and which indicates the type of integrals for which the 
* list is generated.
* @return arma::mat (3, nR' >= nR) matrix with the aforementioned Bravais vectors by columns.
*/
arma::mat Lattice::generateRlist(const uint32_t nR, arma::mat& combs, const std::string& IntegralType, const int procMPI_rank){

    arma::rowvec norms_Ri = arma::sqrt(arma::sum(Rbasis_ % Rbasis_, 0));
    // Automatic correction accounting for possibly large differences of norms in the lattice vectors
    uint normRatio = std::ceil(0.5*arma::max(norms_Ri) / arma::min(norms_Ri)); 
    // Conservative estimate to make sure that none of the first n vectors is left out
    int32_t RindmaxAux = std::ceil(3*normRatio*std::pow(nR,1./(double)ndim));
	RindmaxAux += 1 - (RindmaxAux % 2);

	combs = ( generateCombinations({RindmaxAux}, true) ).t();
    arma::mat generated_Rlist = Rbasis*combs;
    arma::rowvec generated_norms = arma::sqrt( arma::sum(generated_Rlist % generated_Rlist ,0) );

    arma::urowvec indices = (arma::sort_index(generated_norms)).t();
    generated_norms = arma::sort(generated_norms);
    generated_Rlist = generated_Rlist.cols(indices); // Order the lattice vectors (columns of generated_Rlist) according to the norms 
    combs = combs.cols(indices);
    
    double requested_norm = generated_norms(nR - 1);
    uint32_t countr = nR;
    double current_norm = generated_norms(countr);
    if(procMPI_rank == 0){
        std::cout << "Requested direct lattice vectors maximum norm: " << requested_norm << " Angstrom (" + IntegralType + ")" << std::endl;
    }

    while(current_norm - requested_norm < 1e-3){ // Complete the current star of direct lattice vectors
        countr++;
        current_norm = generated_norms(countr);
    }
    combs = combs.cols(0, countr - 1);
    return generated_Rlist.cols(0, countr - 1);

}

/**
* Method to create the matrix of the first nR (at least) 3-component Bravais vectors in a supercell defined by scale factors in scalei.
* @details The number of returned vectors is at least nR because full stars are given. The units are preserved from those of the input 
* Bravais vectors, which are by default in Angstrom. It is used in the direct lattice term of the Ewald potential.
* @param nR Minimum number of supercell lattice vectors that will be listed.
* @param scalei Vector where each component is the scaling factor for the corresponding original (unit cell) Bravais basis vectors Ri.
* @return arma::mat (3, nR' >= nR) matrix with the aforementioned supercell lattice vectors by columns.
*/
arma::mat Lattice::generateRlist_supercell(const uint32_t nR, const std::vector<int32_t>& scalei, const int procMPI_rank){

    int32_t n1 = 0;
    int32_t n2 = 0;
    int32_t n3 = 0;
    unify_ni(scalei,n1,n2,n3);
    arma::mat Rbasis_supercell = Rbasis;
    Rbasis_supercell.col(0) *= n1;
    if(ndim >= 2){
        Rbasis_supercell.col(1) *= n2;
        if(ndim == 3){
            Rbasis_supercell.col(2) *= n3;
        }
    }
    arma::rowvec norms_Ri = arma::sqrt(arma::sum(Rbasis_supercell % Rbasis_supercell, 0));
    // Automatic correction accounting for possibly large differences of norms in the supercell lattice vectors
    uint normRatio = std::ceil(0.5*arma::max(norms_Ri) / arma::min(norms_Ri)); 
    // Conservative estimate to make sure that none of the first n vectors is left out
    int32_t RindmaxAux = std::ceil(3*normRatio*std::pow(nR,1./(double)ndim));
    RindmaxAux += 1 - (RindmaxAux % 2);

    arma::mat combs = ( generateCombinations({RindmaxAux}, true) ).t();
    arma::mat generated_Rlist = Rbasis_supercell*combs;
    arma::rowvec generated_norms = arma::sqrt( arma::sum(generated_Rlist % generated_Rlist ,0) );

    arma::urowvec indices = (arma::sort_index(generated_norms)).t();
    generated_norms = arma::sort(generated_norms);
    generated_Rlist = generated_Rlist.cols(indices); // Order the lattice vectors (columns of generated_Rlist) according to the norms 
    
    double requested_norm = generated_norms(nR - 1);
    uint32_t countr = nR;
    double current_norm = generated_norms(countr);
    if(procMPI_rank == 0){
        std::cout << "Requested supercell lattice vectors maximum norm: " << requested_norm << " Angstrom" << std::endl;
    }

    while(current_norm - requested_norm < 1e-3){ // Complete the current star of supercell lattice vectors
        countr++;
        current_norm = generated_norms(countr);
    }
    
    return generated_Rlist.cols(0, countr - 1);

}

/**
* Method to create a list of Bravais vectors, stored by columns, whose fractional coordinates ({R1,R2,R3} basis) are spanned by an 
* input vector of maximum values. The fractional coordinates of the generated vectors are centered at zero. The units are preserved 
* from those of the input Bravais vectors, which are by default in Angstrom. It is used in the 2-center Coulomb integrals.
* @param nRi Vector with the number of fractional coordinates along each Ri. Only the first ndim components are taken into account.
* @param combs List of fractional coordinates, centered at 0. It is an output of this method.
* @param IntegralType String which will be printed along with norm(nR) and which indicates the type of integrals for which the 
* list is generated.
* @return arma::mat (3, nRi_1*..*nRi_ndim) matrix with the aforementioned Bravais vectors by columns.
*/
arma::mat Lattice::generateRlist_fixed(const std::vector<int32_t>& nRi, arma::mat& combs, const std::string& IntegralType, const int procMPI_rank){

    combs = ( generateCombinations(nRi, true) ).t();
    arma::mat Rlist_fixed = Rbasis*combs;
    arma::rowvec generated_norms = arma::sqrt( arma::sum(Rlist_fixed % Rlist_fixed ,0) );
    if(procMPI_rank == 0){
        std::cout << "Requested direct lattice vectors maximum norm: " << arma::max(generated_norms) << " Angstrom (" + IntegralType + ")" << std::endl;
    }

    return Rlist_fixed;

}

/**
 * Returns a map where each entry is the index of the vector in the input generated_Rlist (stores vector by columns)
 * opposite to the vector whose index is the corresponding map's key. 
 * @param generated_Rlist List of real vectors stored by columns. 
 * @return std::map The list whose n-th entry is -R_{n}, where R_{n} = Rlist(n).
 */
std::map<uint32_t,uint32_t> Lattice::generateRlistOpposite(const arma::mat& generated_Rlist){

    uint32_t nRlist = generated_Rlist.n_cols; 
    std::map<uint32_t,uint32_t> RlistOpposites;
    for(uint32_t RindOpp = 0; RindOpp < nRlist; RindOpp++){
        int countr = 0;
        for(uint32_t Rind = 0; Rind < nRlist; Rind++){
            arma::colvec Rsum = generated_Rlist.col(Rind) + generated_Rlist.col(RindOpp);
            if( Rsum.is_zero(0.001) ){
                RlistOpposites[RindOpp] = Rind;
                countr++;
            }
        }
        if(countr == 0){
            throw std::invalid_argument("ERROR generateRlistOpposite: unable to find opposite vector");
        }

    }
    return RlistOpposites;

}


}

