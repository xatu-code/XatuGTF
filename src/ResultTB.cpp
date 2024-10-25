#ifndef RESULTTB_CPP
#define RESULTTB_CPP

#include "xatu/ResultTB.hpp"

namespace xatu {

/* -------------------- Observables -------------------- */

ResultTB::ResultTB(ExcitonTB* exciton_, arma::vec& eigval_, arma::cx_mat& eigvec_) : 
    Result<SystemTB>( (Exciton<SystemTB> *)exciton_, eigval_, eigvec_){};

/** 
 * Routine to compute the expected Sz spin value of the electron
 * and hole that form a given exciton->
 * @param coefs Coefficients of the exciton state. Note that the coefficients must be given
 * in the exact ordering used in the exciton basis. Otherwise, wrong results will be obtained.
 * @return Vector with the total spin of the exciton, the spin of the hole and that of the electron
 */
arma::cx_vec ResultTB::spinX(const arma::cx_vec& coefs){
    
    // Initialize Sz for both electron and hole to zero
    arma::cx_double electronSpin = 0;
    arma::cx_double holeSpin = 0;
    double totalSpin = 0;
    uint32_t dimX = exciton->basisStates.n_rows;

    arma::cx_vec spinEigvalues = {1./2, -1./2};
    arma::cx_vec spinVector = arma::zeros<arma::cx_vec>(system->norbitals);
    int vecIterator = 0;
    for(int atomIndex = 0; atomIndex < system->natoms; atomIndex++){
        int species = system->motif.col(atomIndex)(3);
        int norb = system->orbitalsPerSpecies(species);
        spinVector.subvec(vecIterator, vecIterator + norb - 1) = 
                          arma::kron(arma::ones(norb/2), spinEigvalues);
        vecIterator += system->orbitalsPerSpecies(species);
    }
    
	arma::cx_vec eigvec, spinEigvec;

    // Initialize hole spin and electron spin operators
    int nvbands = exciton->valenceBands.n_elem;
    int ncbands = exciton->conductionBands.n_elem;
    int npairs = nvbands*ncbands;

    arma::cx_mat spinHole = arma::zeros<arma::cx_mat>(dimX, dimX);
    arma::cx_mat spinElectron = arma::zeros<arma::cx_mat>(dimX, dimX);

    arma::cx_mat vMatrix = arma::eye<arma::cx_mat>(nvbands, nvbands);
    arma::cx_mat cMatrix = arma::eye<arma::cx_mat>(ncbands, ncbands);

    // Initialize list of pairs of valence-conduction bands
    arma::mat bandPairs = arma::zeros(npairs, 2);
    int i = 0;
    for(double v : exciton->valenceBands){
        for(double c : exciton->conductionBands){
            bandPairs.row(i) = arma::rowvec{v, c};
            i++;
        }
    }

    for(uint32_t k = 0; k < system->kpointsBSE.n_cols; k++){
        arma::cx_mat spinHoleReduced = arma::zeros<arma::cx_mat>(nvbands, nvbands);
        arma::cx_mat spinElectronReduced = arma::zeros<arma::cx_mat>(ncbands, ncbands);
        for(int i = 0; i < nvbands; i++){
            int vIndex = exciton_->bandToIndex[exciton->valenceBands(i)];
            for(int j = 0; j < nvbands; j++){
                int vIndex2 = exciton_->bandToIndex[exciton->valenceBands(j)];
                eigvec = exciton->eigvecKStack.slice(k).col(vIndex);
                spinEigvec = eigvec % spinVector;
                eigvec = exciton->eigvecKStack.slice(k).col(vIndex2);
                spinHoleReduced(i,j) = arma::cdot(eigvec, spinEigvec);
            }
        }
        for(int i = 0; i < ncbands; i++){
            int cIndex = exciton_->bandToIndex[exciton->conductionBands(i)];
            for(int j = 0; j < ncbands; j++){
                int cIndex2 = exciton_->bandToIndex[exciton->conductionBands(j)];
                eigvec = exciton->eigvecKQStack.slice(k).col(cIndex2);
                spinEigvec = eigvec % spinVector;
                eigvec = exciton->eigvecKQStack.slice(k).col(cIndex);
                spinElectronReduced(i,j) = arma::cdot(eigvec, spinEigvec);
            }
        }
                
        spinHole.submat(k*npairs, k*npairs, (k+1)*npairs - 1, (k+1)*npairs - 1) = arma::kron(cMatrix, spinHoleReduced);
        spinElectron.submat(k*npairs, k*npairs, (k+1)*npairs - 1, (k+1)*npairs - 1) = arma::kron(spinElectronReduced, vMatrix);
    }

    // Perform tensor products with the remaining quantum numbers
    holeSpin = -arma::cdot(coefs, spinHole*coefs);
    electronSpin = arma::cdot(coefs, spinElectron*coefs);
    totalSpin = real((holeSpin + electronSpin));
    
    arma::cx_vec results = {totalSpin, holeSpin, electronSpin};
    return results;
}


/**
 * Method to compute the velocity of an exciton eigenstate.
 * @details Computes the expectation value of the velocity operator, 
 * using the eigenvector of the exciton state, assuming an underlying tight-binding basis.
 * @param stateindex Index of exciton eigenstate.
 * @return Outputs a matrix where the first column is the center-of-mass velocity of the exciton,
 * and the second is the relative velocity of the exciton->
 */
arma::mat ResultTB::velocity(int index){

    arma::cx_mat velocity = arma::zeros<arma::cx_mat>(3, 2);

    arma::cx_vec total_e_velocity = arma::zeros<arma::cx_vec>(3);
    arma::cx_vec total_h_velocity = arma::zeros<arma::cx_vec>(3);

    #pragma omp parallel for collapse(3)
    for (int64_t n = 0; n < system->nkBSE; n++){
    for (int j = 0; j < exciton->conductionBands.n_elem; j++){
    for (int i = 0; i < exciton->valenceBands.n_elem; i++){

        arma::cx_vec local_e_velocity = arma::zeros<arma::cx_vec>(3);
        arma::cx_vec local_h_velocity = arma::zeros<arma::cx_vec>(3);

        int v = exciton->valenceBands(i);
        int c = exciton->conductionBands(j);
        int eigvecIndex = n*exciton->valenceBands.n_elem * exciton->conductionBands.n_elem + j*exciton->valenceBands.n_elem + i;

        std::complex<double> coef = eigvec.col(index)(eigvecIndex);

        for (int jp = 0; jp < exciton->conductionBands.n_elem; jp++){
            int cp = exciton->conductionBands(jp);
            arma::cx_vec velocitySP = velocitySingleParticle(cp, c, n, "conduction");

            int eigvecIndexP = n*exciton->valenceBands.n_elem * exciton->conductionBands.n_elem + jp*exciton->valenceBands.n_elem + i;
            local_e_velocity += velocitySP * coef * std::conj(eigvec.col(index)(eigvecIndexP));
        }

        for (int ip = 0; ip < exciton->valenceBands.n_elem; ip++){
            int vp = exciton->valenceBands(ip);
            arma::cx_vec velocitySP = velocitySingleParticle(v, vp, n, "valence");

            int eigvecIndexP = n*exciton->valenceBands.n_elem * exciton->conductionBands.n_elem + j*exciton->valenceBands.n_elem + ip;
            local_h_velocity += velocitySP * coef * std::conj(eigvec.col(index)(eigvecIndexP));
        }

        #pragma omp critical
        {
            total_e_velocity += local_e_velocity;
            total_h_velocity += local_h_velocity;
        }
    }}}

    arma::cout << "Total e velocity: " << total_e_velocity << arma::endl;
    arma::cout << "Total h velocity: " << total_h_velocity << arma::endl;

    velocity.col(0) = total_e_velocity + total_h_velocity;
    velocity.col(1) = total_e_velocity - total_h_velocity;

    return arma::real(velocity);
}

/*
 * Method to compute the velocity of a single particle.
 * @details Computes the matrix elements of the velocity operator in the
 * single particle basis.
 * @param fIndex First index.
 * @param sIndex Second index.
 * @param kIndex Index of the kpoint.
 * @param bandType Type of band, either 'valence' or 'conduction'.
 */
arma::cx_vec ResultTB::velocitySingleParticle(int fIndex, int sIndex, int kIndex, std::string bandType){

    if (bandType != "valence" && bandType != "conduction"){
        throw std::invalid_argument("bandType must be either 'valence' or 'conduction'");
    }

    arma::cx_cube hkDerivative = arma::zeros<arma::cx_cube>(system->norbitals, system->norbitals, 3);
    arma::cx_cube iHt = arma::zeros<arma::cx_cube>(system->norbitals, system->norbitals, 3);

    arma::colvec k = system->kpointsBSE.col(kIndex);

    arma::colvec Q = arma::zeros<arma::colvec>(3);
    if (bandType == "conduction"){
        Q = exciton->Q;
    }

    // First compute Hk derivative
    for (int j = 0; j < 3; j++){
        for (int i = 0; i < system->ncells; i++){
            arma::colvec cell = system->Rlist.col(i);
            hkDerivative.slice(j) += (*(system->ptr_hamiltonianMatrices)).slice(i) * 
                                     std::exp(imag*arma::dot(k + Q, cell)) * cell(j) * imag;
	    };
    }

    // Next compute iH(t-t') matrix
    arma::cx_cube motifDifference = arma::zeros<arma::cx_cube>(system->norbitals, system->norbitals, 3);
    arma::cx_mat extendedMotif = arma::zeros<arma::cx_mat>(system->norbitals, 3);
    int currentIndex = 0;
    for (int i = 0; i < system->natoms; i++){
        int norb = system->orbitalsPerSpecies(system->motif.col(i)(3));
        extendedMotif.rows(currentIndex, currentIndex + norb - 1) = arma::kron((system->motif.col(i).subvec(0, 2)).t(),
                                                                         arma::ones<arma::cx_vec>(norb));
        currentIndex += norb;
    }

    arma::cx_mat blochHamiltonian = system->hamiltonian(k + Q);
    for (int j = 0; j < 3; j++){
        motifDifference.slice(j) = arma::kron(extendedMotif.col(j), arma::ones<arma::cx_rowvec>(system->norbitals)) -
                                   arma::kron(extendedMotif.col(j).t(), arma::ones<arma::cx_vec>(system->norbitals));
        iHt.slice(j) = imag * blochHamiltonian % motifDifference.slice(j).t();
    }

    // Finally compute velocity matrix elements
    arma::cx_vec velocityMatrixElement = arma::zeros<arma::cx_vec>(3);
    arma::cx_vec fState, sState;
    int n = exciton_->bandToIndex[fIndex];
    int m = exciton_->bandToIndex[sIndex];

    if (bandType == "valence"){
        fState = exciton->eigvecKStack.slice(kIndex).col(n);
        sState = exciton->eigvecKStack.slice(kIndex).col(m);
    }
    else{
        fState = exciton->eigvecKQStack.slice(kIndex).col(n);
        sState = exciton->eigvecKQStack.slice(kIndex).col(m);
    }
    for (int j = 0; j < 3; j++){
        velocityMatrixElement(j) = arma::cdot(fState, (hkDerivative.slice(j) + iHt.slice(j)) * sState);
    }

    return velocityMatrixElement;
}

/**
 * Method to compute the oscillator strength of the exciton->
 * @details Computes the oscillator strength of the exciton, which is a measure of the brightness
 * of the exciton, and it is used to obtain the optical conducitivity.
 * @return Matrix with all the oscillator strenghts in all three directions for all excitons.
 */
arma::cx_mat ResultTB::excitonOscillatorStrength(){

    int nR = system->Rlist.n_cols;
    int norb = system->norbitals;
    int norb_ex = exciton->dimBSE;
    int filling = (system->highestValenceBand) + 1;
    int nv = exciton->valenceBands.n_elem;
    int nc = exciton->conductionBands.n_elem;

    arma::mat Rvec = (system->Rlist).t();
    // Extend bravais lattice to 3x3 matrix
    arma::mat R = arma::zeros(3, 3);
    for (uint i = 0; i < system->Rbasis.n_cols; i++){
        R.col(i) = system->Rbasis.col(i);
    }
    R = R.t();

    arma::mat extendedMotif = arma::zeros(system->norbitals, 3);
    int it = 0;
    for(int i = 0; i < system->natoms; i++){
        arma::rowvec atom = (system->motif.col(i).subvec(0, 2)).t();
        int species = system->motif.col(i)(3);
        for(uint j = 0; j < system->orbitalsPerSpecies(species); j++){
            extendedMotif.row(it) = atom; 
            it++;
        }
    }
    arma::cx_cube hhop = *(system->ptr_hamiltonianMatrices);
    arma::cube shop(arma::size(hhop));
    if ((*(system->ptr_overlapMatrices)).empty()){
        for (uint i = 0; i < hhop.n_slices; i++){
            shop.slice(i) = arma::eye(size(hhop.slice(i)));
        }
    }
    else{
        shop = arma::real(*(system->ptr_overlapMatrices));
    }
    int nk = system->nkBSE;
    arma::rowvec rkx = system->kpointsBSE.row(0);
    arma::rowvec rky = system->kpointsBSE.row(1);
    arma::rowvec rkz = system->kpointsBSE.row(2);

    arma::mat eigval_sp = exciton->eigvalKStack;
    arma::cx_cube eigvec_sp = exciton->eigvecKStack;

    arma::cx_cube vme_ex = arma::zeros<arma::cx_cube>(3, norb_ex, 2);

    std::complex<double>* vme = new std::complex<double>[3*nk*(nv + nc)*(nv + nc)];
    bool convert_to_au = true;

    exciton_oscillator_strength_(&nR, &norb, &norb_ex, &nv, &nc, &filling, 
             Rvec.memptr(), R.memptr(), extendedMotif.memptr(), hhop.memptr(), shop.memptr(), &nk, rkx.memptr(),
             rky.memptr(), rkz.memptr(), m_eigvec.memptr(), m_eigval.memptr(), eigval_sp.memptr(), eigvec_sp.memptr(),
             vme, vme_ex.memptr(), &convert_to_au);

    return vme_ex.slice(0);
}

/**
 * Method to compute the real-space amplitude of an exciton state (not necessarily an eigenstate).
 * @details Used by writeRealSpaceAmplitude to write the probability density over several unit cells.
 * @param BSEcoefs State whose real-space amplitude we want to obtain.
 * @param electronIndex Index of the atom where we put the electron.
 * @param holeIndex Index of atom where we put the hole.
 * @param eCell Unit cell of the electron.
 * @param hCell Unit cell of the hole.
 * @return Real-space amplitude evaluated at those electron and hole positions.
 */
double ResultTB::realSpaceWavefunction(const arma::cx_vec& BSEcoefs, int electronIndex, int holeIndex,
                             const arma::colvec& eCell, const arma::colvec& hCell){

    std::complex<double> imag(0, 1);
    double totalAmplitude = 0;
    arma::cx_vec eigvec = arma::cx_vec(BSEcoefs);
    int eOrbitals = system->orbitalsPerSpecies(system->motif.col(electronIndex)(3));
    int hOrbitals = system->orbitalsPerSpecies(system->motif.col(holeIndex)(3));

    // Compute index corresponding to electron and hole
    int eIndex = 0;
    int hIndex = 0;
    for(int i = 0; i < electronIndex; i++){
        eIndex += system->orbitalsPerSpecies(system->motif.col(i)(3));
    }
    for(int i = 0; i < holeIndex; i++){
        hIndex += system->orbitalsPerSpecies(system->motif.col(i)(3));
    }
    eigvec = addExponential(eigvec, eCell - hCell);

    for(int alpha = 0; alpha < eOrbitals; alpha++){
        for(int beta = 0; beta < hOrbitals; beta++){

        arma::cx_cube c = exciton->eigvecKQStack.tube(eIndex + alpha, exciton->valenceBands.n_elem, 
                                eIndex + alpha, exciton->valenceBands.n_elem + exciton->conductionBands.n_elem - 1);
        arma::cx_rowvec cFlat = arma::reshape(c, 1, c.n_elem, 1);
        arma::cx_rowvec cExtended = arma::kron(cFlat, arma::ones<arma::cx_rowvec>(exciton->valenceBands.n_elem));

        arma::cx_cube v = exciton->eigvecKStack.tube(hIndex + beta, 0, 
                                hIndex + beta, exciton->valenceBands.n_elem - 1);

        arma::cx_rowvec vFlat = arma::reshape(v, 1, v.n_elem, 1);
        arma::cx_rowvec vExtended = arma::zeros<arma::cx_rowvec>(vFlat.n_elem*exciton->conductionBands.n_elem);
        int blockSize = exciton->conductionBands.n_elem * exciton->valenceBands.n_elem;
        for(uint32_t i = 0; i < system->nkBSE; i++){
            vExtended.subvec(i*blockSize, (i + 1)*blockSize - 1) = arma::kron(arma::ones<arma::cx_rowvec>(exciton->conductionBands.n_elem), 
                                                        vFlat.subvec(i*exciton->valenceBands.n_elem, (i + 1)*exciton->valenceBands.n_elem - 1));
        }
        
        arma::cx_rowvec coefs = cExtended % arma::conj(vExtended);

        totalAmplitude += std::norm(arma::dot(coefs, eigvec));    
        }
    }

    return totalAmplitude;
};

/* -------------------- Output -------------------- */

/**
 * Writes the probability density of finding the electron at a given position, having
 * fixed the position of the hole.
 * @param statecoefs State whose probability density we want to determine.
 * @param holeIndex Index of atom of the motif where we fix the hole.
 * @param holeCell Unit cell where we fix the hole.
 * @param textfile File to write the amplitudes.
 * @param ncells Number of unit cells where we compute the amplitudes.
 * @return void
 */
void ResultTB::writeRealspaceAmplitude(const arma::cx_vec& statecoefs, int holeIndex,
                                     const arma::colvec& holeCell, FILE* textfile, int ncells){

    arma::colvec holePosition = system->motif.col(holeIndex).subvec(0, 2) + holeCell;
    fprintf(textfile, "%11.8lf\t%11.8lf\t%14.11lf\n", holePosition(0), holePosition(1), 0.0);

    double radius = arma::norm(system->Rbasis.col(0)) * ncells;
    arma::mat cellCombinations = system->truncateSupercell(exciton->nki[0], radius);
    arma::vec coefs = arma::zeros(cellCombinations.n_cols*system->motif.n_cols);

    // Compute probabilities
    #pragma omp parallel for
    for(uint32_t cellIndex = 0; cellIndex < cellCombinations.n_cols; cellIndex++){
        arma::colvec cell = cellCombinations.col(cellIndex);
        for (unsigned int atomIndex = 0; atomIndex < system->motif.n_cols; atomIndex++){
            int idx = atomIndex + cellIndex*system->motif.n_cols;
            //coefs(it) = atomCoefficientSquared(atomIndex, cell, holeCell, RScoefs);
            coefs(idx) = realSpaceWavefunction(statecoefs, atomIndex, holeIndex, cell, holeCell);
        }
    }

    // Write probabilities to file
    for(uint32_t cellIndex = 0; cellIndex < cellCombinations.n_cols; cellIndex++){
        arma::colvec cell = cellCombinations.col(cellIndex);
        for(unsigned int atomIndex = 0; atomIndex < system->motif.n_cols; atomIndex++){
            int idx = atomIndex + cellIndex*system->motif.n_cols;
            arma::colvec position = system->motif.col(atomIndex).subvec(0, 2) + cell;
            fprintf(textfile, "%11.8lf\t%11.8lf\t%14.11lf\n",
                            position(0), position(1), coefs(idx));
        }
    }
    fprintf(textfile, "#\n");                              
}

/**
 * Method to compute and write the absorption spectra to a file.
 * @details This method computes both the single particle absorption, and the absorption
 * from the exciton spectrum. All the required parameters must be specified in a separate text file
 * named kubo_w.in
 * @return void 
 */
void ResultTB::writeAbsorptionSpectrum(){

    int nR = system->Rlist.n_cols;
    int norb = system->norbitals;
    int norb_ex = exciton->dimBSE;
    int filling = (system->highestValenceBand) + 1;
    int nv = exciton->valenceBands.n_elem;
    int nc = exciton->conductionBands.n_elem;

    arma::mat Rvec = (system->Rlist).t();
    // Extend bravais lattice to 3x3 matrix
    arma::mat R = arma::zeros(3, 3);
    for (uint i = 0; i < system->Rbasis.n_cols; i++){
        R.col(i) = system->Rbasis.col(i);
    }
    R = R.t();

    arma::mat extendedMotif = arma::zeros(system->norbitals, 3);
    int it = 0;
    for(int i = 0; i < system->natoms; i++){
        arma::rowvec atom = (system->motif.col(i).subvec(0, 2)).t();
        int species = system->motif.col(i)(3);
        for(uint j = 0; j < system->orbitalsPerSpecies(species); j++){
            extendedMotif.row(it) = atom; 
            it++;
        }
    }
    arma::cx_cube hhop = *(system->ptr_hamiltonianMatrices);
    arma::cube shop(arma::size(hhop));
    if ((*(system->ptr_overlapMatrices)).empty()){
        for (uint i = 0; i < hhop.n_slices; i++){
            shop.slice(i) = arma::eye(size(hhop.slice(i)));
        }
    }
    else{
        shop = arma::real(*(system->ptr_overlapMatrices));
    }
    int nk = system->nkBSE;
    arma::rowvec rkx = system->kpointsBSE.row(0);
    arma::rowvec rky = system->kpointsBSE.row(1);
    arma::rowvec rkz = system->kpointsBSE.row(2);

    arma::mat eigval_sp = exciton->eigvalKStack;
    arma::cx_cube eigvec_sp = exciton->eigvecKStack;

    skubo_w_(&nR, &norb, &norb_ex, &nv, &nc, &filling, 
             Rvec.memptr(), R.memptr(), extendedMotif.memptr(), hhop.memptr(), shop.memptr(), &nk, rkx.memptr(),
             rky.memptr(), rkz.memptr(), m_eigvec.memptr(), m_eigval.memptr(), eigval_sp.memptr(), eigvec_sp.memptr());
}

/**
 * Method to add exponentials to some vector of coefficients.
 * @details Used in realSpaceWavefunction to compute the real-space exciton amplitudes.
 * Basically multiplies each coefficient by an exponential with phase ikR.
 * @param coefs Vector of electron-hole pair coefficients.
 * @param cell Unit cell used in the exponential.
 * @return Coefficients with the added exponential.
 */
arma::cx_vec ResultTB::addExponential(arma::cx_vec& coefs, const arma::colvec& cell){

    arma::vec product = system->kpointsBSE.t() * cell;
    arma::cx_vec exponentials = arma::exp(imag*product);
    int nBandCombinations = exciton->valenceBands.n_elem*exciton->conductionBands.n_elem;
    exponentials = arma::kron(exponentials, arma::ones<arma::cx_vec>(nBandCombinations));

    coefs = coefs % exponentials;

    return coefs;
}

}

#endif