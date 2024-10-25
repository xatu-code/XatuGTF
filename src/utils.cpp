#include "xatu/utils.hpp"

namespace xatu {

void writeVectorToFile(arma::vec vector, FILE* file){
	for (unsigned int i = 0; i < vector.n_elem; i++){
		fprintf(file, "%f\t", vector(i));
	}
	fprintf(file, "\n");
}

void writeVectorToFile(arma::rowvec vector, FILE* file){
	for (unsigned int i = 0; i < vector.n_elem; i++){
		fprintf(file, "%f\t", vector(i));
	}
	fprintf(file, "\n");
}

void writeVectorsToFile(const arma::mat& vectors, FILE* textfile, std::string mode){
	if (mode == "row"){
		for(unsigned int i = 0; i < vectors.n_rows; i++){
			writeVectorToFile((arma::rowvec)vectors.row(i), textfile);
		}
	}
	else if (mode == "col"){
		for(unsigned int i = 0; i < vectors.n_cols; i++){
			writeVectorToFile((arma::vec)vectors.col(i), textfile);
		}
	}
	else{
		std::cout << "Error: writeVectorsToFile: mode not recognized" << std::endl;
	}
}

arma::vec readVectorFromFile(std::string filename){
    std::ifstream file(filename.c_str());
    std::string line;
    std::vector<double> vector;
    double value;
    while(std::getline(file, line)){
        std::istringstream iss(line);
        while(iss >> value){
            vector.push_back(value);
        };
    };

    arma::vec coefs(vector);
    return coefs;
}

/* Definition of non-interacting retarded Green function */
std::complex<double> rGreenF(double energy, double delta, double eigEn){

	std::complex<double> i(0,1);
	return 1./((energy + i*delta) - eigEn);
}

/* Routine to calcule the density of states at a given energy,
associated to a given set of eigenvalues (e.g. bulk or edge).
NB: It is NOT normalized. */
double densityOfStates(double energy, double delta, const arma::mat& energies){

		double dos = 0;
		for(int i = 0; i < (int)energies.n_rows; i++){
			for(int j = 0; j < (int)energies.n_cols; j++){
				double eigEn = energies(i,j);
				dos += -PI*std::imag(rGreenF(energy, delta, eigEn));
			};
		};
        // Divide by number of k's and length a (currently a is missing)
		dos /= energies.n_cols; 

		return dos;
}

/* Routine to calculate and write the density of states associated 
to a given set of eigenenergies. Density of states is normalized to 1
(integral of DOS over energies equals 1)
Input: mat energies (eigenvalues), double delta (convergence parameter),
FILE* dosfile output file
Output: void (write results to output file) */
void writeDensityOfStates(const arma::mat& energies, double delta, FILE* dosfile){

	double minE = energies.min();
	double maxE = energies.max();
	int nE = 2000; // Number of points in energy mesh

	arma::vec energyMesh = arma::linspace(minE - 0.5, maxE + 0.5, nE);

	// Main loop
	double totalDos = 0;
	double dos = 0;
	// First loop over energies to normalize
	for(int n = 0; n < (int)energyMesh.n_elem; n++){
		double energy = energyMesh(n);
		double deltaE = energyMesh(1) - energyMesh(0);

		dos = densityOfStates(energy, delta, energies);
		totalDos += dos*deltaE;
	};
	// Main loop to write normalized DOS
	for(int n = 0; n < (int)energyMesh.n_elem; n++){
		double dos = 0;
		double energy = energyMesh(n);

		dos = densityOfStates(energy, delta, energies);
		dos = dos/totalDos; // Normallise
		fprintf(dosfile, "%lf\t%lf\n", energy, dos);
	};
	return;
}

/* Intended to be used within printEnergies, not in an standalone way. Computes degeneracy of each 
up to a given precision with cost O(n) */
std::vector<std::vector<double>> detectDegeneracies(const arma::vec& eigval, int64_t n, int precision){
	
    if(n < 0){
        throw std::invalid_argument("detectDegeneracies: n must be a positive integer");
    }
    else if(n > static_cast<int64_t>(eigval.n_elem)){
        throw std::invalid_argument("detectDegeneracies: n must be lower than total number of eigenstates");
    }

    std::vector<std::vector<double>> pairs;
    std::vector<double> pair;
    double previusEnergy = eigval(0);
    int degeneracy = 1;
    double threshold = pow(10, -precision);
    double energy;
    for(int64_t i = 1; i < n; i++){
        energy = eigval(i);
        if(std::abs(energy - previusEnergy) < threshold){
            degeneracy++;
        }
        else{
            pair = std::vector<double>{previusEnergy, (double)degeneracy};
            pairs.push_back(pair);
            degeneracy = 1;
        }
        previusEnergy = energy;
    }
    pair = std::vector<double>{previusEnergy, (double)degeneracy};
    pairs.push_back(pair);

    return pairs;
}

/**
 * Prints the header of the original code with the credits.
 */
void printHeader(){
    std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    std::cout << "|                                                                           |" << std::endl;
    std::cout << "|                                     Xatu                                  |" << std::endl;
    std::cout << "|                              v1.3.1 - 28/03/2024                          |" << std::endl;
    std::cout << "|                    https://github.com/alejandrojuria/xatu                 |" << std::endl;
    std::cout << "|                                                                           |" << std::endl;
    std::cout << "|                                  [Authors]                                |" << std::endl;
    std::cout << "| A. J. Uria-Alvarez, J. J. Esteve-Paredes, M. A. Garcia-Blazquez,          |" << std::endl;
    std::cout << "| J. J. Palacios                                                            |" << std::endl;
    std::cout << "| Universidad Autonoma de Madrid, Spain                                     |" << std::endl;
    std::cout << "+---------------------------------------------------------------------------+" << std::endl;
}

/**
 * Prints the provisional header of the GTF implementation.
 */
void printHeaderGTFprovisional(){
    std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    std::cout << "|                                                                           |" << std::endl;
    std::cout << "|                                    Xatu                                   |" << std::endl;
    std::cout << "|                  Ab-initio Gaussian version - 25/10/2024                  |" << std::endl;
    std::cout << "|                    https://github.com/xatu-code/XatuGTF                   |" << std::endl;
    std::cout << "|                                                                           |" << std::endl;
    std::cout << "|                                  [Authors]                                |" << std::endl;
    std::cout << "| M. A. Garcia-Blazquez, J. J. Palacios                                     |" << std::endl;
    std::cout << "| Universidad Autonoma de Madrid, Spain                                     |" << std::endl;
    std::cout << "+---------------------------------------------------------------------------+" << std::endl;
}

/**
 * Prints the basic parallelization settings.
 */
void printParallelizationMPI(const int procMPI_size){
	std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    std::cout << "|                              Parallelization                              |" << std::endl;
    std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    std::cout << "Number of MPI processes: " << procMPI_size << std::endl;
    std::cout << "Number of OpenMP threads: " << omp_get_max_threads() << std::endl;
}

/**
 * Auxiliary routine used to check if a matrix is triangular (either upper or lower)
 * @details Used in System when constructed from SystemConfiguration objects to determine
 * how to build H(k) and S(k).
 * @param matrix Complex matrix.
 * @return bool True (false) if the matrix is triangular (not triangular, or also diagonal; respectively). 
*/
bool checkIfTriangular(const arma::cx_mat& matrix){

	return (matrix.is_trimatu() != matrix.is_trimatl());

}


}