#include "xatu/System.hpp"

namespace xatu {

/**
 * Default constructor.
 * @details The default constructor throws an error as the class must always be initialized from either a ConfigurationSystem 
 * or a ConfigurationCRYSTAL object, or with a copy constructor.
 */
System::System(){
	
    throw std::invalid_argument("System must be called with either a ConfigurationSystem, a ConfigurationCRYSTAL or another System object");

}

/**
 * Constructor from ConfigurationSystem, to be used in the TB mode only. 
 * @details The Hamiltonian pointer is not initialized here but in the child classes.
 * @param SystemConfig ConfigurationSystem object obtained from any configuration file.
 */
System::System(const ConfigurationSystem& SystemConfig) : Lattice{SystemConfig}{

	this->filling_	                = SystemConfig.filling;
	this->highestValenceBand_	    = SystemConfig.filling - 1;
	this->ncells_                   = SystemConfig.ncells;
	this->norbitals_   			    = SystemConfig.hamiltonianMatrices.n_cols;
	this->ptr_overlapMatrices       = &SystemConfig.overlapMatrices;
	
}

/**
 * System name setter.
 * @param systemName Name of the system.
 * @return void.
*/
void System::setSystemName(const std::string& systemName){
	
	this->systemName = systemName;

}

/**
 * Compute the grid of k-points that will be used as a basis for the exciton states, and store it in the corresponding attribute.
 * @param nki Vector with the desired number of k-points per direction. Only the first ndim components are considered.
 * @param fractionalCoords True (false) to return the k-points in fractional coordinates (Angstrom^-1, respectively).
 * @return arma::mat Grid of k-points, stored by columns.
*/
arma::mat System::generateBSEgrid(const std::vector<int32_t>& nki, const bool fractionalCoords){

	std::vector<int32_t> nki_ndim(ndim);
	for(int n = 0; n < ndim; n++){
		nki_ndim[n] = nki[n];
	}
	arma::mat kpointsBSE = gridMonkhorstPack(nki_ndim, true, fractionalCoords);
	return kpointsBSE;

}

/**
 * Method to write to a file the energy bands evaluated on a set of kpoints specified on a file. This method is just a general envelope, 
 * and the core diagonalization method must be defined in the relevant derived class, depending on the mode (TB or Gaussian).
 * @details Each k-point must occupy a row in the file, with no blank lines. The format for each k-point is: kx ky kz ,
 * whose values are in Angstrom. It creates a file with the name "kpointsfile.bands" where the bands are stored.
 * @param kpointsfile Name of the file with the kpoints where we want to obtain the bands.
 * @return void.
*/
void System::printBands(const std::string& kpointsfile) {
	std::ifstream inputfile;
	std::string line;
	double kx, ky, kz;
	arma::vec eigval;
	arma::cx_mat eigvec;
	std::string outputfilename = kpointsfile + ".bands";
	FILE* bandfile = fopen(outputfilename.c_str(), "w");
	try{
		inputfile.open(kpointsfile.c_str());
		while(std::getline(inputfile, line)){
			std::istringstream iss(line);
			iss >> kx >> ky >> kz;
			arma::rowvec kpoint{kx, ky, kz};
			solveBands(kpoint, eigval, eigvec);
			for (uint i = 0; i < eigval.n_elem; i++){
				fprintf(bandfile, "%12.6f\t", eigval(i));
			}
			fprintf(bandfile, "\n");
		}
	}
	catch(const std::exception& e){
		std::cerr << e.what() << std::endl;
	}
	inputfile.close();
	fclose(bandfile);
	arma::cout << "Done" << arma::endl;
}


}