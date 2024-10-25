#include <mpi.h>
#include "xatu.hpp"

int main(int argc, char* argv[]){

    int procMPI_rank, procMPI_size;

    MPI_Init (&argc,&argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &procMPI_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &procMPI_size);
    if(procMPI_rank == 0){
        xatu::printHeaderGTFprovisional();
        std::cout << std::endl;
        xatu::printParallelizationMPI(procMPI_size);
        std::cout << std::endl;
        std::cout << "+---------------------------------------------------------------------------+" << std::endl;
        std::cout << "|                          EWALD/COULOMB INTEGRALS                          |" << std::endl;
        std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    }

    // INPUT PARAMETERS /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string outp_file = "InputFiles/Pho_PBE0_1D_0f.outp";
    int ncells = 43;                                 //Number of H(R) and S(R) matrices taken into account from the .outp file
    std::string bases_file = "InputFiles/Bases_custom.txt";
    std::string savefile = "custom";        //Result will be saved in file Results/1-Integrals/E2Mat_savefile.E2c
    bool is_for_Dk = false;                           //True for scalei_supercell corresponding to Deltak. If true, the integrals and lattice vectores will be stored with the extension .E2cDk instead of the usual .E2c 
    std::vector<int32_t> scalei_supercell = {88,66};  //Scaling factor for the corresponding original (unit cell) Bravais basis vectors Ri to form the supercell
    int nR = 60;                                      //Minimum number of external supercell lattice vectors to be included in the Ewald direct lattice sum
    int nG = 300;                                     //Minimum number of reciprocal supercell vectors to be included in the Ewald reciprocal lattice sum
    int tol = 8;                                      //Threshold tolerance for the integrals: only entries > 10^-tol are stored
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    xatu::ConfigurationCRYSTAL_MPI CRYSTALconfig(outp_file, procMPI_rank, procMPI_size, ncells, true);
    xatu::ConfigurationGTF_MPI GTFconfig(procMPI_rank, procMPI_size, CRYSTALconfig.nspecies, CRYSTALconfig.atomic_number_ordering, bases_file);
    xatu::IntegralsBase IntBase(CRYSTALconfig, GTFconfig);    
    
    xatu::IntegralsEwald2C_MPI Ewald2C(IntBase, procMPI_rank, procMPI_size, tol, scalei_supercell, nR, nG, is_for_Dk, savefile);

    MPI_Finalize();
    return 0;

}
