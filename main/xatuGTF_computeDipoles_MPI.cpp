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
        std::cout << "|                          DIPOLE (X,Y,Z) INTEGRALS                         |" << std::endl;
        std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    }

    // INPUT PARAMETERS /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string outp_file = "InputFiles/Pho_PBE0_1D_0f.outp";
    int ncells = 43;                                 //Number of H(R) and S(R) matrices taken into account from the .outp file
    std::string bases_file = "InputFiles/Bases_custom.txt";
    std::string savefile = "custom";        //Result will be saved in file Results/1-Integrals/dipoleMat_savefile.dip
    int nR = 120;                                     //Minimum number of direct lattice vectors for which the dipole integrals will be computed
    int tol = 8;                                      //Threshold tolerance for the integrals: only entries > 10^-tol are stored
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    xatu::ConfigurationCRYSTAL_MPI CRYSTALconfig(outp_file, procMPI_rank, procMPI_size, ncells, true);
    xatu::ConfigurationGTF_MPI GTFconfig(procMPI_rank, procMPI_size, CRYSTALconfig.nspecies, CRYSTALconfig.atomic_number_ordering, bases_file);
    xatu::IntegralsBase IntBase(CRYSTALconfig, GTFconfig); 

    xatu::IntegralsDipole_MPI Dipole(IntBase, procMPI_rank, procMPI_size, tol, nR, savefile); 

    MPI_Finalize();
    return 0;

}
