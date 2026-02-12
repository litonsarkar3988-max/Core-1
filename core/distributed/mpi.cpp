#include <mpi.h>
#include <iostream>
#include <cstdlib>

/*
========================================
 TITANCORE MPI CONTROLLER
 Multi-node cluster bootstrap
========================================
*/

struct TitanMPI {

    int world_size;
    int rank;
    std::string host;

    TitanMPI(int argc, char** argv) {
        MPI_Init(&argc, &argv);

        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        char hostname[MPI_MAX_PROCESSOR_NAME];
        int len;
        MPI_Get_processor_name(hostname, &len);
        host = hostname;

        if (rank == 0) {
            std::cout << "\n[TITANCORE MPI]" << std::endl;
            std::cout << "Nodes online: " << world_size << std::endl;
        }

        std::cout << "[Rank " << rank << "] running on " << host << std::endl;
    }

    bool is_master() {
        return rank == 0;
    }

    void barrier() {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void shutdown() {
        if (rank == 0)
            std::cout << "\n[MPI] Cluster shutdown.\n";

        MPI_Finalize();
    }
};
