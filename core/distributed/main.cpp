#include <iostream>
#include <vector>
#include "mpi_manager.cpp" //
#include "../../distributed/nccl.cpp"
#include "../../distributed/fsdp.cpp"
#include "../model/gpt.cpp"
#include "../dataloader/dataset.cpp"

/*
====================================================
  TITANCORE: MAIN BOOTSTRAPPER & ENGINE
====================================================
  Initializes Cluster via MPI and Communication 
  via NCCL for 1T+ Parameter Training.
====================================================
*/

int main(int argc, char** argv) {

    // 1. Initialize MPI for Cluster Management
    TitanMPI mpi(argc, argv);
    int rank = mpi.get_rank();
    int world_size = mpi.get_world_size();

    if (mpi.is_master()) {
        std::cout << "[TitanCore] Booting cluster with " << world_size << " nodes...\n";
    }

    mpi.barrier();

    // 2. Initialize NCCL
    // 
    
    // Master Node Generates Unique ID
    ncclUniqueId id;
    if (mpi.is_master()) {
        ncclGetUniqueId(&id);
    }
    
    // Broadcast Unique ID to all nodes using MPI
    mpi.broadcast(&id, sizeof(id), 0);

    // Initialize NCCL Communicator with ID
    init_nccl(rank, world_size, id);
    auto comm = get_nccl();

    if (mpi.is_master()) {
        std::cout << "[TitanCore] NCCL initialized, GPU communication ready.\n";
    }
    
    mpi.barrier();
    
    // --- Initialize Model, FSDP, and Training Loop Here ---
    // 

    // 3. Shutdown
    mpi.shutdown();                
    return 0;
}
