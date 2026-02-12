int main(int argc, char** argv) {

    TitanMPI mpi(argc, argv);

    if (mpi.is_master()) {
        std::cout << "TitanCore booting cluster...\n";
    }

    mpi.barrier();

    /* initialize NCCL here */

    mpi.shutdown();
}
