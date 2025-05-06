//run

// mpic++ mpi_graph_splitter.cpp -o graph_splitter
// mpirun -np 4 ./graph_splitter bipartite.txt

/*
-> what it's doing
Step 1: Master reads graph file
Step 2: Splits lines equally among all processes
Step 3: Sends lines to workers
Step 4: Workers process and send data back
Step 5: Master saves data into separate files
Step 6: (Optionally) deletes them later based on user input
*/

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>

/*void wrapper(rank)
{

}*/

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<std::string> lines;
    int total_lines = 0;

    if (argc < 2) 
    {
        if (rank == 0) std::cerr << "Usage: mpirun -n <num_processes> ./your_program <graph_file.txt>\n";
        MPI_Abort(MPI_COMM_WORLD, 1); //abort all processes
    }

    std::string filename = argv[1];

    if (rank == 0) 
    {
        std::ifstream file(filename);
        if (!file.is_open()) 
        {
            std::cerr << "Error opening file: " << filename << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1); //exit if file can't be opened
        }

        std::string line;
        while (getline(file, line)) 
        {
            lines.push_back(line);
        }
        total_lines = lines.size();
        std::cout << "Total lines in " << filename << ": " << total_lines << std::endl;
    }

    MPI_Bcast(&total_lines, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int lines_per_process = total_lines / size;
    int remainder = total_lines % size;

    std::vector<int> counts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) 
    {
        counts[i] = lines_per_process + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }

    int local_count = counts[rank];
    std::vector<std::string> local_lines(local_count);

    if (rank == 0) 
    {
        for (int i = 1; i < size; ++i) 
        {
            int dest_count = counts[i];
            for (int j = 0; j < dest_count; ++j) 
            {
                int global_idx = displs[i] + j;
                MPI_Send(lines[global_idx].c_str(), lines[global_idx].size() + 1, MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }
        for (int j = 0; j < local_count; ++j) 
        {
            local_lines[j] = lines[displs[rank] + j];
        }
    } 
    else 
    {
        for (int j = 0; j < local_count; ++j) 
        {
            char buffer[1024];
            MPI_Recv(buffer, 1024, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_lines[j] = std::string(buffer);
        }
    }

    std::ofstream outfile("subgraph_" + std::to_string(rank) + ".txt");
    for (const auto& line : local_lines) 
    {
        outfile << line << std::endl;
    }
    std::cout << "Rank " << rank << " created subgraph_" << rank << ".txt" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD); //ensure all files are created before prompting

    if (rank == 0) 
    {
        std::string input;
        std::cout << "Now delete? (y/n): ";
        std::cin >> input;

        if (input == "y" || input == "Y") 
        {
            for (int i = 0; i < size; ++i) 
            {
                std::string filename = "subgraph_" + std::to_string(i) + ".txt";
                if (std::remove(filename.c_str()) == 0) 
                {
                    std::cout << "Deleted " << filename << std::endl;
                } 
                else 
                {
                    std::cerr << "Failed to delete " << filename << std::endl;
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}

