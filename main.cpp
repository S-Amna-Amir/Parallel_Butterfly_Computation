//run: 

// mpic++ -std=c++17 -O3 -fopenmp main.cpp graph.cpp -o butterfly
// ./butterfly tiny_graph.txt

// export OMP_NUM_THREADS=2
// mpirun -np 4 ./butterfly tiny_graph.txt

#include "graph.h"
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <chrono>

unsigned long long count_butterflies_parallel(const Graph& g, int rank, int size) //global butterfly counting
{
    unsigned long long local_count = 0;
    unsigned long long total_count = 0;
    
    #pragma omp parallel for reduction(+:local_count) //thread-level parallelism
    for (int u = rank; u < g.num_vertices(); u += size) //process-level parallelism -> each process handles a subset of vertices
    {
        if (!g.is_vertex_in_U(u)) //only process set U vertices
        {
            continue;
        }
        for (int v_idx = 0; v_idx < g.degree(u); ++v_idx) //iterate thru all neighbours of u vertex
        {
            int v = g.neighbors(u)[v_idx]; //get v vertex from set V
            for (int w_idx = 0; w_idx < g.degree(v); ++w_idx) //iterate thru all neighbours of v vertex
            {
                int w = g.neighbors(v)[w_idx]; //get w vertex from set U (wedge: u -> v -> w)
                if (w > u) //avoid double counting by vertex ordering
                {
                    int common = g.count_common_neighbors(u, w); //count common neighbours between u and w vertices
                    local_count += common * (common - 1) / 2; //each pair of common neighbours forms 1 butterfly
                }
            }
        }
    }
    std::cout << "Process " << rank << " -> Local butterflies: " << local_count << "\n";
 
    MPI_Reduce(&local_count, &total_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); //sum results from all processes
    
    return total_count; //return total butterfly count in the graph
}

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD); //to ensure all processes start timing together
    auto total_start = std::chrono::high_resolution_clock::now();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2 && rank == 0) 
    {
        std::cerr << "Usage: " << argv[0] << " <graph_file>\n";
        MPI_Abort(MPI_COMM_WORLD, 1); //abort if input is missing
    }

    Graph g;
    if (rank == 0) //only rank 0 reads the graph and broadcasts metadata 
    {
        auto load_start = std::chrono::high_resolution_clock::now();
        g = Graph(argv[1]);
        auto load_end = std::chrono::high_resolution_clock::now();
        std::cout << "Graph loading time: " << std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count() << " ms\n";
    }
    
    auto bcast_start = std::chrono::high_resolution_clock::now();
    g.broadcast(0); //broadcast graph data to all processes
    auto bcast_end = std::chrono::high_resolution_clock::now();
    
    auto comp_start = std::chrono::high_resolution_clock::now();
    unsigned long long total_butterflies = count_butterflies_parallel(g, rank, size); //global butterfly count (two-layer parallelism)
    auto comp_end = std::chrono::high_resolution_clock::now();
    auto total_end = std::chrono::high_resolution_clock::now();

    if (rank == 0) 
    {
        std::cout << "Broadcast time: " << std::chrono::duration_cast<std::chrono::milliseconds>(bcast_end - bcast_start).count() << " ms\n";
        std::cout << "Computation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(comp_end - comp_start).count() << " ms\n";
        std::cout << "Total program time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count() << " ms\n";
        std::cout << "Graph: " << g.num_vertices() << " vertices, " << g.num_edges() << " edges\n";
        std::cout << "Total butterflies: " << total_butterflies << "\n";
    }
    
    MPI_Finalize();
    return 0;
}
