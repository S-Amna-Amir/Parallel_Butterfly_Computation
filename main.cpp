//run: 

// mpic++ -std=c++17 -O3 -fopenmp main.cpp graph.cpp -o butterfly
// ./butterfly tiny_graph.txt

// export OMP_NUM_THREADS=2
// mpirun -np 4 ./butterfly tiny_graph.txt

#include "graph.h"
#include <mpi.h>
#include <omp.h>
#include <iostream>

unsigned long long count_butterflies(const Graph& g) //global butterfly counting
{
    unsigned long long count = 0; //butterfly counter

    #pragma omp parallel for reduction(+:count)
    for (int u = 0; u < g.num_vertices(); ++u) //distribute vertices across threads (openmp)
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
                    count += common * (common - 1) / 2; //each pair of common neighbours forms 1 butterfly
                }
            }
        }
    }
    return count; //return total butterfly count in the graph
}

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2 && rank == 0) 
    {
        std::cerr << "Usage: " << argv[0] << " <graph_file>\n";
        MPI_Abort(MPI_COMM_WORLD, 1); //abort if input is missing
    }

    //rank 0 reads the graph and broadcasts metadata
    Graph g(argv[1]);
    
    unsigned long long total_butterflies = count_butterflies(g); //global butterfly count

    if (rank == 0) 
    {
        std::cout << "Graph: " << g.num_vertices() << " vertices, " << g.num_edges() << " edges\n";
        std::cout << "Total butterflies: " << total_butterflies << "\n";
    }

    MPI_Finalize();
    return 0;
}
