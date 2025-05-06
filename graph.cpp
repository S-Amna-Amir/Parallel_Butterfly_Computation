// graph.cpp
#include "graph.h"
#include <fstream>
#include <unordered_map>
#include <algorithm>

Graph::Graph(const std::string& filename) 
{
    std::ifstream file(filename);
    int u, v;
    std::vector<std::pair<int, int>> edges;

    while (file >> u >> v) //read edges
    {
        edges.emplace_back(u, v);
    }

    //find max vertex to determine partitions
    int max_u = 0, max_v = 0;
    for (const auto& [u, v] : edges) 
    {
        max_u = std::max(max_u, u);
        max_v = std::max(max_v, v);
    }
    num_vertices_U = max_u + 1;

    //build csr for U->V and V->U
    //why csr? memory efficiency, cache-friendly access, and parallelization ready :D
    //example: for a graph with 1M vertices but only 5M edges:
	//adjacency matrix: 1M × 1M = 1 trillion entries (mostly zeros)
	//csr: just stores ~5M edges 
	//Format	Memory	 Neighbor Access   Parallel Safe
	//CSR           Best	 O(1)	           Yes
	//Adj. List	High	 Pointer chasing   Maybe
	//Adj. Matrix	Huge	 O(1)	           Yes
    auto build_csr = [](const auto& edges, auto& offsets, auto& adj, bool swap_uv) 
    {
        std::unordered_map<int, std::vector<int>> edge_map;
        for (const auto& [u, v] : edges) 
        {
            int src = swap_uv ? v : u;
            int dst = swap_uv ? u : v;
            edge_map[src].push_back(dst);
        }

        offsets.resize(edge_map.size() + 1);
        offsets[0] = 0;
        for (int i = 0; i < edge_map.size(); ++i) 
        {
            offsets[i+1] = offsets[i] + edge_map[i].size();
            adj.insert(adj.end(), edge_map[i].begin(), edge_map[i].end());
        }
    };

    build_csr(edges, offsets_U, edges_U, false); //U->V
    build_csr(edges, offsets_V, edges_V, true); //V->U
}

// ------------------- Graph Methods -------------------
const int* Graph::neighbors(int vertex) const 
{
    return is_vertex_in_U(vertex) ? 
        &edges_U[offsets_U[vertex]] : 
        &edges_V[offsets_V[vertex - num_vertices_U]];
}

int Graph::degree(int vertex) const 
{
    return is_vertex_in_U(vertex) ? offsets_U[vertex+1] - offsets_U[vertex] : offsets_V[vertex - num_vertices_U + 1] - offsets_V[vertex - num_vertices_U];
}

bool Graph::is_vertex_in_U(int vertex) const 
{
    return vertex < num_vertices_U;
}

int Graph::count_common_neighbors(int u, int w) const 
{
    const int* nu = neighbors(u);
    const int* nw = neighbors(w);
    int i = 0, j = 0, count = 0;
    int du = degree(u), dw = degree(w);

    while (i < du && j < dw) 
    {
        if (nu[i] == nw[j]) 
        { 
        	count++; i++; j++; 
        }
        else if (nu[i] < nw[j])
        {
        	i++;
        }
        else 
        {
        	j++;
    	}
    }
    return count;
}

int Graph::num_vertices() const 
{ 
    return num_vertices_U + (offsets_V.size() - 1); 
}

int Graph::num_edges() const 
{ 
    return edges_U.size(); 
}





#include <mpi.h>

void Graph::broadcast(int root_rank) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Broadcast basic metadata first
    MPI_Bcast(&num_vertices_U, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

    // Broadcast U->V CSR data
    if (rank != root_rank) {
        // On receiving processes, resize vectors to receive data
        offsets_U.resize(num_vertices_U + 1);
    }
    MPI_Bcast(offsets_U.data(), offsets_U.size(), MPI_INT, root_rank, MPI_COMM_WORLD);

    // Broadcast edges_U size first
    int edges_U_size;
    if (rank == root_rank) {
        edges_U_size = edges_U.size();
    }
    MPI_Bcast(&edges_U_size, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

    if (rank != root_rank) {
        edges_U.resize(edges_U_size);
    }
    MPI_Bcast(edges_U.data(), edges_U_size, MPI_INT, root_rank, MPI_COMM_WORLD);

    // Broadcast V->U CSR data
    int offsets_V_size;
    if (rank == root_rank) {
        offsets_V_size = offsets_V.size();
    }
    MPI_Bcast(&offsets_V_size, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

    if (rank != root_rank) {
        offsets_V.resize(offsets_V_size);
    }
    MPI_Bcast(offsets_V.data(), offsets_V_size, MPI_INT, root_rank, MPI_COMM_WORLD);

    // Broadcast edges_V size
    int edges_V_size;
    if (rank == root_rank) {
        edges_V_size = edges_V.size();
    }
    MPI_Bcast(&edges_V_size, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

    if (rank != root_rank) {
        edges_V.resize(edges_V_size);
    }
    MPI_Bcast(edges_V.data(), edges_V_size, MPI_INT, root_rank, MPI_COMM_WORLD);
}
