// run:
// mpicxx -std=c++17 -fopenmp main.cpp graph.cpp -lmetis -o x
// export OMP_NUM_THREADS=2
// ime mpirun -np 2 ./x 6cores.txt 2 v

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <metis.h>
#include <mpi.h>
#include <utility>
#include "graph.h"
#include <climits>
#include <unordered_set>
#include <algorithm>
using namespace std;

std::tuple<int,int,int> bipartite_stats(const std::vector<std::pair<idx_t,idx_t>>& edges) //stats: total edges, |U|, |V|
{
    int E = edges.size();
    unordered_set<idx_t> U, V; //using set to track unique nodes
    for (auto &e : edges) 
    {
        U.insert(e.first); //set U nodes
        V.insert(e.second); //set V nodes
    }
    return {E, (int)U.size(), (int)V.size()};
}

int main(int argc, char *argv[]) 
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) //checking cli args
    {
        if (rank == 0)
        {
            cerr << "Usage: " << argv[0] << " <input_file> <num_partitions> <mode:v|e>" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    string filename = argv[1];
    int num_partitions = stoi(argv[2]);
    char mode = argv[3][0];

    //timing variables
    double start, end;
    double load_partition_time = 0.0, preprocess_time = 0.0;
    double vertex_count_time = 0.0, vertex_peel_time = 0.0;
    double edge_count_time = 0.0, edge_peel_time = 0.0;
    
    //data structures for graph storage:
    unordered_map<string, int> vertex_to_id; //map vertex names to ids
    vector<string> id_to_vertex; //map ids back to names
    vector<vector<idx_t>> adj; //adjacency list
    vector<pair<idx_t, idx_t>> edges; //edge list
    idx_t n = 0; //total vertex count

    if (rank == 0) //read input file (will only happen once)
    {
        ifstream file(filename);
        string u, v;
        while (file >> u >> v) 
        {
            if (!vertex_to_id.count(u)) //create ids for new vertices
            {
                vertex_to_id[u] = id_to_vertex.size();
                id_to_vertex.push_back(u);
                adj.resize(id_to_vertex.size());
            }
            int u_id = vertex_to_id[u];

            if (!vertex_to_id.count(v)) 
            {
                vertex_to_id[v] = id_to_vertex.size();
                id_to_vertex.push_back(v);
                adj.resize(id_to_vertex.size());
            }
            int v_id = vertex_to_id[v];
            edges.push_back({u_id, v_id}); //store edge
        }
        n = id_to_vertex.size();
        if (n == 0) 
        {
            cerr << "No vertices found!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD); //broadcast total vertex count to all processes

    if (rank != 0) //prepare containers
    {
        id_to_vertex.resize(n);
        adj.resize(n);
        edges.resize(0);
    }

    for (int i = 0; i < n; ++i) //broadcast vertex names to all processes
    {
        int len;
        char buf[256];
        if (rank == 0) 
        {
            len = id_to_vertex[i].size();
            strcpy(buf, id_to_vertex[i].c_str());
        }
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(buf, len, MPI_CHAR, 0, MPI_COMM_WORLD);
        if (rank != 0) 
        {
            id_to_vertex[i] = string(buf, len);
        }
    }

    int num_edges;
    if (rank == 0) num_edges = edges.size();
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD); //broadcast edge count

    if (rank != 0) edges.resize(num_edges);

    for (int i = 0; i < num_edges; ++i) //broadcast all edges
    {
        idx_t u_id, v_id;
        if (rank == 0) 
        {
            u_id = edges[i].first;
            v_id = edges[i].second;
        }
        MPI_Bcast(&u_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&v_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) 
        {
            edges[i].first = u_id;
            edges[i].second = v_id;
        }
    }

    for (int i = 0; i < num_edges; ++i) //build adjacency list (undirected graph)
    {
        idx_t u_id = edges[i].first;
        idx_t v_id = edges[i].second;
        #pragma omp critical
        {
            adj[u_id].push_back(v_id);
            adj[v_id].push_back(u_id);
        }
    }

    vector<idx_t> part;
    if (num_partitions > 1) //partition graph into subgraphs for each process using metis
    {
        if (rank == 0) 
        {
            //metis input format:
            vector<idx_t> xadj(n + 1, 0); //csr index array
            vector<idx_t> adjncy; //csr neighbour array
            for (idx_t i = 0; i < n; ++i) 
            {
                xadj[i + 1] = xadj[i] + adj[i].size();
                adjncy.insert(adjncy.end(), adj[i].begin(), adj[i].end());
            }
            
            //metis partitioning parameters:
            idx_t ncon = 1;
            idx_t objval;
            part.resize(n);
            idx_t options[METIS_NOPTIONS];
            METIS_SetDefaultOptions(options);
            int ret = METIS_PartGraphKway(&n, &ncon, xadj.data(), adjncy.data(), NULL, NULL, NULL, &num_partitions, NULL, NULL, options, &objval, part.data());
            if (ret != METIS_OK) 
            {
                cerr << "METIS partitioning failed!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        if (rank != 0) part.resize(n);
        MPI_Bcast(part.data(), n, MPI_INT, 0, MPI_COMM_WORLD); //broadcast partition results to all processes
    } 
    else 
    {
        part.resize(n, 0); //if only one partition, assign all to partition 0
    }

    vector<int> local_vertices;
    for (idx_t i = 0; i < n; ++i) //get local vertices assigned to this process
    {
        if (part[i] == rank) 
        {
            local_vertices.push_back(i);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    std::cout << "Rank " << rank << " has " << local_vertices.size() << " vertices." << std::endl;

    Graph g; //g is the subgraph assigned to this process
    start = MPI_Wtime();
    g.loadPartition(local_vertices, adj);
    end = MPI_Wtime();
    load_partition_time = end - start;

    start = MPI_Wtime();
    g.preprocess();
    end = MPI_Wtime();
    preprocess_time = end - start;

    int vertex_iterations = 0, edge_iterations = 0;
    int total_count = 0, local_vertex_total = 0;

    if (mode == 'v') 
    {
        start = MPI_Wtime();
        auto butterfly_counts = g.count_butterflies_vertex(); //count butterflies per vertex
        end = MPI_Wtime();
        vertex_count_time = end - start;

        start = MPI_Wtime();
        auto vertex_peel_order = g.peel_vertices_by_butterfly_count(butterfly_counts, vertex_iterations); //vertex peeling
        end = MPI_Wtime();
        vertex_peel_time = end - start;
        
        for (const auto& [vertex, count] : butterfly_counts) 
        {
            if (count > 0) local_vertex_total += count;
        }
        local_vertex_total /= 4;
    }
    else if (mode == 'e') 
    {
        start = MPI_Wtime();
        auto butterfly_edges = g.count_butterflies_edge(); //count butterflies per edge
        end = MPI_Wtime();
        edge_count_time = end - start;

        start = MPI_Wtime();
        auto edge_peel_order = g.peel_edges_by_butterfly_count(butterfly_edges, edge_iterations); //edge peeling
        end = MPI_Wtime();
        edge_peel_time = end - start;

        for (const auto& [edge_pair, count] : butterfly_edges) {
            if (count > 0) total_count += count;
        }
        total_count /= 4;
    }

    //mpi reductions
    int global_vertex_iter = 0, global_edge_iter = 0;
    int global_edge_total = 0, global_vertex_total = 0;
    if (mode == 'v') 
    {
        MPI_Reduce(&vertex_iterations, &global_vertex_iter, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&local_vertex_total, &global_vertex_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    }
    else 
    {
        MPI_Reduce(&edge_iterations, &global_edge_iter, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_count, &global_edge_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    //timing reductions
    double timings[6] = {load_partition_time, preprocess_time, vertex_count_time, vertex_peel_time, edge_count_time, edge_peel_time};
    double max_timings[6];
    MPI_Reduce(timings, max_timings, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) 
    {
        auto [E, numU, numV] = bipartite_stats(edges);
        cout << "\n---> Global results <---\n";
        if (mode == 'v') 
        {
            cout << "Total butterflies: " << global_vertex_total << endl;
            cout << "Vertex peeling iterations: " << global_vertex_iter << endl;
        } 
        else 
        {
            cout << "Total butterflies: " << global_edge_total << endl;
            cout << "Edge peeling iterations: " << global_edge_iter << endl;
        }
        cout << "Total edges: " << E << endl;
        cout << "|U| = " << numU << "\n|V| = " << numV << endl;

        cout << "\n---> Timing Results (Max Across Processes) <---\n"
             << "loadPartition: " << max_timings[0] << "s\n"
             << "preprocess: " << max_timings[1] << "s\n";
        if (mode == 'v') 
        {
            cout << "vertex_count: " << max_timings[2] << "s\n"
                 << "vertex_peel: " << max_timings[3] << "s\n";
        } 
        else 
        {
            cout << "edge_count: " << max_timings[4] << "s\n"
                 << "edge_peel: " << max_timings[5] << "s\n";
        }
    }

    MPI_Finalize();
    return 0;
}
