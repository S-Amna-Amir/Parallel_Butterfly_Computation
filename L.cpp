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
    std::unordered_set<idx_t> U, V; //using set to track unique nodes
    for (auto &e : edges) 
    {
        U.insert(e.first); //set U nodes
        V.insert(e.second); //set V nodes
    }
    return { E, (int)U.size(), (int)V.size() };
}

int main(int argc, char *argv[]) 
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) //checking cli args
    {
        if (rank == 0) 
        {
            cerr << "Usage: " << argv[0] << " <input_file> <num_partitions>" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    string filename = argv[1];
    int num_partitions = stoi(argv[2]);
    
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
    if (rank == 0) //broadcasr edge count
    {
        num_edges = edges.size();
    }
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) 
    {
        edges.resize(num_edges);
    }

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

            int ret = METIS_PartGraphKway(&n, &ncon, xadj.data(), adjncy.data(), NULL, NULL, NULL, &num_partitions, NULL, NULL, options, &objval, part.data()); //k-way partitioning

            if (ret != METIS_OK) 
            {
                cerr << "METIS partitioning failed!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        if (rank != 0)
            part.resize(n);
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
    //print local vertices and their connections:
    cout << "Process " << rank << " has " << local_vertices.size() << " vertices:" << endl;
    for (int local_v : local_vertices) 
    {
        cout << "Process " << rank <<":  " << id_to_vertex[local_v] << " connects to: ";
        for (idx_t neighbor : adj[local_v]) 
        {
            cout << id_to_vertex[neighbor] << " ";
        }
        cout << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) //print global partitioning mapping
    {
        cout << "\nGlobal partitions:" << endl;
        for (idx_t i = 0; i < n; ++i) 
        {
            cout << id_to_vertex[i] << " -> " << part[i] << endl;
        }
    }

    Graph g; //g is the subgraph assigned to this process
    g.loadPartition(local_vertices, adj); 
    g.preprocess();
    
    if (rank == 0) //print sample vertex degrees after preprocessing
    {
        std::cout << "\nPreprocessing complete. Sample local vertex degrees:\n";
        const auto& deg_u = g.getDegU();
        for (size_t i = 0; i < 5 && i < deg_u.size(); ++i) 
        {
            std::cout << "Vertex " << g.getLocalVertexIDs()[i] << " modified degree: " << deg_u[i] << "\n";
        }
    }
    else 
    {
        std::cout << "\nPreprocessing complete. Sample local vertex degrees:\n";
        const auto& deg_u = g.getDegU();
        for (size_t i = 0; i < 5 && i < deg_u.size(); ++i) 
        {
            std::cout << "Vertex " << g.getLocalVertexIDs()[i] << " modified degree: " << deg_u[i] << "\n";
        }
    }
    
    auto butterfly_counts = g.count_butterflies_vertex(); //count butterflies per vertex
    for (const auto& [int_id, count] : butterfly_counts) 
    {
        if (count > 0) 
        {
            std::cout << "Process " << rank << ": Vertex (rank) " << id_to_vertex[int_id] << " is part of " << count << " butterflies\n";
        }
    }

    int vertex_iterations = 0;
    auto vertex_peel_order = g.peel_vertices_by_butterfly_count(butterfly_counts, vertex_iterations); //vertex peeling
    std::cout << "\nProcess " << rank << ": Vertex (rank) peeling took " << vertex_iterations << " iterations" << std::endl;
    std::cout << "\n\nProcess " << rank << ": Vertex (rank) peeling order based on butterfly participation: ";
    for (int v : vertex_peel_order) 
    {
        std::cout << v << " ";
    }
    std::cout << std::endl;
    
    int global_vertex_iterations = 0;
    MPI_Reduce(&vertex_iterations, &global_vertex_iterations, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); //sum vertex peeling iterations across all processes
    if (rank == 0) 
    {
        std::cout << "Global total vertex peeling iterations: " << global_vertex_iterations << std::endl;
    }

    auto butterfly_edges = g.count_butterflies_edge(); //count butterflies per edge
    int total_count = 0;
    for (const auto& [edge_pair, count] : butterfly_edges) 
    {
        if (count > 0) 
        {
            //get original vertex names from global ranks:
            std::string u_name = id_to_vertex[edge_pair.first];
            std::string v_name = id_to_vertex[edge_pair.second];
            std::cout << "Process " << rank << ": Edge (" << u_name << " - " << v_name << ") is part of " << count << " butterflies\n";
            total_count += count;
        }
    }
    total_count /= 4; //each butterfly counted 4 times (once per edge)
    
    int edge_iterations = 0;
    auto edge_butterflies = g.count_butterflies_edge();
    auto edge_peel_order = g.peel_edges_by_butterfly_count(edge_butterflies, edge_iterations); //edge peeling
    std::cout << "\nProcess " << rank << ": Edge (rank) peeling took " << edge_iterations << " iterations" << std::endl;
    std::cout << "\nProcess " << rank << ": Edge peeling order:\n";
    for (const auto& edge : edge_peel_order) 
    {
        std::cout << "(" << id_to_vertex[edge.first] << " - " << id_to_vertex[edge.second] << ") ";
    }
    std::cout << std::endl;
	
    int global_edge_iterations = 0;
    MPI_Reduce(&edge_iterations, &global_edge_iterations, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); //sum edge peeling iterations across all processes
    if (rank == 0) 
    {
        std::cout << "Global total edge peeling iterations: " << global_edge_iterations << std::endl;
    }

    auto [E, numU, numV] = bipartite_stats(edges);
    std::cout <<"Total edges = "<< E << ", |U| = "<< numU << ", |V| = "<< numV<<std::endl;
    std::cout<<"Total number of butterflies: "<<total_count<<std::endl;
	
    int global_total_butterfly_count = 0;
    MPI_Reduce(&total_count, &global_total_butterfly_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); //sum butterfly counts across all processes
    if (rank == 0) 
    {
        std::cout << "global_total_butterfly_count: " << global_total_butterfly_count << std::endl;
    }
	
    MPI_Finalize();
    return 0;
}
