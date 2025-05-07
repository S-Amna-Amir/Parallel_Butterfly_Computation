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

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <input_file> <num_partitions>" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    string filename = argv[1];
    int num_partitions = stoi(argv[2]);
    unordered_map<string, int> vertex_to_id;
    vector<string> id_to_vertex;
    vector<vector<idx_t>> adj;
    vector<pair<idx_t, idx_t>> edges;
    idx_t n = 0;

    if (rank == 0) {
        ifstream file(filename);
        string u, v;
        while (file >> u >> v) {
            if (!vertex_to_id.count(u)) {
                vertex_to_id[u] = id_to_vertex.size();
                id_to_vertex.push_back(u);
                adj.resize(id_to_vertex.size());
            }
            int u_id = vertex_to_id[u];

            if (!vertex_to_id.count(v)) {
                vertex_to_id[v] = id_to_vertex.size();
                id_to_vertex.push_back(v);
                adj.resize(id_to_vertex.size());
            }
            int v_id = vertex_to_id[v];
            edges.push_back({u_id, v_id});
        }

        n = id_to_vertex.size();
        if (n == 0) {
            cerr << "No vertices found!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        id_to_vertex.resize(n);
        adj.resize(n);
        edges.resize(0);
    }

    for (int i = 0; i < n; ++i) {
        int len;
        char buf[256];
        if (rank == 0) {
            len = id_to_vertex[i].size();
            strcpy(buf, id_to_vertex[i].c_str());
        }
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(buf, len, MPI_CHAR, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            id_to_vertex[i] = string(buf, len);
        }
    }

    int num_edges;
    if (rank == 0) {
        num_edges = edges.size();
    }
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        edges.resize(num_edges);
    }

    for (int i = 0; i < num_edges; ++i) {
        idx_t u_id, v_id;
        if (rank == 0) {
            u_id = edges[i].first;
            v_id = edges[i].second;
        }
        MPI_Bcast(&u_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&v_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            edges[i].first = u_id;
            edges[i].second = v_id;
        }
    }

    for (int i = 0; i < num_edges; ++i) {
        idx_t u_id = edges[i].first;
        idx_t v_id = edges[i].second;
        #pragma omp critical
        {
            adj[u_id].push_back(v_id);
            adj[v_id].push_back(u_id);
        }
    }

    vector<idx_t> part;
    if (rank == 0) {
        vector<idx_t> xadj(n + 1, 0);
        vector<idx_t> adjncy;

        for (idx_t i = 0; i < n; ++i) {
            xadj[i + 1] = xadj[i] + adj[i].size();
            adjncy.insert(adjncy.end(), adj[i].begin(), adj[i].end());
        }

        idx_t ncon = 1;
        idx_t objval;
        part.resize(n);
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);

        int ret = METIS_PartGraphKway(&n, &ncon, xadj.data(), adjncy.data(),
                                    NULL, NULL, NULL, &num_partitions,
                                    NULL, NULL, options, &objval, part.data());

        if (ret != METIS_OK) {
            cerr << "METIS partitioning failed!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (rank != 0)
        part.resize(n);
    MPI_Bcast(part.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> local_vertices;
    for (idx_t i = 0; i < n; ++i) {
        if (part[i] == rank) {
            local_vertices.push_back(i);
        }
    }

    cout << "Process " << rank << " has " << local_vertices.size()
         << " vertices:" << endl;
    for (int local_v : local_vertices) {
        cout << "Process " << rank <<":  " << id_to_vertex[local_v] << " connects to: ";
        for (idx_t neighbor : adj[local_v]) {
            cout << id_to_vertex[neighbor] << " ";
        }
        cout << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "\nGlobal partitions:" << endl;
        for (idx_t i = 0; i < n; ++i) {
            cout << id_to_vertex[i] << " -> " << part[i] << endl;
        }
    }

    Graph g;
    g.loadPartition(local_vertices, adj); 
    g.preprocess();
    
    if (rank == 0) {
        std::cout << "\nPreprocessing complete. Sample local vertex degrees:\n";
        const auto& deg_u = g.getDegU();
        for (size_t i = 0; i < 5 && i < deg_u.size(); ++i) {
            std::cout << "Vertex " << g.getLocalVertexIDs()[i] 
                      << " modified degree: " << deg_u[i] << "\n";
        }
    }
    else {
        std::cout << "\nPreprocessing complete. Sample local vertex degrees:\n";
        const auto& deg_u = g.getDegU();
        for (size_t i = 0; i < 5 && i < deg_u.size(); ++i) {
            std::cout << "Vertex " << g.getLocalVertexIDs()[i] 
                      << " modified degree: " << deg_u[i] << "\n";
        }
    }
    
    //auto wedges = g.get_wedges();


	// ---------------------- Butterfly Counting ----------------------

    // Call the reference butterfly counting function
    auto butterfly_counts = g.count_butterflies_vertex();

    // Process and print the results
    for (const auto& [int_id, count] : butterfly_counts) {
        // Only output if count > 0
        if (count > 0) {
            std::cout << "Process " << rank << ": Vertex (rank) "
                      << id_to_vertex[int_id] << " is part of "
                      << count << " butterflies\n";
        }
    }

	auto peel_order = g.peel_vertices_by_butterfly_count(butterfly_counts);

		std::cout << "\n\nProcess " << rank << ": Vertex (rank) peeling order based on butterfly participation: ";
		for (int v : peel_order) {
			std::cout << v << " ";
	}
	std::cout << std::endl;

	//------------------------------------
	
	    // ... [previous code remains unchanged]

    auto butterfly_edges = g.count_butterflies_edge();

    // Process and print EDGE results
    for (const auto& [edge_pair, count] : butterfly_edges) {
        if (count > 0) {
            // Get original vertex names from global ranks
            std::string u_name = id_to_vertex[edge_pair.first];
            std::string v_name = id_to_vertex[edge_pair.second];
            
            std::cout << "Process " << rank << ": Edge (" 
                      << u_name << " - " << v_name << ") is part of "
                      << count << " butterflies\n";
        }
    }



    MPI_Finalize();
    return 0;
}
