#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <metis.h>
#include <mpi.h>
#include <utility>  // For pair
#include "graph.h"

using namespace std;

int main(int argc, char *argv[]) 
{
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
vector<pair<idx_t, idx_t>> edges;  // Store original edges
idx_t n = 0;

// Only root reads the file
		if (rank == 0) {
		ifstream file(filename);
		string u, v;
// Read edges and build graph
		while (file >> u >> v) {
		// Process vertex u
		if (!vertex_to_id.count(u)) {
		vertex_to_id[u] = id_to_vertex.size();
		id_to_vertex.push_back(u);
		adj.resize(id_to_vertex.size());
		}
		int u_id = vertex_to_id[u];

		// Process vertex v
		if (!vertex_to_id.count(v)) {
		vertex_to_id[v] = id_to_vertex.size();
		id_to_vertex.push_back(v);
		adj.resize(id_to_vertex.size());
            }
            int v_id = vertex_to_id[v];

		// Store original edge
            	edges.push_back({u_id, v_id});

	}

        n = id_to_vertex.size();
        if (n == 0) {
            cerr << "No vertices found!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize vectors on other processes
    if (rank != 0) {
        id_to_vertex.resize(n);
        adj.resize(n);
        edges.resize(0); // Clear edges on non-root processes
    }

    // Broadcast id_to_vertex (vertex names)
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

    // Broadcast edges
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

    // Build adjacency list on all processes
    for (int i = 0; i < num_edges; ++i) {
        idx_t u_id = edges[i].first;
        idx_t v_id = edges[i].second;
        adj[u_id].push_back(v_id);
        adj[v_id].push_back(u_id);
    }

    // METIS partitioning (only on root)
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

    // Broadcast partition vector
    if (rank != 0)
        part.resize(n);
    MPI_Bcast(part.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process prints its local vertices and adjacency list
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

    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes finish printing before root prints

    // Root prints global info
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
    else
    {
    	std::cout << "\nPreprocessing complete. Sample local vertex degrees:\n";
        const auto& deg_u = g.getDegU();
        for (size_t i = 0; i < 5 && i < deg_u.size(); ++i) {
            std::cout << "Vertex " << g.getLocalVertexIDs()[i] 
                      << " modified degree: " << deg_u[i] << "\n";
        }
    }
    
    
	 auto wedges = g.get_wedges();

	std::cout << "Process " << rank << " found " << wedges.size() 
              << " wedges\n";
    if (!wedges.empty()) {
        const auto& [u, w, v] = wedges[0];
        std::cout << "Sample wedge: (" << u << ", " << w << ", " << v << ")\n";
    }

	
	

	
    MPI_Finalize();
    return 0;
}

