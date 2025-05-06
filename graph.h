#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <unordered_map>
#include <metis.h>  // For idx_t

class Graph {
private:
    std::vector<int> local_vertex_ids;       // Original global vertex IDs
    std::vector<std::vector<int>> adjacency_list; // Adjacency list (global IDs)
    std::vector<int> deg_u;                  // deg_u(u) for each local vertex

public:
    // Load partition from METIS results (in-memory)
    void loadPartition(const std::vector<idx_t>& local_vertices, const std::vector<std::vector<idx_t>>& global_adj);

    void preprocess();
    
    // Getters
    const std::vector<int>& getLocalVertexIDs() const { return local_vertex_ids; }
    const std::vector<std::vector<int>>& getAdjacencyList() const { return adjacency_list; }
    const std::vector<int>& getDegU() const { return deg_u; }
};

#endif
