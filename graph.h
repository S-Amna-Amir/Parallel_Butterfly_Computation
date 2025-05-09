#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <unordered_map>
#include <metis.h>  // For idx_t
#include <tuple>
#include <metis.h>  // For idx_t
#include <boost/functional/hash.hpp>  // Add this for boost::hash


struct PairHash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};
inline std::pair<int, int> OrderedPair(int a, int b) { // Add "inline" here
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

// A wedge is represented as (endpoint1, endpoint2, center)
struct Wedge {
    int u;
    int w;
    int v;
    Wedge(int u, int w, int v) : u(u), w(w), v(v) {}
};
class Graph {
private:
    std::vector<int> local_vertex_ids;       // Original global vertex IDs
    std::vector<std::vector<int>> adjacency_list; // Adjacency list (global IDs)
    std::vector<int> deg_u;                  // deg_u(u) for each local vertex

public:
    // Load partition from METIS results (in-memory)
    void loadPartition(const std::vector<idx_t>& local_vertices, const std::vector<std::vector<idx_t>>& global_adj);

    void preprocess();
    std::vector<Wedge> get_wedges() const;
    
    // Getters
    const std::vector<int>& getLocalVertexIDs() const { return local_vertex_ids; }
    const std::vector<std::vector<int>>& getAdjacencyList() const { return adjacency_list; }
    const std::vector<int>& getDegU() const { return deg_u; }
    
    
    // Butterfly counting
    std::unordered_map<int, int> count_butterflies_vertex() const;
    std::unordered_map<std::pair<int, int>, int, PairHash> count_butterflies_edge() const;
    
    // Peeling
    std::vector<int> peel_vertices_by_butterfly_count(const std::unordered_map<int, int>&, int& num_iterations) const;
    std::vector<std::pair<int, int>> peel_edges_by_butterfly_count(const std::unordered_map<std::pair<int, int>, int, PairHash>&, int& num_iterations) const;
};

#endif
