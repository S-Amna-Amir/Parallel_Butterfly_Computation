#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <unordered_map>
#include <metis.h>
#include <tuple>

struct PairHash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

struct Wedge {
    int u;
    int w;
    int v;
    Wedge(int u, int w, int v) : u(u), w(w), v(v) {}
};

class Graph {
private:
    std::vector<int> local_vertex_ids;
    std::vector<std::vector<int>> adjacency_list;
    std::vector<int> deg_u;

public:
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
    std::vector<int> peel_vertices_by_butterfly_count(const std::unordered_map<int, int>&) const;
};

#endif
