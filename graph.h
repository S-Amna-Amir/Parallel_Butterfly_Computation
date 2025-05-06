// graph.h
#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <unordered_map>
#include <metis.h>
#include <tuple>
#include <utility>
#include <set>

using Wedge = std::tuple<int, int, int>;
struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        return std::hash<T1>{}(p.first) ^ std::hash<T2>{}(p.second);
    }
};

class Graph {
private:
    std::vector<int> local_vertex_ids;
    std::vector<std::vector<int>> adjacency_list;
    std::vector<int> deg_u;
    std::unordered_map<std::pair<int, int>, int, PairHash> edge_counts;
    std::set<std::pair<int, int>> active_edges;

public:
    void loadPartition(const std::vector<idx_t>& local_vertices,
                      const std::vector<std::vector<idx_t>>& global_adj);
    void preprocess();
    std::vector<Wedge> get_wedges() const;
    std::unordered_map<std::pair<int, int>, int, PairHash> count_edges() const;
    void initialize_edge_counts();
    std::vector<std::pair<int, int>> get_edges_to_peel(int current_min) const;
    void update_after_peeling(const std::vector<std::pair<int, int>>& peeled_edges);
    void adjust_counts(const std::vector<std::pair<int, int>>& peeled_edges);

    const std::vector<int>& getLocalVertexIDs() const { return local_vertex_ids; }
    const std::vector<std::vector<int>>& getAdjacencyList() const { return adjacency_list; }
    const std::vector<int>& getDegU() const { return deg_u; }
    const std::unordered_map<std::pair<int, int>, int, PairHash>& getEdgeCounts() const { return edge_counts; }
};

#endif
