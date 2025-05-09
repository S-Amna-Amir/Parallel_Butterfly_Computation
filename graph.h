#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <unordered_map>
#include <metis.h> //for idx_t
#include <tuple>

struct PairHash //custom for edge storage
{
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const 
    {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

inline std::pair<int, int> OrderedPair(int a, int b) //ensures edges are always stored as (smaller_id, larger_id)
{
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

struct Wedge //wedge is represented as (endpoint1, endpoint2, center)
{
    int u;
    int w;
    int v;
    Wedge(int u, int w, int v) : u(u), w(w), v(v) {}
};

class Graph 
{
private:
    std::vector<int> local_vertex_ids; //original global vertex ids
    std::vector<std::vector<int>> adjacency_list; //adjacency list (global ids)
    std::vector<int> deg_u; //deg_u(u) for each local vertex

public:
    void loadPartition(const std::vector<idx_t>& local_vertices, const std::vector<std::vector<idx_t>>& global_adj); //load partition from metis results (in-memory) 
    void preprocess(); //prepare graph for distributed computation
    std::vector<Wedge> get_wedges() const; //generate all wedges (u-v-w paths) in the graph
    
    //getters
    const std::vector<int>& getLocalVertexIDs() const { return local_vertex_ids; }
    const std::vector<std::vector<int>>& getAdjacencyList() const { return adjacency_list; }
    const std::vector<int>& getDegU() const { return deg_u; }
    
    //butterfly counting
    std::unordered_map<int, int> count_butterflies_vertex() const;
    std::unordered_map<std::pair<int, int>, int, PairHash> count_butterflies_edge() const;
    
    //peeling
    std::vector<int> peel_vertices_by_butterfly_count(const std::unordered_map<int, int>&, int& num_iterations) const;
    std::vector<std::pair<int, int>> peel_edges_by_butterfly_count(const std::unordered_map<std::pair<int, int>, int, PairHash>&, int& num_iterations) const;
};
#endif
