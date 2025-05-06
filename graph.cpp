#include "graph.h"
#include <mpi.h>
#include <algorithm>
#include <vector>
#include <unordered_map>


#include <climits>


void Graph::initialize_edge_counts() {
    edge_counts = count_edges();
    active_edges.clear();
    for (const auto& [edge, _] : edge_counts) {
        active_edges.insert(edge);
    }
}

std::vector<std::pair<int, int>> Graph::get_edges_to_peel(int current_min) const {
    std::vector<std::pair<int, int>> to_peel;
    for (const auto& [edge, count] : edge_counts) {
        if (count == current_min && active_edges.count(edge)) {
            to_peel.push_back(edge);
        }
    }
    return to_peel;
}

void Graph::adjust_counts(const std::vector<std::pair<int, int>>& peeled_edges) {
    // Temporary storage for affected edges
    std::unordered_map<std::pair<int, int>, int, PairHash> delta;

    #pragma omp parallel for
    for (size_t i = 0; i < peeled_edges.size(); ++i) {
        const auto& [u1, v1] = peeled_edges[i];
        
        // Find u1's neighbors (v1's partition)
        auto u1_it = std::find(local_vertex_ids.begin(), local_vertex_ids.end(), u1);
        if (u1_it == local_vertex_ids.end()) continue;
        size_t u1_idx = u1_it - local_vertex_ids.begin();
        const auto& u1_neighbors = adjacency_list[u1_idx];

        // Find all u2 in N(v1) \ {u1}
        for (int u2 : u1_neighbors) {
            if (u2 == v1) continue;
            
            // Find u2's neighbors
            auto u2_it = std::find(local_vertex_ids.begin(), local_vertex_ids.end(), u2);
            if (u2_it == local_vertex_ids.end()) continue;
            size_t u2_idx = u2_it - local_vertex_ids.begin();
            const auto& u2_neighbors = adjacency_list[u2_idx];
            
            // Compute intersection of N(u1) and N(u2)
            std::vector<int> intersection;
            std::set_intersection(u1_neighbors.begin(), u1_neighbors.end(),
                                  u2_neighbors.begin(), u2_neighbors.end(),
                                  std::back_inserter(intersection));
            
            // Update counts for affected edges
            for (int v2 : intersection) {
                if (v2 == v1) continue;
                #pragma omp critical
                {
                    delta[{u1, v2}]--;
                    delta[{u2, v1}]--;
                    delta[{u2, v2}]--;
                }
            }
        }
    }

    // Apply delta to edge_counts
    for (const auto& [edge, change] : delta) {
        if (edge_counts.find(edge) != edge_counts.end()) {
            edge_counts[edge] += change;
            if (edge_counts[edge] <= 0) {
                active_edges.erase(edge);
                edge_counts.erase(edge);
            }
        }
    }
}

void Graph::update_after_peeling(const std::vector<std::pair<int, int>>& peeled_edges) {
    for (const auto& edge : peeled_edges) {
        active_edges.erase(edge);
        edge_counts.erase(edge);
    }
    adjust_counts(peeled_edges);
}

std::unordered_map<std::pair<int,int>,int,PairHash> Graph::count_edges() const {
    // Step A: Enumerate all wedges in this partition
    auto wedges = get_wedges();

    // Step B: Group wedges by the *original* edge (u,v)
    // We only count butterflies on edges that actually appear in adjacency_list.
    std::unordered_map<std::pair<int,int>, std::vector<int>, PairHash> edge_to_ws;
    for (auto &w : wedges) {
        int u, w_rank, v;
        std::tie(u, w_rank, v) = w;
        // the two *original* edges in that wedge are (u,v) and (w,v)
        // here we only collect for (u,v):
        auto e = std::minmax(u, v);
        // record the endpoint w for edge (e.first, e.second)
        edge_to_ws[e].push_back(w_rank);
    }

    // Step C: For each original edge, compute #butterflies = C(d,2)
    // where d = number of distinct w's for that edge.
    std::unordered_map<std::pair<int,int>,int,PairHash> edge_counts;
    for (auto &kv : edge_to_ws) {
        auto edge = kv.first;
        auto &vec_w = kv.second;

        // remove duplicates if any
        std::sort(vec_w.begin(), vec_w.end());
        vec_w.erase(std::unique(vec_w.begin(), vec_w.end()), vec_w.end());

        int d = static_cast<int>(vec_w.size());
        // each pair of distinct w's forms one butterfly on this edge
        edge_counts[edge] = d * (d - 1) / 2;
    }

    return edge_counts;
}


std::vector<Wedge> Graph::get_wedges() const {
    std::vector<Wedge> local_wedges;
    
    #pragma omp parallel
    {
        std::vector<Wedge> private_wedges;
        
        #pragma omp for schedule(dynamic, 16)
        for (size_t u_idx = 0; u_idx < local_vertex_ids.size(); ++u_idx) 
        {
            const int u_rank = local_vertex_ids[u_idx];
            const auto& u_neighbors = adjacency_list[u_idx];
            
            // iterates over all neighbors of u, because any of them could be a center
            for (int v_rank : u_neighbors) 
            {
                // Find if v, a neighbor, exists in our local partition
                auto v_it = std::lower_bound(local_vertex_ids.begin(),local_vertex_ids.end(),v_rank);
                
                if (v_it == local_vertex_ids.end() || *v_it != v_rank) continue;	//it doesnt exist here
                
                const size_t v_idx = std::distance(local_vertex_ids.begin(), v_it);	//calculates index of v
                const auto& v_neighbors = adjacency_list[v_idx];					//takes neighbours of v
                
                // Find valid endpoints w where w_rank > u_rank
                auto w_start = v_neighbors.begin();
                auto w_end = std::upper_bound(v_neighbors.begin(), v_neighbors.end(), u_rank, std::greater<int>()); // greater for comparison
                
                for (auto w_it = w_start; w_it != w_end; ++w_it) 
                {
                    if (*w_it != u_rank) {  // Avoid self-wedges
                        private_wedges.emplace_back(u_rank, *w_it, v_rank);
                    }
                }
            }
        }
        
        #pragma omp critical
        local_wedges.insert(local_wedges.end(), 
private_wedges.begin(), private_wedges.end());
    }
    
    return local_wedges;
}


// Load partition data into the Graph object
void Graph::loadPartition(const std::vector<idx_t>& metis_local_vertices, const std::vector<std::vector<idx_t>>& global_adj) {
    // Clear existing data
    local_vertex_ids.clear();
    adjacency_list.clear();

    // Convert METIS idx_t to int and populate local vertices
    for (idx_t global_id : metis_local_vertices) {
        local_vertex_ids.push_back(static_cast<int>(global_id));
    }

    // Build adjacency list with global vertex IDs
    for (int local_idx = 0; local_idx < local_vertex_ids.size(); ++local_idx) {
        int global_id = local_vertex_ids[local_idx];
        std::vector<int> neighbors;
        for (idx_t neighbor : global_adj[global_id]) {
            neighbors.push_back(static_cast<int>(neighbor));
        }
        adjacency_list.push_back(neighbors);
    }
}



void Graph::preprocess() 
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Step 1: Compute local degrees
    int local_vertex_count = local_vertex_ids.size();
    std::vector<int> local_degrees(local_vertex_count);
    for (int i = 0; i < local_vertex_count; ++i) 
    {
        local_degrees[i] = adjacency_list[i].size();
    }

    // Step 2: Prepare send buffer (vertex ID, degree pairs)
    std::vector<int> send_buffer;
    send_buffer.reserve(2 * local_vertex_count);
    for (int i = 0; i < local_vertex_count; ++i) 
    {
        send_buffer.push_back(local_vertex_ids[i]);
        send_buffer.push_back(local_degrees[i]);
    }

    // Step 3: Gather send counts from all processes
    std::vector<int> recv_counts(world_size);
    int send_count = send_buffer.size();
    MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Step 4: Compute displacements for MPI_Allgatherv
    std::vector<int> displacements(world_size, 0);
    for (int i = 1; i < world_size; ++i) 
    {
        displacements[i] = displacements[i-1] + recv_counts[i-1];
    }

    // Step 5: Allgatherv to gather all vertex-degree pairs
    int total_recv_count = displacements.back() + recv_counts.back();
    std::vector<int> recv_buffer(total_recv_count);
    MPI_Allgatherv(send_buffer.data(), send_count, MPI_INT,
                   recv_buffer.data(), recv_counts.data(), displacements.data(),
                   MPI_INT, MPI_COMM_WORLD);

    // Step 6: Parse into global_vertex_degrees
    std::vector<std::pair<int, int>> global_vertex_degrees;
    for (size_t i = 0; i < recv_buffer.size(); i += 2) 
    {
        int id = recv_buffer[i];
        int degree = recv_buffer[i+1];
        global_vertex_degrees.emplace_back(id, degree);
    }

    // Step 7: Sort by degree (descending) and vertex ID (ascending for ties)
    std::sort(global_vertex_degrees.begin(), global_vertex_degrees.end(),
          [](const std::pair<int,int>& a,
             const std::pair<int,int>& b) -> bool
          {
              // If degrees equal, tie-break on vertex ID ascending:
              if (a.second == b.second)
                  return a.first < b.first;
              // Otherwise sort by descending degree:
              return a.second > b.second;
          });


    // Step 8: Create global vertex ID to rank map
    std::unordered_map<int, int> vertex_id_to_rank;
    for (size_t i = 0; i < global_vertex_degrees.size(); ++i) 
    {
        vertex_id_to_rank[global_vertex_degrees[i].first] = i;
    }

    // Step 9: Rename local vertex IDs to their global ranks
    for (int i = 0; i < local_vertex_count; ++i) 
    {
        local_vertex_ids[i] = vertex_id_to_rank[local_vertex_ids[i]];
    }

    // Step 10: Rename neighbors in adjacency lists and sort them
    #pragma omp parallel for
    for (size_t i = 0; i < adjacency_list.size(); ++i) 
    {
        auto& neighbors = adjacency_list[i];
        for (auto& neighbor : neighbors) {
            neighbor = vertex_id_to_rank[neighbor];
        }
        std::sort(neighbors.begin(), neighbors.end(), std::greater<int>());
    }

    // Step 11: Compute deg_u for each local vertex
    deg_u.resize(local_vertex_count);
    #pragma omp parallel for
    for (size_t i = 0; i < adjacency_list.size(); ++i) {
        int u_rank = local_vertex_ids[i];
        auto& neighbors = adjacency_list[i];
        auto it = std::upper_bound(neighbors.begin(), neighbors.end(), u_rank, std::greater<int>());
        deg_u[i] = std::distance(neighbors.begin(), it);
    }
}
