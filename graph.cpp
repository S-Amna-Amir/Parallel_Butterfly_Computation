#include "graph.h"
#include <mpi.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <unordered_set>

std::vector<int> Graph::peel_vertices_by_butterfly_count(const std::unordered_map<int, int>& butterfly_counts) const {
    std::unordered_map<int, int> counts = butterfly_counts;
    std::unordered_set<int> removed;
    std::vector<int> peel_order;

    while (!counts.empty()) {
        // Find vertex with minimum count
        auto min_it = std::min_element(
            counts.begin(), counts.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });

        int v = min_it->first;
        peel_order.push_back(v);
        removed.insert(v);
        counts.erase(v);
        
        // NOTE: If doing dynamic peeling, you'd recompute butterfly_counts here.
    }

    return peel_order;
}

//==========================================================================

std::unordered_map<int, int> Graph::count_butterflies_vertex() const {
    std::unordered_map<int, int> butterfly_counts;
    std::unordered_map<int, std::unordered_set<int>> adj_map;

    // Build adjacency map including all vertices (not just locals)
    for (size_t i = 0; i < local_vertex_ids.size(); ++i) {
        int u = local_vertex_ids[i];
        for (int v : adjacency_list[i]) {
            adj_map[u].insert(v);
            adj_map[v].insert(u); // Ensure bidirectionality
        }
    }

    // Get all wedges from local partition
    auto wedges = get_wedges();

    // Count butterflies
    for (const auto& wedge : wedges) {
        int u = wedge.u;
        int v = wedge.v;
        int w = wedge.w;

        // Look for a common neighbor x connected to both v and w, x != u
        for (int x : adj_map[v]) {
            if (x != u && adj_map[w].count(x)) {
                // (u, v, w, x) is a butterfly
                butterfly_counts[u]++;
                butterfly_counts[v]++;
                butterfly_counts[w]++;
                butterfly_counts[x]++;
            }
        }
    }

    // Divide counts by 4 to account for 4 appearances per butterfly
    for (auto& [vertex, count] : butterfly_counts) {
        count /= 4;
    }

    return butterfly_counts;
}

std::vector<Wedge> Graph::get_wedges() const {
    std::vector<Wedge> wedges;

    for (size_t i = 0; i < local_vertex_ids.size(); ++i) {
        int u = local_vertex_ids[i];
        const auto& neighbors = adjacency_list[i];

        // Sort neighbors in ascending order
        std::vector<int> sorted_neighbors = neighbors;
        std::sort(sorted_neighbors.begin(), sorted_neighbors.end());

        for (size_t j = 0; j < sorted_neighbors.size(); ++j) {
            int v = sorted_neighbors[j];
            for (size_t k = j + 1; k < sorted_neighbors.size(); ++k) {
                int w = sorted_neighbors[k];

                // Only consider wedges u-v-w with v < w to avoid duplication
                wedges.push_back({u, v, w});
            }
        }
    }

    return wedges;
}

//==========================================================================

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
