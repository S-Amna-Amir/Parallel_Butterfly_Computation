#include "graph.h"
#include <mpi.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>
#include <queue>
#include <utility>
std::vector<std::pair<int, int>> Graph::peel_edges_by_butterfly_count(
    const std::unordered_map<std::pair<int, int>, int, PairHash>& edge_counts, int& num_iterations) const 
{
    std::unordered_map<std::pair<int, int>, int, PairHash> counts = edge_counts;
    std::unordered_set<std::pair<int, int>, PairHash> removed_edges;
    std::vector<std::pair<int, int>> peel_order;

    std::unordered_map<int, std::unordered_set<int>> adj_map;
    for (size_t i = 0; i < local_vertex_ids.size(); ++i) {
        int u = local_vertex_ids[i];
        adj_map[u] = std::unordered_set<int>(
            adjacency_list[i].begin(), 
            adjacency_list[i].end()
        );
    }
    
    num_iterations = 0;
    while (!counts.empty()) {
        // Find current minimum butterfly count
        int min_count = std::min_element(
            counts.begin(), counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->second;

        // Collect all edges with the minimum count (bucket)
        std::vector<std::pair<int, int>> bucket;
        for (const auto& [e, count] : counts) {
            if (count == min_count) {
                bucket.push_back(e);
            }
        }

        // Remove all edges in the bucket
        for (const auto& edge : bucket) {
            peel_order.push_back(edge);
            removed_edges.insert(edge);
            counts.erase(edge);

            // Remove edge from adjacency map
            if (adj_map[edge.first].count(edge.second)) {
                adj_map[edge.first].erase(edge.second);
            }
            if (adj_map[edge.second].count(edge.first)) {
                adj_map[edge.second].erase(edge.first);
            }
        }

        // Find affected edges (edges in wedges/triangles involving removed edges)
        std::unordered_set<std::pair<int, int>, PairHash> affected;
        #pragma omp parallel
        {
            std::unordered_set<std::pair<int, int>, PairHash> private_affected;

            #pragma omp for
            for (size_t i = 0; i < bucket.size(); ++i) {
                auto [u, v] = bucket[i];
                for (int endpoint : {u, v}) {
                    if (adj_map.find(endpoint) == adj_map.end()) continue;

                    for (int neighbor : adj_map[endpoint]) {
                        auto e1 = OrderedPair(endpoint, neighbor);
                        if (!removed_edges.count(e1)) {
                            private_affected.insert(e1);
                        }

                        for (int second_neighbor : adj_map[neighbor]) {
                            if (second_neighbor == endpoint) continue;
                            auto e2 = OrderedPair(neighbor, second_neighbor);
                            if (!removed_edges.count(e2)) {
                                private_affected.insert(e2);
                            }
                        }
                    }
                }
            }

            #pragma omp critical
            affected.insert(private_affected.begin(), private_affected.end());
        }

        // Recompute butterfly counts for affected edges
        std::vector<std::pair<int, int>> affected_vec(affected.begin(), affected.end());
        #pragma omp parallel for
        for (size_t idx = 0; idx < affected_vec.size(); ++idx) {
            auto e = affected_vec[idx];
            if (removed_edges.count(e)) continue;

            int u = e.first, v = e.second;
            if (adj_map[u].size() > adj_map[v].size()) std::swap(u, v);

            int count = 0;
            for (int w : adj_map[u]) {
                if (w == v) continue;
                if (adj_map[v].count(w)) {
                    count += (adj_map[u].size() - 1) * (adj_map[v].size() - 1);
                }
            }

            #pragma omp critical
            counts[e] = count;
        }
        num_iterations++;
    }

    return peel_order;
}

//==========================================================================
// Algorithm 4: Parallel work-efficient butterfly counting per edge

std::unordered_map<std::pair<int, int>, int, PairHash> Graph::count_butterflies_edge() const {
    std::unordered_map<std::pair<int, int>, int, PairHash> butterfly_counts;
    auto wedges = get_wedges();

    // Step 1: Group wedges by sorted endpoints (u1, u2)
    using EndpointPair = std::pair<int, int>;
    std::unordered_map<EndpointPair, std::vector<int>, PairHash> endpoint_groups;

    #pragma omp parallel
    {
        std::unordered_map<EndpointPair, std::vector<int>, PairHash> local_groups;

        #pragma omp for nowait
        for (size_t i = 0; i < wedges.size(); ++i) {
            const auto& wedge = wedges[i];
            // endpoints are wedge.v and wedge.w
            int e1 = std::min(wedge.v, wedge.w);
            int e2 = std::max(wedge.v, wedge.w);
            EndpointPair key(e1, e2);
            // collect the true center: wedge.u
            local_groups[key].push_back(wedge.u);
         }

        #pragma omp critical
        {
            for (const auto& entry : local_groups) {
                auto& global_centers = endpoint_groups[entry.first];
                global_centers.insert(global_centers.end(), entry.second.begin(), entry.second.end());
            }
        }
    }

    // Step 2: Convert map to vector for OpenMP compatibility
    std::vector<std::pair<EndpointPair, std::vector<int>>> groups_vector(
        endpoint_groups.begin(), endpoint_groups.end()
    );

    // Process each group to compute edge counts
    #pragma omp parallel
    {
        std::unordered_map<EndpointPair, int, PairHash> local_counts;

        #pragma omp for nowait
        for (size_t idx = 0; idx < groups_vector.size(); ++idx) {
            const auto& entry = groups_vector[idx];
            const auto& endpoints = entry.first;
            int u1 = endpoints.first;
            int u2 = endpoints.second;
            const auto& centers = entry.second;
            int d = centers.size();

            for (int v : centers) {
                // Normalize edge pairs
                EndpointPair edge1(std::min(u1, v), std::max(u1, v));
                EndpointPair edge2(std::min(u2, v), std::max(u2, v));

                local_counts[edge1] += d - 1;
                local_counts[edge2] += d - 1;
            }
        }

        #pragma omp critical
        {
            for (const auto& [edge, cnt] : local_counts) {
                butterfly_counts[edge] += cnt;
            }
        }
    }
    
    for (auto& [edge, count] : butterfly_counts) {
        count /= 2;  // Fixes overcounting
    }
    

    return butterfly_counts;
}
//==========================================================================

std::vector<int> Graph::peel_vertices_by_butterfly_count(
    const std::unordered_map<int, int>& butterfly_counts,
    int& num_iterations) const
{
    std::unordered_map<int, int> counts = butterfly_counts;
    std::unordered_set<int> removed;
    std::vector<int> peel_order;

    // Convert to map for parallel access
    std::unordered_map<int, std::vector<int>> adj_map;
    for (size_t i = 0; i < local_vertex_ids.size(); ++i) {
        adj_map[local_vertex_ids[i]] = adjacency_list[i];
    }

    num_iterations = 0;

    while (!counts.empty()) {
        // Find the current minimum butterfly count
        int min_butterfly = std::min_element(
            counts.begin(), counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->second;

        // Gather all vertices with this min count
        std::vector<int> bucket;
        for (const auto& [v, count] : counts) {
            if (count == min_butterfly) {
                bucket.push_back(v);
            }
        }

        // Remove all vertices in the current bucket
        for (int v : bucket) {
            peel_order.push_back(v);
            removed.insert(v);
            counts.erase(v);

            // Remove v from adjacency lists of its neighbors
            if (adj_map.find(v) != adj_map.end()) {
                #pragma omp parallel for
                for (size_t idx = 0; idx < adj_map[v].size(); ++idx) {
                    int neighbor = adj_map[v][idx];
                    if (adj_map.find(neighbor) != adj_map.end()) {
                        #pragma omp critical
                        {
                            auto& nl = adj_map[neighbor];
                            nl.erase(std::remove(nl.begin(), nl.end(), v), nl.end());
                        }
                    }
                }
                adj_map.erase(v);
            }
        }

        // Track all affected vertices
        std::unordered_set<int> affected;
        for (int v : bucket) {
            if (adj_map.find(v) == adj_map.end()) continue;

            for (int u : adj_map[v]) {
                if (removed.count(u)) continue;
                affected.insert(u);

                if (adj_map.find(u) != adj_map.end()) {
                    for (int w : adj_map[u]) {
                        if (!removed.count(w)) {
                            affected.insert(w);
                        }
                    }
                }
            }
        }

        std::vector<int> affected_vec(affected.begin(), affected.end());

        // Recompute butterfly counts for affected vertices
        #pragma omp parallel for
        for (size_t idx = 0; idx < affected_vec.size(); ++idx) {
            int u = affected_vec[idx];
            if (removed.count(u) || adj_map.find(u) == adj_map.end()) continue;

            int b_count = 0;
            const auto& neighbors = adj_map[u];
            for (size_t i = 0; i < neighbors.size(); ++i) {
                int n1 = neighbors[i];
                if (adj_map.find(n1) == adj_map.end() || removed.count(n1)) continue;

                for (size_t j = i + 1; j < neighbors.size(); ++j) {
                    int n2 = neighbors[j];
                    if (adj_map.find(n2) != adj_map.end() && !removed.count(n2)) {
                        const auto& n1_neighbors = adj_map[n1];
                        if (std::find(n1_neighbors.begin(), n1_neighbors.end(), n2) != n1_neighbors.end()) {
                            b_count++;
                        }
                    }
                }
            }

            #pragma omp critical
            counts[u] = b_count;
        }

        num_iterations++; // Count one layer of peeling
    }

    return peel_order;
}

//==========================================================================

std::unordered_map<int, int> Graph::count_butterflies_vertex() const 
{
    std::unordered_map<int, int> butterfly_counts;
    std::unordered_map<int, std::unordered_set<int>> adj_map;

    // Build adjacency map
    for (size_t i = 0; i < local_vertex_ids.size(); ++i) 
    {
        int u = local_vertex_ids[i];
        for (int v : adjacency_list[i]) 
        {
            adj_map[u].insert(v);
            adj_map[v].insert(u);
        }
    }

    auto wedges = get_wedges();

    #pragma omp parallel
    {
        std::unordered_map<int, int> local_counts;
        #pragma omp for nowait
        for (size_t i = 0; i < wedges.size(); ++i) 
        {
            const auto& wedge = wedges[i];
            int u = wedge.u;
            int v = wedge.v;
            int w = wedge.w;

            const auto& v_neighbors = adj_map.at(v);
            const auto& w_neighbors = adj_map.at(w);

            for (int x : v_neighbors) 
            {
                if (x != u && w_neighbors.count(x)) 
                {
                    local_counts[u]++;
                    local_counts[v]++;
                    local_counts[w]++;
                    local_counts[x]++;
                }
            }
        }

        #pragma omp critical
        for (const auto& [vertex, count] : local_counts) 
        {
            butterfly_counts[vertex] += count;
        }
    }

    // Adjust counts
    for (auto& [vertex, count] : butterfly_counts) 
    {
        count /= 4;
    }

    return butterfly_counts;
}

std::vector<Wedge> Graph::get_wedges() const 
{
    std::vector<Wedge> wedges;

    #pragma omp parallel
    {
        std::vector<Wedge> local_wedges;
        #pragma omp for nowait
        for (size_t i = 0; i < local_vertex_ids.size(); ++i) 
        {
            int u = local_vertex_ids[i];
            const auto& neighbors = adjacency_list[i];

            std::vector<int> sorted_neighbors(neighbors.begin(), neighbors.end());
            std::sort(sorted_neighbors.begin(), sorted_neighbors.end());

            for (size_t j = 0; j < sorted_neighbors.size(); ++j) 
            {
                int v = sorted_neighbors[j];
                for (size_t k = j + 1; k < sorted_neighbors.size(); ++k) 
                {
                    int w = sorted_neighbors[k];
                    local_wedges.emplace_back(u, v, w);
                }
            }
        }

        #pragma omp critical
        wedges.insert(wedges.end(), local_wedges.begin(), local_wedges.end());
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
