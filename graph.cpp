#include "graph.h"
#include <mpi.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>
#include <queue>
#include <utility>

//==========================================================================

std::vector<std::pair<int, int>> Graph::peel_edges_by_butterfly_count(const std::unordered_map<std::pair<int, int>, int, PairHash>& edge_counts, int& num_iterations) const 
{
    std::unordered_map<std::pair<int, int>, int, PairHash> counts = edge_counts;
    std::unordered_set<std::pair<int, int>, PairHash> removed_edges;
    std::vector<std::pair<int, int>> peel_order;

    std::unordered_map<int, std::unordered_set<int>> adj_map;
    for (size_t i = 0; i < local_vertex_ids.size(); ++i) //build adjacency map for efficient edge removal
    {
        int u = local_vertex_ids[i];
        adj_map[u] = std::unordered_set<int>(adjacency_list[i].begin(), adjacency_list[i].end());
    }
    
    num_iterations = 0;
    while (!counts.empty()) 
    {
        //find current minimum butterfly count in remaining edges:
        int min_count = std::min_element(counts.begin(), counts.end(),[](const auto& a, const auto& b) { return a.second < b.second; })->second;

        //collect all edges with the minimum count of butterflies (bucket)
        std::vector<std::pair<int, int>> bucket;
        for (const auto& [e, count] : counts) 
        {
            if (count == min_count) 
            {
                bucket.push_back(e);
            }
        }

        for (const auto& edge : bucket) //remove all edges in that bucket
        {
            peel_order.push_back(edge);
            removed_edges.insert(edge);
            counts.erase(edge);

            //remove edge from adjacency map
            if (adj_map[edge.first].count(edge.second))
            {
                adj_map[edge.first].erase(edge.second);
            }
            if (adj_map[edge.second].count(edge.first))
            {
                adj_map[edge.second].erase(edge.first);
            }
        }

        //find affected edges by the removal of the bucket
        std::unordered_set<std::pair<int, int>, PairHash> affected;
        #pragma omp parallel
        {
            std::unordered_set<std::pair<int, int>, PairHash> private_affected;

            #pragma omp for
            for (size_t i = 0; i < bucket.size(); ++i) 
            {
                auto [u, v] = bucket[i];
                for (int endpoint : {u, v}) //check neighbours of both endpoints
                {
                    if (adj_map.find(endpoint) == adj_map.end()) continue;

                    for (int neighbor : adj_map[endpoint]) 
                    {
                        auto e1 = OrderedPair(endpoint, neighbor);
                        if (!removed_edges.count(e1)) 
                        {
                            private_affected.insert(e1);
                        }

                        for (int second_neighbor : adj_map[neighbor]) //check second-level neighbours
                        {
                            if (second_neighbor == endpoint) continue;
                            auto e2 = OrderedPair(neighbor, second_neighbor);
                            if (!removed_edges.count(e2)) 
                            {
                                private_affected.insert(e2);
                            }
                        }
                    }
                }
            }

            #pragma omp critical
            affected.insert(private_affected.begin(), private_affected.end());
        }

        //recompute butterfly counts for affected edges
        std::vector<std::pair<int, int>> affected_vec(affected.begin(), affected.end());
        #pragma omp parallel for
        for (size_t idx = 0; idx < affected_vec.size(); ++idx) 
        {
            auto e = affected_vec[idx];
            if (removed_edges.count(e)) continue;

            int u = e.first, v = e.second;
            if (adj_map[u].size() > adj_map[v].size()) std::swap(u, v); //process smaller adjacency list first

            int count = 0;
            for (int w : adj_map[u]) //count common neighbours to compute butterfly counts
            {
                if (w == v) continue;
                if (adj_map[v].count(w)) 
                {
                    count += (adj_map[u].size() - 1) * (adj_map[v].size() - 1);
                }
            }

            #pragma omp critical
            counts[e] = count;
        }
        num_iterations++; //count one layer of peeling
    }

    return peel_order;
}

//==========================================================================

std::unordered_map<std::pair<int, int>, int, PairHash> Graph::count_butterflies_edge() const 
{
    std::unordered_map<std::pair<int, int>, int, PairHash> butterfly_counts;
    auto wedges = get_wedges(); //get all wedges in (sub)graph

    //step 1: group wedges by sorted endpoints (u1, u2)
    using EndpointPair = std::pair<int, int>;
    std::unordered_map<EndpointPair, std::vector<int>, PairHash> endpoint_groups;

    #pragma omp parallel //parallel grouping
    {
        std::unordered_map<EndpointPair, std::vector<int>, PairHash> local_groups;

        #pragma omp for nowait
        for (size_t i = 0; i < wedges.size(); ++i) 
        {
            const auto& wedge = wedges[i];
            //endpoints are wedge.v and wedge.w
            int e1 = std::min(wedge.v, wedge.w);
            int e2 = std::max(wedge.v, wedge.w);
            EndpointPair key(e1, e2);
            //collect the true center: wedge.u
            local_groups[key].push_back(wedge.u);
         }

        #pragma omp critical //merge local results
        {
            for (const auto& entry : local_groups) 
            {
                auto& global_centers = endpoint_groups[entry.first];
                global_centers.insert(global_centers.end(), entry.second.begin(), entry.second.end());
            }
        }
    }

    //convert map to vector for openmp compatibility
    std::vector<std::pair<EndpointPair, std::vector<int>>> groups_vector(endpoint_groups.begin(), endpoint_groups.end());

    //step 2: compute butterfly contributions for each edge pair
    #pragma omp parallel
    {
        std::unordered_map<EndpointPair, int, PairHash> local_counts;

        #pragma omp for nowait
        for (size_t idx = 0; idx < groups_vector.size(); ++idx) 
        {
            const auto& entry = groups_vector[idx];
            const auto& endpoints = entry.first;
            int u1 = endpoints.first;
            int u2 = endpoints.second;
            const auto& centers = entry.second;
            int d = centers.size(); //num of wedges between u1-u2

            for (int v : centers) //distribute counts to participating edges
            {
                //normalize edge pairs to avoid duplicates
                EndpointPair edge1(std::min(u1, v), std::max(u1, v));
                EndpointPair edge2(std::min(u2, v), std::max(u2, v));

                local_counts[edge1] += d - 1;
                local_counts[edge2] += d - 1;
            }
        }

        #pragma omp critical //merge thread-local counts
        {
            for (const auto& [edge, cnt] : local_counts) 
            {
                butterfly_counts[edge] += cnt;
            }
        }
    }
    
    for (auto& [edge, count] : butterfly_counts) //fix overcounting: each butterfly is counted 2 times per edge
    {
        count /= 2;
    }
    
    return butterfly_counts;
}

//==========================================================================

std::vector<int> Graph::peel_vertices_by_butterfly_count(const std::unordered_map<int, int>& butterfly_counts, int& num_iterations) const
{
    std::unordered_map<int, int> counts = butterfly_counts;
    std::unordered_set<int> removed;
    std::vector<int> peel_order;

    //build adjacency map for efficient modifications
    std::unordered_map<int, std::vector<int>> adj_map;
    for (size_t i = 0; i < local_vertex_ids.size(); ++i) 
    {
	adj_map[local_vertex_ids[i]] = adjacency_list[i];
    }

    num_iterations = 0;

    while (!counts.empty()) 
    {
        //find current minimum butterfly count in remaining vertices:
        int min_butterfly = std::min_element(counts.begin(), counts.end(), [](const auto& a, const auto& b) { return a.second < b.second; })->second;

        //collect all vertices with this min count
        std::vector<int> bucket;
        for (const auto& [v, count] : counts) 
        {
            if (count == min_butterfly) 
            {
                bucket.push_back(v);
            }
        }

        //remove all vertices in the current bucket
        for (int v : bucket) 
        {
            peel_order.push_back(v);
            removed.insert(v);
            counts.erase(v);

            //remove vertex from neighbors' adjacency lists
            if (adj_map.find(v) != adj_map.end()) 
            {
                #pragma omp parallel for
                for (size_t idx = 0; idx < adj_map[v].size(); ++idx) 
                {
                    int neighbor = adj_map[v][idx];
                    if (adj_map.find(neighbor) != adj_map.end()) 
                    {
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

        //identify vertices affected by removal of vertices in the bucket
        std::unordered_set<int> affected;
        for (int v : bucket) 
        {
            if (adj_map.find(v) == adj_map.end()) continue;

            for (int u : adj_map[v]) 
            {
                if (removed.count(u)) continue;
                affected.insert(u);

                if (adj_map.find(u) != adj_map.end()) //check second-level neighbours
                {
                    for (int w : adj_map[u]) 
                    {
                        if (!removed.count(w)) 
                        {
                            affected.insert(w);
                        }
                    }
                }
            }
        }

        std::vector<int> affected_vec(affected.begin(), affected.end());

        //recompute butterfly counts for affected vertices
        #pragma omp parallel for
        for (size_t idx = 0; idx < affected_vec.size(); ++idx) 
        {
            int u = affected_vec[idx];
            if (removed.count(u) || adj_map.find(u) == adj_map.end()) continue;

            int b_count = 0;
            const auto& neighbors = adj_map[u];
            for (size_t i = 0; i < neighbors.size(); ++i) //count butterflies via common neighbours
            {
                int n1 = neighbors[i];
                if (adj_map.find(n1) == adj_map.end() || removed.count(n1)) continue;

                for (size_t j = i + 1; j < neighbors.size(); ++j) 
                {
                    int n2 = neighbors[j];
                    if (adj_map.find(n2) != adj_map.end() && !removed.count(n2)) //check if n1 and n2 are connected
                    {
                        const auto& n1_neighbors = adj_map[n1];
                        if (std::find(n1_neighbors.begin(), n1_neighbors.end(), n2) != n1_neighbors.end()) 
                        {
                            b_count++;
                        }
                    }
                }
            }
            #pragma omp critical
            counts[u] = b_count;
        }
        num_iterations++; //count one layer of peeling
    }

    return peel_order;
}

//==========================================================================

std::unordered_map<int, int> Graph::count_butterflies_vertex() const 
{
    std::unordered_map<int, int> butterfly_counts;
    std::unordered_map<int, std::unordered_set<int>> adj_map;

    //build adjacency map
    for (size_t i = 0; i < local_vertex_ids.size(); ++i) 
    {
        int u = local_vertex_ids[i];
        for (int v : adjacency_list[i]) 
        {
            adj_map[u].insert(v);
            adj_map[v].insert(u);
        }
    }

    auto wedges = get_wedges(); //get all wedges in the subgraph

    #pragma omp parallel //parallel processing of wedges
    {
        std::unordered_map<int, int> local_counts;
        #pragma omp for nowait
        for (size_t i = 0; i < wedges.size(); ++i) 
        {
            const auto& wedge = wedges[i];
            int u = wedge.u; //center of the wedge
            int v = wedge.v;
            int w = wedge.w;

	    //gind common neighbors between v and w to complete butterflies
            const auto& v_neighbors = adj_map.at(v);
            const auto& w_neighbors = adj_map.at(w);

            for (int x : v_neighbors) 
            {
                if (x != u && w_neighbors.count(x)) //all 4 vertices (u, v, w, x) form a butterfly
                {
                    local_counts[u]++;
                    local_counts[v]++;
                    local_counts[w]++;
                    local_counts[x]++;
                }
            }
        }

        #pragma omp critical //merge thread-local counts
        for (const auto& [vertex, count] : local_counts) 
        {
            butterfly_counts[vertex] += count;
        }
    }
    
    for (auto& [vertex, count] : butterfly_counts) //adjust counts: each butterfly is counted 4 times (once per vertex)
    {
        count /= 4;
    }

    return butterfly_counts;
}

std::vector<Wedge> Graph::get_wedges() const //generate all wedges (u-v-w paths) in the graph
{
    std::vector<Wedge> wedges;

    #pragma omp parallel //parallel wedge generation
    {
        std::vector<Wedge> local_wedges;
        #pragma omp for nowait
        for (size_t i = 0; i < local_vertex_ids.size(); ++i) 
        {
            int u = local_vertex_ids[i];
            const auto& neighbors = adjacency_list[i];

	    //sort neighbors for consistent wedge ordering
            std::vector<int> sorted_neighbors(neighbors.begin(), neighbors.end());
            std::sort(sorted_neighbors.begin(), sorted_neighbors.end());

            for (size_t j = 0; j < sorted_neighbors.size(); ++j) //generate all (v, w) pairs where v < w
            {
                int v = sorted_neighbors[j];
                for (size_t k = j + 1; k < sorted_neighbors.size(); ++k) 
                {
                    int w = sorted_neighbors[k];
                    local_wedges.emplace_back(u, v, w); //store wedge u-v-w
                }
            }
        }

        #pragma omp critical //merge local results
        wedges.insert(wedges.end(), local_wedges.begin(), local_wedges.end());
    }

    return wedges;
}

//==========================================================================

void Graph::loadPartition(const std::vector<idx_t>& metis_local_vertices, const std::vector<std::vector<idx_t>>& global_adj) //load partition data into the graph object
{
    //clear any existing data
    local_vertex_ids.clear();
    adjacency_list.clear();

    //convert metis idx_t (vertex ids) to int and populate local vertices
    for (idx_t global_id : metis_local_vertices) 
    {
        local_vertex_ids.push_back(static_cast<int>(global_id));
    }

    //build adjacency list with global vertex ids
    for (int local_idx = 0; local_idx < local_vertex_ids.size(); ++local_idx) 
    {
        int global_id = local_vertex_ids[local_idx];
        std::vector<int> neighbors;
        for (idx_t neighbor : global_adj[global_id]) 
        {
            neighbors.push_back(static_cast<int>(neighbor));
        }
        adjacency_list.push_back(neighbors);
    }
}

//==========================================================================

void Graph::preprocess() 
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //step 1: compute local vertex degrees
    int local_vertex_count = local_vertex_ids.size();
    std::vector<int> local_degrees(local_vertex_count);
    for (int i = 0; i < local_vertex_count; ++i) 
    {
        local_degrees[i] = adjacency_list[i].size();
    }

    //step 2: prepare send buffer (vertex id, degree pairs) for mpi exchange
    std::vector<int> send_buffer;
    send_buffer.reserve(2 * local_vertex_count);
    for (int i = 0; i < local_vertex_count; ++i) 
    {
        send_buffer.push_back(local_vertex_ids[i]);
        send_buffer.push_back(local_degrees[i]);
    }

    //step 3: gather send counts from all processes
    std::vector<int> recv_counts(world_size);
    int send_count = send_buffer.size();
    MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Step 4: Compute displacements for MPI_Allgatherv
    std::vector<int> displacements(world_size, 0);
    for (int i = 1; i < world_size; ++i) 
    {
        displacements[i] = displacements[i-1] + recv_counts[i-1];
    }

    //step 5: Allgatherv to gather all vertex-degree pairs
    int total_recv_count = displacements.back() + recv_counts.back();
    std::vector<int> recv_buffer(total_recv_count);
    MPI_Allgatherv(send_buffer.data(), send_count, MPI_INT, recv_buffer.data(), recv_counts.data(), displacements.data(), MPI_INT, MPI_COMM_WORLD);

    //step 6: parse into global_vertex_degrees
    std::vector<std::pair<int, int>> global_vertex_degrees;
    for (size_t i = 0; i < recv_buffer.size(); i += 2) 
    {
        int id = recv_buffer[i];
        int degree = recv_buffer[i+1];
        global_vertex_degrees.emplace_back(id, degree);
    }

    //step 7: sort by degree (descending) and vertex id (ascending for ties)
    std::sort(global_vertex_degrees.begin(), global_vertex_degrees.end(), 
    [](const std::pair<int,int>& a, const std::pair<int,int>& b) -> bool
          {
              //if degrees equal, tie-break on vertex id ascending:
              if (a.second == b.second)
                  return a.first < b.first;
              //otherwise sort by descending degree:
              return a.second > b.second;
          });


    //step 8: create global vertex id to rank map
    std::unordered_map<int, int> vertex_id_to_rank;
    for (size_t i = 0; i < global_vertex_degrees.size(); ++i) 
    {
        vertex_id_to_rank[global_vertex_degrees[i].first] = i;
    }

    //step 9: rename local vertex ids to their global ranks
    for (int i = 0; i < local_vertex_count; ++i) 
    {
        local_vertex_ids[i] = vertex_id_to_rank[local_vertex_ids[i]];
    }

    //step 10: rename neighbors in adjacency lists and sort them
    #pragma omp parallel for
    for (size_t i = 0; i < adjacency_list.size(); ++i) 
    {
        auto& neighbors = adjacency_list[i];
        for (auto& neighbor : neighbors) 
        {
            neighbor = vertex_id_to_rank[neighbor];
        }
        std::sort(neighbors.begin(), neighbors.end(), std::greater<int>());
    }

    //step 11: compute deg_u for each local vertex
    deg_u.resize(local_vertex_count);
    #pragma omp parallel for
    for (size_t i = 0; i < adjacency_list.size(); ++i) 
    {
        int u_rank = local_vertex_ids[i];
        auto& neighbors = adjacency_list[i];
        auto it = std::upper_bound(neighbors.begin(), neighbors.end(), u_rank, std::greater<int>());
        deg_u[i] = std::distance(neighbors.begin(), it);
    }
}

//==========================================================================
