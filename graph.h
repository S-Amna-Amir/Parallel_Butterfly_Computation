// graph.h
#include <vector>
#include <string>

class Graph 
{
	private:
	    std::vector<int> offsets_U, edges_U;  // CSR for U->V
	    std::vector<int> offsets_V, edges_V;  // CSR for V->U
	    int num_vertices_U;

	public:
	    Graph() = default;
	    Graph(const std::string& filename);
	    
	    // Accessors
	    const int* neighbors(int vertex) const;
	    int degree(int vertex) const;
	    bool is_vertex_in_U(int vertex) const;
	    int count_common_neighbors(int u, int w) const;
	    
	    // Stats
	    int num_vertices() const;
	    int num_edges() const;
	    
	    //mpi broadcast
	    void broadcast(int root_rank); 
};
