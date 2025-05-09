# 🦋 Parallel Butterfly Computations 🦋 

This project implements parallel algorithms for butterfly counting and peeling in bipartite graphs, based on the research paper **"Parallel Algorithms for Butterfly Computations" by Jessica Shi and Julian Shun**.

---

## Team Members
- Amna  
- Maha

## Course
Parallel and Distributed Computing (PDC)

---

## What the Code Does

- Loads a bipartite graph from a text file.
- Partitions the graph using **METIS** and distributes it across processes using **MPI**.
- Performs **butterfly counting** (subgraph patterns of 4 nodes) for:
  - Vertices
  - Edges
- Applies **peeling algorithms** to iteratively remove nodes or edges with the least butterfly count.
- Uses **OpenMP** to parallelize counting and peeling steps.
- Uses **MPI** to combine results across processes.

---

## How to Run

### 1. Compile
```bash
mpicxx -std=c++17 -fopenmp main.cpp graph.cpp -lmetis -o butterfly
```

### 2. Run
```bash
export OMP_NUM_THREADS=4
mpirun -np 2 ./butterfly input.txt 2
```

Replace `input.txt` with your bipartite graph file.

---

## Example Output
```text
---> Global results <---
total butterfly count: 1234
total edge peeling iterations: 27
total vertex peeling iterations: 31
total edges: 4567
|U| = 123
|V| = 321
```

---

## Reference
> Shi, J., & Shun, J. (2019, July 19). *Parallel Algorithms for Butterfly Computations*. arXiv. https://arxiv.org/abs/1907.08607

---

## Presentation Slides

https://www.canva.com/design/DAGlLxFQHdI/JDQrr_PWkP8MxJ-_Eb8IEg/edit?utm_content=DAGlLxFQHdI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## Report

https://docs.google.com/document/d/1je0lmq5zgEo0O5QQV-IYuXpn6faTKrjA6MfHRA1kIIQ/edit?tab=t.0

---
