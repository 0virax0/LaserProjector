from postman_problems.solver import cpp
from postman_problems.stats import calculate_postman_solution_stats

# find CPP solution
circuit, graph = cpp(edgelist_filename='graph.csv', start_node='A')

# print solution route
for e in circuit:
    print(e)

# print solution summary stats
for k, v in calculate_postman_solution_stats(circuit).items():
    print(k, v)
