.npz represents the adjacent matrix

In this case, we set weight = 1 for all edge , value = lowerbound = 1 for all nodes.

fb_MILP is the exact defend result but the running time is very long cause its time complexity is exponential.

fb_pruneMILP is also the exact defend result but the running time will be less because we use the defend result of LP as the lowerbound of MILP.

fb_approximation.py is the proposed approximation method by ours. It can get very approximate solution but only need a bit time because its time complexity is polynomial.

fb_optimal_result=0 is the program to get minimum resource when there is no loss(every node is well-defended).

.png contains the graph structure and the degree statistic information. 