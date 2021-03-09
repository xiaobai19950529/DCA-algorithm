We  generate  the  dataset with n = 200 and p = 0.04 using the graph generator NetworkX, where there is an edge between each pair of nodes independently with probability p. The thresholds and values are chosen uniformly at random from integers in [1,10]. The edge weights are uniformly chosen from [0.3,1]. Beside, we set the random seed = 20200905.

random_gnp_MILP is the exact defend result but the running time is very long cause its time complexity is exponential.

random_gnp_pruneMILP is also the exact defend result but the running time will be less because we use the defend result of LP as the lowerbound of MILP.

random_gnp_approximation.py is the proposed approximation method by ours. It can get very approximate solution but only need a bit time because its time complexity is polynomial.

random_gnp_optimal_result=0 is the program to get minimum resource when there is no loss(every node is well-defended).

.png contains the graph structure and the degree statistic information. 