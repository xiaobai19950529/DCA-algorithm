We use the graph generator by NetworkX to generate the power-law distribution graphs, where we set the parameters to be (700, 3, 0.5) for pow_s. The thresholds and values are chosen uniformly at random from integers in [1,10]. The edge weights are uniformly chosen from [0.3,1]. Beside, we set the random seed = 20200904.

pow_l_MILP is the exact defend result but the running time is very long cause its time complexity is exponential.

pow_l_pruneMILP is also the exact defend result but the running time will be less because we use the defend result of LP as the lowerbound of MILP.

pow_l_approximation.py is the proposed approximation method by ours. It can get very approximate solution but only need a bit time because its time complexity is polynomial.

pow_l_optimal_result=0 is the program to get minimum resource when there is no loss(every node is well-defended).

.png contains the graph structure and the degree statistic information. 