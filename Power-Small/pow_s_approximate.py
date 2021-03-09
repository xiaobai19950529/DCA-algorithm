#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import matplotlib.pyplot as plt
import collections
import time
import math
import numpy as np
from numpy.linalg import matrix_power
from scipy import sparse
import gurobipy as gp
from gurobipy import GRB
from scipy import sparse

start_all = time.time()

seed = 20200904
np.random.seed(seed)
np.set_printoptions(precision=2)

n = 400
ne = 4
p = 0.5

num_node = n
ew_bnds = [0.3, 1]
values = np.random.randint(1, 10, n)
lbs = np.array(values)
vals = values

k = 2

def find_runs(x):

    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        run_values = x[loc_run_start]

        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths
    
def time_diff(s, e, str_name):
    dur = e - s
    hours, seconds = divmod(dur, 3600)
    minutes, seconds = divmod(seconds, 60)
    total_time = "{:02.0f}:{:02.0f}:{:02.0f}".format(hours, minutes, seconds)
    print('Total {} time taken: {} ({} seconds)\n'.format(str_name, total_time, dur))


G = nx.powerlaw_cluster_graph(n, ne, p, seed=seed)

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)

plt.axes([0.4, 0.4, 0.5, 0.5])
Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
pos = nx.spring_layout(G)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size=20)
nx.draw_networkx_edges(G, pos, alpha=0.4)

plt.show()


print("ne = {}".format(ne))
print("node = {}, edge = {}".format(n, sum(degree_sequence)/2))


r_R = 0.5
resource_rate = r_R
R = float(lbs.sum() * r_R)

a_mtx = nx.adjacency_matrix(G)
print("Ajacency matrix shape:", a_mtx.shape)

a_mtx_csr = a_mtx.tocsr()
a_mtx_nonzero = a_mtx_csr.nonzero()

run_values, run_starts, run_lengths = find_runs(a_mtx_nonzero[0])

nbr_dict = {}

for i in range(len(run_values) - 1):
    i_start = run_starts[i]
    i_end = run_starts[i + 1]
    
    nbr_dict[run_values[i]] = list(a_mtx_nonzero[1][i_start:i_end])
    
nbr_dict[run_values[-1]] = list(a_mtx_nonzero[1][run_starts[-1]:])


np.random.seed(seed)
w_rand = np.random.rand(a_mtx_nonzero[0].shape[0]) * (ew_bnds[1] - ew_bnds[0]) + ew_bnds[0]
w_mtx = sparse.csr_matrix((w_rand, a_mtx_nonzero), shape=(n, n))
w_mtx = (w_mtx + w_mtx.T)/2

k_nbrs = sparse.diags(np.ones(n)).tocsr()
if k > 0:
    k_nbrs += a_mtx_csr
    
    if k > 1:
        k_pow = sparse.csr_matrix(a_mtx_csr)
        for i in range(2, k + 1):
            k_pow = k_pow.dot(a_mtx_csr)
            k_nbrs += k_pow


k_nbrs_nonzero = k_nbrs.nonzero()

run_values, run_starts, run_lengths = find_runs(k_nbrs_nonzero[0])

knbr_dict = {}

for i in range(len(run_values) - 1):
    i_start = run_starts[i]
    i_end = run_starts[i + 1]
    
    knbr_dict[run_values[i]] = list(k_nbrs_nonzero[1][i_start:i_end])
    
knbr_dict[run_values[-1]] = list(k_nbrs_nonzero[1][run_starts[-1]:])


x_keys = []
b_keys = []
b_dict = {}

for i, items in knbr_dict.items():
    arr_i = np.ones((len(items), 1)) * i
    arr_v = np.array(items)[:, np.newaxis]
    arr_tmp = np.concatenate((arr_i, arr_v), axis=1)
    x_keys += list(map(tuple, arr_tmp))
    
    i_keys = []
    i_dict = []
    for j in items:
        n_nbrs = a_mtx_csr[j, :].sum()
        if n_nbrs == 0:
            continue
        j_nbrs = np.array(nbr_dict[j])[:, np.newaxis]
        arr_i = np.ones((n_nbrs, 1)) * i
        arr_j = np.ones((n_nbrs, 1)) * j
        arr_tmp = np.concatenate((arr_i, j_nbrs, arr_j), axis=1)
        arr_dict = np.concatenate((j_nbrs, arr_j), axis=1)
        i_keys += list(map(tuple, arr_tmp))
        i_dict += list(map(tuple, arr_dict))
                
    b_keys += i_keys
    b_dict[i] = list(i_dict)
    
    print('{}/{}'.format(i + 1, n), end="\r", flush=True)


# LP algorithm
def LP_tau(resource):
    model = gp.Model('ADG')
    obj = model.addVar(vtype=GRB.CONTINUOUS, name="obj")
    x = model.addVars(x_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='x')
    r = model.addVars(range(num_node), vtype=GRB.CONTINUOUS, name='r')
    b = model.addVars(b_keys, vtype=GRB.CONTINUOUS, name='b')
    
    model.addConstr(r.sum() <= resource, name='total resource sum')
    model.addConstrs((b[(i, j, k)] <= w_mtx[j, k] * r[j] for (i, j, k) in b_keys), name='borrowing limitations')
    model.addConstrs((gp.quicksum((np.ones_like(knbr_dict[i]) - x.select(i, knbr_dict[i])) * vals[knbr_dict[i]]) <= obj
                  for i in range(num_node)), name='minimize maximum')
    for ra in range(num_node):
        model.addConstrs((r[k] - gp.quicksum(b.select(ra, k, '*')) + gp.quicksum(b.select(ra, '*', k))
                      >= lbs[k] * x[(ra, k)] for k in knbr_dict[ra]),
                     name='reallocation results for node {}'.format(i))
        model.addConstrs((gp.quicksum(b.select(ra, k, '*')) <= r[k] for k in range(num_node)),
                     name='total out-borrowings')
        
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    
    return model, r, x
    
    
def remove_x_constrain(model):
    for i in range(num_node):
        for j in knbr_dict[i]:
            obj_constrain = model.getConstrByName('x_fixed for node {}_{}'.format(i,j))
            model.remove(obj_constrain)
    return model

def remove_r_constrain(model):
    obj1_constrain = model.getConstrByName('total resource sum')
    model.remove(obj1_constrain)
    return model

def add_x_constrain(model, x, x_vals_tau):
    for i in range(num_node):
        for j in knbr_dict[i]:
            if x_vals_tau[i][j] == 1:
                model.addConstr(x[(i,j)] >= x_vals_tau[i][j], name='x_fixed for node {}_{}'.format(i,j))
            else:
                model.addConstr(x[(i,j)] <= x_vals_tau[i][j], name='x_fixed for node {}_{}'.format(i,j))
    return model

def add_r_constrain(model, resource, r):
    model.addConstr(r.sum() <= resource, name='total resource sum')
    return model


# defend_result of rounding(tau)
def compute_result(x_vals_tau):
    max_obj = 0
    for i in range(num_node):  # attack a node with maximum payoff
        obj = 0
        for j in knbr_dict[i]:
            obj += (1 - x_vals_tau[i][j]) * vals[j]
        if obj > max_obj:
            max_obj = obj
    return max_obj


# bi-criteria approximation algorithm

resource_rate = 0.5
alphas = [0.9, 1]  # this is epsilon in paper
print(alphas)
R = sum(lbs) * resource_rate  # sum(lbs) * 0.5

defend_results = []
time_t = []
r_all = []
b_all = []
defend_results_LP = []
defend_result_milp = []
defend_results_fix_alpha = []
isModeled = 0
taus = []
x_isadded = 0


for alpha in alphas:
    start_time = time.time()
    x_change = np.zeros((num_node, num_node))
    if(isModeled == 0):  # in order to reduce the time of preloading constraint 
        model, r, x, b = LP_tau(alpha * R)  # the OPT of LP and this OPT is the lowerbound of MILP
        pre_end = time.time()
        first_time = pre_end - start_time 
        isModeled = 1
        xx = []
        # reserve initial value of x for following rounding
        for p in range(num_node):
            for q in knbr_dict[p]:
                x_change[p][q] = x[(p, q)].X
                xx.append(x[(p, q)].X)
        
    else:
        if x_isadded == 1:
            model = remove_x_constrain(model)
            x_isadded = 0
        model = remove_r_constrain(model)
        model = add_r_constrain(model, alpha * R, r)
        model.update()
        model.optimize()
    
        xx = []
        for var in model.getVars():
            if var.varName[0] == 'x':
                xx.append(var.X)

        idx = 0
        for p in range(num_node):
            for q in knbr_dict[p]:
                x_change[p][q] = xx[idx]
                idx += 1


    time_LP = model.Runtime
    
    # In experiment, we found if feasibility LP is not feasible, the time is very long, about 10*time_LP even more,
    # So in this location, we add the timelimit to quick exit and run the next loop to accelerate. After multiple experiments,
    # we found this trick method is effective and efficient. Besides, we can ensure the robustness of running time
    # If you choose this trick method, you will get better running time than the running time of approximation in paper Table-3
    if time_LP > 20:
        model.setParam('TimeLimit', 3*time_LP) 

    defend_result_LP = model.getObjective().getValue()
    
    # round(alpha)
    x_vals_tau = copy.deepcopy(x_change)
    x_vals_tau[x_vals_tau > alpha] = 1
    x_vals_tau[x_vals_tau <= alpha] = 0
    
    defend_result_fix_alpha = compute_result(x_vals_tau)  # defend result of round(alpha)
    defend_results_fix_alpha.append(defend_result_fix_alpha)
    print("alpha = {}, defend_result_round = {}".format(alpha, defend_result_fix_alpha))

    f = 0
    xx_new = []
    for var in xx:
        if var > 0 and var < 1:
            xx_new.append(var) 

    if len(xx_new) == 0:  # if all x value is 0 or 1, LP = MILP, need not round
        tau_low = 0
        tau_high = 0
        f = 2
        tau = 0
        print("tau = 0")
    else:
        tau_low = 0
        tau_high = max(xx_new)
        print("tau_high = {}".format(tau_high))
        f = 3
        
    if alpha == 1:
        obj_model = model.getVarByName('obj')
        obj_model.lb = defend_result_LP
        model.update()

    cnt = 0  
    model = remove_r_constrain(model)
    model = add_r_constrain(model, R, r)
    mark = 0
    # dichotomy to find the minimal feasible tau
    while(tau_high - tau_low >= 0.05):   
        cnt += 1 
        tau = (tau_low + tau_high) / 2
        # round(tau)
        x_vals_tau = copy.deepcopy(x_change)
        x_vals_tau[x_vals_tau > tau] = 1 
        x_vals_tau[x_vals_tau <= tau] = 0 
        # verify the feasibility of round(tau)
        if x_isadded == 0: 
            model = add_x_constrain(model, x, x_vals_tau)
            first = 0
            x_isadded = 1
        else:
            model = remove_x_constrain(model)
            model = add_x_constrain(model, x, x_vals_tau)
        model.update()
        model.optimize()
        time_MILP = model.Runtime
        print("cnt = %.2f, verify time = %.3f" % (cnt, time_MILP))
        status = model.Status
        if status == 3 or status == 9:  # Infeasible or TimeLimit = Infeasible
            print("tau = %.2f is infeasible" % tau)
            tau_low = tau
            f = 0
        elif status == 2:
            print("tau = %.2f is feasible" % tau)
            tau_high = tau
            f = 1
            defend_result_curr = model.getObjective().getValue()
            mark = 1  # exist at least one feasible tau
            
    if mark == 0:
        f = 3  
        
    if f == 3:  # if the tau_high is too small to reach 0.05(the bound of dichotomy), we need tau = tau_high to ensure feasibility
        tau = tau_high 
        x_vals_tau = copy.deepcopy(x_change)
        x_vals_tau[x_vals_tau > tau] = 1 
        x_vals_tau[x_vals_tau <= tau] = 0 
        
        if x_isadded == 1:
            model = remove_x_constrain(model)
        model = add_x_constrain(model, x, x_vals_tau)
        model.update()
        model.optimize()
        defend_result_MILP = model.getObjective().getValue()
    
    if f == 0 or f == 1:  
        defend_result_MILP = defend_result_curr
    
    if f == 0:  # if the final loop is infeasible, recover x_vals_tau to last feasible rounding
        tau = tau_high
        x_vals_tau = copy.deepcopy(x_change)
        x_vals_tau[x_vals_tau > tau] = 1 
        x_vals_tau[x_vals_tau <= tau] = 0 

    if f == 2:  
        defend_result_MILP = defend_result_LP
        
    defend_result_fix = compute_result(x_vals_tau) 
    
    print("total cnt = %d，alpha = %.2f, tau = %.2f, defend_result = %.2f, time = %.3f" % (cnt, alpha, tau, defend_result_fix, time.time() - start_time))
    time_t.append(time.time() - start_time)
    taus.append(tau)
    defend_results.append(defend_result_fix) 
    defend_result_milp.append(defend_result_MILP) 
    defend_results_LP.append(defend_result_LP)
    r_all.append(r)  # allocation stategy
    b_all.append(b)  # reallocation

for i in range(len(defend_results)):
    print("resource = %.2f, defend_result = %.2f, defend_result_LP = %.3f, defend_result_fix_alpha = %.3f" 
                % (alphas[i] * R, defend_results[i], defend_results_LP[i], defend_results_fix_alpha[i]))

t = 0
time_t[0] -= first_time  # the first time contains the time of preload constrain
for i in range(len(time_t)):
    print("alpha = %.2f, time = %.3f, final tau = %.2f" % (alphas[i], time_t[i], taus[i]))
    t += time_t[i]

# log the final results：
min_index = np.argmin(defend_results)
allocation_strategy = r_all[min_index]
reallocation_strategy = b_all[min_index]
defend_result = min(defend_results)
print("Final defend_result = %.3f" % defend_result)

end_time = time.time()
time_diff(start_all, pre_end, "preprocessing") # preprocessing time
time_diff(0,t, "optimization")  # running time 
time_diff(start_all, end_time, "")  # total time

