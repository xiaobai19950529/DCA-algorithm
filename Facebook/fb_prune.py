#!/usr/bin/env python
# coding: utf-8

import time
import math
import numpy as np
from numpy.linalg import matrix_power
from scipy import sparse
import gurobipy as gp
from gurobipy import GRB

start_all = time.time()

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

k = 2


a_mtx = sparse.load_npz('fb_mtx_a_n=600_e=4638.npz')
num_node = a_mtx.shape[0]

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


w_mtx = sparse.csr_matrix(a_mtx)


k_nbrs = sparse.diags(np.ones(num_node), dtype=np.uint16).tocsr()
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
    arr_i = np.ones((len(items), 1), dtype=np.uint16) * i
    arr_v = np.array(items, dtype=np.uint16)[:, np.newaxis]
    arr_tmp = np.concatenate((arr_i, arr_v), axis=1)
    x_keys += list(map(tuple, arr_tmp))
    
    i_keys = []
    i_dict = []
    for j in items:
        if a_mtx_csr[j, :].sum() == 0:
            continue
        n_nbrs = len(nbr_dict[j])
        j_nbrs = np.array(nbr_dict[j], dtype=np.uint16)[:, np.newaxis]
        arr_i = np.ones((n_nbrs, 1), dtype=np.uint16) * i
        arr_j = np.ones((n_nbrs, 1), dtype=np.uint16) * j
        arr_tmp = np.concatenate((arr_i, j_nbrs, arr_j), axis=1)
        arr_dict = np.concatenate((j_nbrs, arr_j), axis=1)
        i_keys += list(map(tuple, arr_tmp))
        i_dict += list(map(tuple, arr_dict))
                
    b_keys += i_keys
    b_dict[i] = list(i_dict)
    
    print('{}/{}'.format(i + 1, num_node), end="\r", flush=True)


resource_rate = 0.5
lbs = np.ones(num_node, dtype=np.uint8)
vals = lbs
resource = num_node * resource_rate


def LP(resource):
    model = gp.Model('ADG')
    obj = model.addVar(vtype=GRB.CONTINUOUS, name="obj")
    x = model.addVars(x_keys, ub=1, vtype=GRB.CONTINUOUS, name='x')
    r = model.addVars(range(num_node), vtype=GRB.CONTINUOUS, name='r')
    b = model.addVars(b_keys, vtype=GRB.CONTINUOUS, name='b')
    
    model.addConstr(r.sum() <= resource, name='total resource sum')
#     model.addConstrs((b[(i, j, k)] <= w_mtx[j, k] * r[j] for (i, j, k) in b_keys), name='borrowing limitations')
    model.addConstrs((gp.quicksum((np.ones_like(knbr_dict[i]) - x.select(i, knbr_dict[i])) * vals[knbr_dict[i]]) <= obj
                  for i in range(num_node)), name='minimize maximum')
    for ra in range(num_node):
        model.addConstrs((r[k] - gp.quicksum(b.select(ra, k, '*')) + gp.quicksum(b.select(ra, '*', k))
                      >= lbs[k] * x[(ra, k)] for k in knbr_dict[ra]),
                     name='reallocation results for node {}'.format(i))
        model.addConstrs((gp.quicksum(b.select(ra, k, '*')) <= r[k] for k in range(num_node)),
                     name='total out-borrowings')
        
    model.setObjective(obj, GRB.MINIMIZE)

    return model, r


# prune
resources = [resource]
defend_results_prune = []
time_prune = []
time_one_times = []
isModeled = 0

for i in range(len(resources)):
    print("resource = %.2f" % resources[i])
    start_time = time.time()
    if(isModeled == 0):
        model, r = LP(resources[i])  # the OPT of LP and this OPT is the lowerbound of MILP, this can reduce the solution space 
        isModeled = 1
        pre_end = time.time()
    else:
        obj = model.getVars()[0]
        obj.lb = 0
        for var in model.getVars():
            if(var.varName[0] == 'x'):
                var.vtype =GRB.CONTINUOUS
        
        obj_constrain = model.getConstrByName('total resource sum')
        model.remove(obj_constrain)
        model.addConstr(r.sum() <= resources[i], name='total resource sum')
        model.update()
    
    model.optimize()
    defend_result_pre = model.getObjective().getValue()
    time_LP = model.Runtime
    print("defend_result_pre = %.2f, LP time = %.3f\n----------" % (defend_result_pre, time_LP))
    obj = model.getVars()[0]
    obj.lb = defend_result_pre
    obj.ub = sum(vals)
    for var in model.getVars():
        if(var.varName[0] == 'x'):
            var.vtype = GRB.INTEGER
    model.update()
    model.optimize()
    defend_result_prune = model.getObjective().getValue()
    time_MILP = model.Runtime
    print("defend_result = %.2f, MILP time = %.3f" % (defend_result_prune, time_MILP))

    end_time = time.time()
    print("t = %d, resource = %.2f, model running time = %.3f， one loop time = %.3f" % (i, resources[i], time_LP + time_MILP, end_time - start_time))
    
    defend_results_prune.append(defend_result_prune)
    time_one_times.append(end_time - start_time)
    time_prune.append(time_LP + time_MILP)  

print("---------------------------------\n\n")

t = 0
t_all = 0
for i in range(len(resources)):
    print("resource = %.2f, defend_result = %.2f, time_prune = %.2f, time_one_time = %.2f"
           % (resources[i], defend_results_prune[i], time_prune[i], time_one_times[i]))
    t += time_prune[i]
    t_all += time_one_times[i]
print("total optimize time = %.3f, total runtime = %.3f" % (t, t_all))


end_time = time.time()
time_diff(start_all, pre_end, "preprocessing") # preprocessing time
time_diff(0,t, "optimization")  # running time 
time_diff(start_all, end_time, "")  # total time


