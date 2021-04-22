import time
import numpy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from scipy import sparse

class GreedyADG:

    # adjacency_matrix: numpy array nxn, here we use weighted adjacency matrix
    # values: numpy array 1xn
    # lower_bounds: numpy array 1xn
    # total_resources: decimal
    # attack_range: integer

    def __init__(self, adjacency_matrix, values, lower_bounds, total_resources, attack_range):
        self.graph = adjacency_matrix
        self.values = values
        self.lower_bounds = lower_bounds
        self.total_resources = total_resources
        self.size = adjacency_matrix.shape[0]
        self.attack_range = attack_range
        self.resources = numpy.zeros(self.size)
        self.defending_powers = numpy.zeros((self.size, self.size))
        self.reallocation = numpy.zeros((self.size, self.size, self.size))
    
    def walk(self, centre, deepth, indicator):
        indicator[centre] = 1
        if deepth == 0:
            return
        for neighbor in self.graph[centre].nonzero()[0]:
            self.walk(neighbor, deepth - 1, indicator)

    def findVerticesUnderAttack(self, attack_range):
        indicator = numpy.zeros((self.size, self.size))
        for centre in range(self.size):
            self.walk(centre, attack_range, indicator[centre])
        return indicator
    
    def calculateDefendingPower(self):
        for centre in range(self.size):
            vertices = self.indicator[centre].nonzero()[0]
            for vertex in vertices:
                for neighbor in self.graph[vertex].nonzero()[0]:
                    if self.indicator[centre][neighbor] == 1:
                        self.defending_powers[centre][vertex] += self.reallocation[centre][neighbor][vertex] - self.reallocation[centre][vertex][neighbor]
                    if self.indicator[centre][neighbor] == 0:
                        self.defending_powers[centre][vertex] += self.reallocation[centre][neighbor][vertex]
    
    def calculateReallocationFromNeighbors(self, centre, target):
        totalReallocation = 0.0
        for vertex in self.graph[target].nonzero()[0]:
            totalReallocation += self.reallocation[centre][vertex][target]
        return totalReallocation
    
    def calculateResults(self):
        results = numpy.zeros(self.size)
        for centre in range(self.size):
            vertices = self.indicator[centre].nonzero()
            dps = self.defending_powers[centre][vertices]
            vals = self.values[vertices]
            lbs = self.lower_bounds[vertices]
            results[centre] = vals[dps < lbs].sum()
        return results.max(), results
    
    def maximumReallocationFromUnattackedNeighbors(self, centre, vertex, resources):
        results = 0.0
        for neighbor in self.graph[vertex].nonzero()[0]:
            if self.indicator[centre][neighbor] == 0:
                results += self.graph[neighbor][vertex] * resources[neighbor]
        return results

    def greedyAllocation(self):
        allocated = 0.0
        self.indicator = self.findVerticesUnderAttack(self.attack_range)
        for vertex in numpy.argsort(self.values)[::-1]:
            if self.total_resources - allocated >= self.lower_bounds[vertex]:
                self.resources[vertex] = self.lower_bounds[vertex]
                allocated += self.resources[vertex]
                self.defending_powers[:, vertex] += self.resources[vertex]
            else:
                self.resources[vertex] = self.total_resources - allocated
                self.defending_powers[:, vertex] += self.resources[vertex]
                break
    
    def greedyAllocationWithGreedyReallocation(self, type=0):
        self.greedyAllocation()
        for centre in range(self.size):
            vertices = self.indicator[centre].nonzero()[0]
            vals = self.values[vertices]
            lbs = self.lower_bounds[vertices]
            res = self.resources[vertices]
            vertices = vertices[res < lbs]
            vals = vals[res < lbs]
            resources_temp = self.resources.copy()
            for vertex in vertices[numpy.argsort(vals)[::-1]]:
                if type == 1:
                    # vertex can not be protected.
                    totalDemand = self.lower_bounds[vertex] - self.resources[vertex]
                    if totalDemand > self.maximumReallocationFromUnattackedNeighbors(centre, vertex, resources_temp):
                        continue
                for neighbor in self.graph[vertex].nonzero()[0]:
                    if self.indicator[centre][neighbor] == 0:
                        currentDemand = self.lower_bounds[vertex] - self.resources[vertex] - self.calculateReallocationFromNeighbors(centre, vertex)
                        if currentDemand > 0:
                            if self.graph[neighbor][vertex] * resources_temp[neighbor] <= currentDemand:
                                self.reallocation[centre][neighbor][vertex] += self.graph[neighbor][vertex] * resources_temp[neighbor]
                                resources_temp[neighbor] = 0.0
                            else:
                                self.reallocation[centre][neighbor][vertex] += currentDemand
                                #resources_temp[neighbor] -= currentDemand / self.graph[neighbor][vertex]
                                resources_temp[neighbor] -= currentDemand
                        else:
                            break
    
    def plotReallocation(self, centre):
        x = self.indicator[centre].nonzero()[0]
        vals = self.values[x]
        lbs = self.lower_bounds[x]
        res = self.resources[x]
        dps = self.defending_powers[centre][x]
        descendingByValues = numpy.argsort(vals)[::-1]
        x = x[descendingByValues]
        vals = vals[descendingByValues]
        lbs = lbs[descendingByValues]
        res = res[descendingByValues]
        dps = dps[descendingByValues]
        x_axis = numpy.array([i for i in range(self.indicator[centre].sum().astype(numpy.int64))])
        plt.figure(figsize=[16, 8])
        plt.xlabel('vertex')
        plt.ylabel('value')
        plt.plot(x_axis, vals, color='black', label='value')
        plt.plot(x_axis, lbs, color='red', label='lower bound')
        plt.plot(x_axis, res, color='blue', label='resource')
        plt.plot(x_axis, dps, color='green', label='defending power')
        plt.legend()
        plt.show()

    def plotResourceAllocation(self):
        
        def getKeyPoint(size, resources, lower_bounds):
            for vertex in range(size):
                if resources[vertex] < lower_bounds[vertex]:
                    return vertex
        
        x = numpy.argsort(self.values)[::-1]
        vals = self.values[x]
        res = self.resources[x]
        lbs = self.lower_bounds[x]
        x_axis = numpy.array([ i for i in range(self.size)])
        key_point = getKeyPoint(self.size, res, lbs)
        plt.figure(figsize=[16, 8])
        plt.xlabel('vertex')
        plt.ylabel('value')
        plt.plot(x_axis, vals, color='black', label='value')
        plt.plot(x_axis, lbs, color='red', label='lower bound')
        plt.plot(x_axis, res, color='blue', label='resource')
        plt.fill_between(x_axis[key_point:greedyADG.size],vals[key_point:greedyADG.size], color='pink', alpha=0.2, hatch="/")
        plt.legend()
        plt.show()

if __name__ == '__main__':

    # example
    adj = sparse.load_npz('twit_mtx_a_n=1000_e=13476.npz').toarray()
    vals = numpy.ones(1000)
    lbs = numpy.ones(1000)
    total_resources = 500
    attack_range = 2

    # greedy allocation w/o reallocation
    greedyADG = GreedyADG(adj, vals, lbs, total_resources, attack_range)
    greedyADG.greedyAllocation()
    r1, rs1 = greedyADG.calculateResults()
    print("result: ", r1)

    # greedy allocation with greedy reallocation
    greedyADG = GreedyADG(adj, vals, lbs, total_resources, attack_range)
    greedyADG.greedyAllocationWithGreedyReallocation(type=0)
    greedyADG.calculateDefendingPower()
    r2, rs2 = greedyADG.calculateResults()
    print("result: ", r2)

    # greedy allocation with better greedy reallocation
    greedyADG = GreedyADG(adj, vals, lbs, total_resources, attack_range)
    greedyADG.greedyAllocationWithGreedyReallocation(type=1)
    greedyADG.calculateDefendingPower()
    r3, rs3 = greedyADG.calculateResults()
    print("result: ", r3)
    
    # greedy allocation with optimal reallocation
    # pass
    
    # in normal, r3 <= r2 <= r1
