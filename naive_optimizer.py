# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:54:52 2016

@author: wangronin
@Note: 
"""
import pdb
from pyDOE import lhs

import numpy as np
from numpy import r_, array, atleast_2d, sqrt, nonzero, zeros

from boundary_handling import boundary_handling

from copy import deepcopy

class MOO_naive:
    
    def __init__(self, dim_d, dim_o, mu=40, fitness=None, opts=None, sampling='lhs', 
                 step_size=0.1, maxiter=None, maximize=True, verbose=False):
        self.dim_d = dim_d               # dimensionality of decision space 
        self.dim_o = dim_o               # dimensionality of objective space
        self.mu = mu                     # population size
        self.step_size = step_size       # step size
        self.verbose = verbose
        self.itercount = 0               # iteration counter
        self.evalcount = 0               # evaluation counter
        self.stop_list = []
        assert self.mu > 1   
        
        if hasattr(maximize, '__iter__') and len(maximize) != self.dim_o:
            raise ValueError("""maximize should have the same length as
                fitnessfuncs""")
        elif isinstance(maximize, bool):
            maximize = [maximize] * self.dim_o
        self.maximize = np.atleast_1d(maximize)
        
        # setup the fitness functions
        if hasattr(fitness, '__iter__'):
            if len(fitness) != self.dim_o:
                raise ValueError("""fitness_grad should have the same length as 
                    fitnessfuncs""")
            self.fitness_func = fitness
            self.vec_eval_fitness = False
        elif hasattr(fitness, '__call__'):
            self.fitness_func = fitness
            self.vec_eval_fitness = True
        else:
            raise Exception("""fitness should be either a list of functions or 
                a vector evaluated function!""")
                
        # setup sampling method
        if sampling not in ['unif', 'lhs', 'grid']:
            raise Exception('{} sampling is not supported!'.format(sampling))
        self.sampling = sampling
        
        # setup boundary in decision space
        lb = eval(opts['lb']) if isinstance(opts['lb'], basestring) else opts['lb']
        ub = eval(opts['ub']) if isinstance(opts['ub'], basestring) else opts['ub']
        lb, ub = atleast_2d(lb), atleast_2d(ub)
        
        self.lb = lb.T if lb.shape[1] != 1 else lb
        self.ub = ub.T if ub.shape[1] != 1 else ub
        
        self.maxiter = maxiter
        
        # setup the performance metric functions for convergence detection
        try:
            self.performance_metric_func = opts['performance_metric']
            self.target_perf_metric = opts['target']
        except KeyError:
            self.performance_metric_func = None
            self.target_perf_metric = None
        
        self.pop = self.init_sample(self.dim_d, self.mu, self.lb, self.ub)
        self.fitness, self._fitness = self.evaluate(self.pop)
        
        
    def init_sample(self, dim, n_sample, x_lb, x_ub, method=None):
        if method == None:
            method = self.sampling
            
        if method == 'lhs':
            # Latin Hyper Cube Sampling: Get evenly distributed sampling in R^dim
            samples = lhs(dim, samples=n_sample).T * (x_ub - x_lb) + x_lb
            
        elif method == 'unif':
            samples = np.random.rand(dim, n_sample) * (x_ub - x_lb) + x_lb
            
        elif method == 'grid':
            
            n_sample_axis = np.ceil(sqrt(self.mu))
            self.mu = int(n_sample_axis ** 2)
            x1 = np.linspace(x_lb[0] + 0.05, x_ub[0]-0.05, n_sample_axis)
            x2 = np.linspace(x_lb[1] + 0.05, x_ub[1]-0.05, n_sample_axis)
            X1, X2 = np.meshgrid(x1, x2)
            samples = r_[X1.reshape(1, -1), X2.reshape(1, -1)]
    
        return samples
        
    def fast_non_dominated_sort(self, fitness):
            
        fronts = []
        dominated_set = []
        n_domination = zeros(self.mu)
        
        for i in range(self.mu):
            p = fitness[:, i]
            p_dominated_set = []
            n_p = 0
            
            for j in range(self.mu):
                q = fitness[:, j]
                if i != j:
                    # TODO: verify this part 
                    # check the strict domination
                    # allow for duplication points on the same front
                    if all(p >= q) and not all(p == q):
                        p_dominated_set.append(j)
                    elif all(p <= q) and not all(p == q):
                        n_p += 1
                        
            dominated_set.append(p_dominated_set)
            n_domination[i] = n_p
        
        # create the first front
        fronts.append(nonzero(n_domination == 0)[0].tolist())
        n_domination[n_domination == 0] = -1
        
        i = 0
        while True:
            for p in fronts[i]:
                p_dominated_set = dominated_set[p]
                n_domination[p_dominated_set] -= 1
                
            _front = nonzero(n_domination == 0)[0].tolist()
            n_domination[n_domination == 0] = -1
                
            if len(_front) == 0:
                break
            fronts.append(_front)
            i += 1
            
        return fronts
        
        
    def evaluate(self, pop):
        pop = np.atleast_2d(pop)
        pop = pop.T if pop.shape[0] != self.dim_d else pop
        n_point = pop.shape[1] 
        
        if self.vec_eval_fitness:
            fitness = array([self.fitness_func(pop[:, i]) for i in range(n_point)]).T
        else:
            fitness = array([[func(pop[:, i]) for i in range(n_point)] \
                for func in self.fitness_func])
        # fitness values need to be reverted under minimization
        _fitness = fitness * (-1) ** np.atleast_2d(~self.maximize).T
        self.evalcount += self.mu
        
        return fitness, _fitness
        
        
    def __is_dominated(self, p, q):
        return all(p >= q) and not all(np.isclose(p, q))
        
    
    def step(self):
            
        # generate mutations: 
        s = np.random.randn(self.dim_d, self.mu)
        s /= max(sqrt(np.sum(s**2.0, axis=0)))
        
        pop_new = self.pop + self.step_size * s
        pop_new = boundary_handling(pop_new, self.lb, self.ub)
        
        # evaluation
        fitness_new, _fitness_new = self.evaluate(pop_new)
        
        # (1+1)-selection for each individual
        for i in range(self.mu):
            if self.__is_dominated(_fitness_new[:, i], self._fitness[:, i]):
                self.pop[:, i] = deepcopy(pop_new[:, i])
                self._fitness[:, i] = deepcopy(_fitness_new[:, i])
                self.fitness[:, i] = deepcopy(fitness_new[:, i])
                
        self.fronts = self.fast_non_dominated_sort(self._fitness)
        self.pareto_front = self.fronts[0]
        self.itercount += 1
        
        return self.pareto_front, self.fitness
        
        
    def check_stop_criteria(self):
        if self.itercount >= self.maxiter:
            self.stop_list += ['maxiter']
        
        # check if the performance metric target is reached
        if self.itercount > 0 and self.performance_metric_func is not None:
            PF = self.fitness[:, self.pareto_front].T
            self.perf_metric = self.performance_metric_func(PF)
            if self.perf_metric <= self.target_perf_metric:
                self.stop_list += ['target']
        
        return self.stop_list
    
        
    def optimize(self):
        # Main iteration    
        while not self.check_stop_criteria():
            self.step()
            
        return self.pareto_front, self.fitness, self.itercount
    
    
    def step_size_control(self):
        # TODO: Applying 1/5 success rule for this algorithm
        # not really important for this test algorithm...
        pass
    
if __name__ == '__main__':
    
    from mo_problem import problem
    
    dim = 2
    mu = 10
    step_size = 0.001
    prob = problem(1, dim)
    x_lb = array(prob.lb)
    x_ub = array(prob.ub)
    
    opts = {'lb' : x_lb,
            'ub' : x_ub,
            'performance_metric' : prob.convergence_metric,
            'target' : 0.03,
            }

    optimizer = MOO_naive(dim, 2, mu, prob.objfun, opts=opts, step_size=step_size,
                           sampling='lhs', maxiter=1e5, maximize=False)
                           
    optimizer.optimize()
    print optimizer.stop_list
    print optimizer.itercount
        
            
            