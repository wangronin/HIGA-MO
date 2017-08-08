# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:20:54 2015

@author: wangronin

@Note:
    ------------------------------------------------------------------------------ 
    Dec 14 2015: 
        1. Cross-check the python implementation to the Javascript code to verify
           the python code. Successful!
        2. TODO: Improve the efficiency of the computation. Currently too slow.
           The possible causes: inefficient python interpreter, unnecessary
           matrix copies... 
        3. The step-size adaptation mechanism needs to be verified. The cumulation
           and the decaying coefficients should be related to the initial step-size,
           population size and the dimensionality.
        4. Possibility to use Conjugate Gradient methods: requires 
           (Backtracking) line search which costs more function evalutations,
    ------------------------------------------------------------------------------
    Feb 5 2016:
        1. Implement hypervolume gradient computation for m > 2 
        2. Implement the gradient computation code in C or Cython
        3. Fixed the gradient projection method 
    ------------------------------------------------------------------------------
    Apr 4 2016:
        1. new exponential rule to control the step-size
        2. new settings for c and alpha
        3. new General control rule for step-size is implemented to increase stability
           and convergence
"""

import pdb
import time
import os

import numpy as np
from numpy.linalg import norm
from numpy import argsort, r_, zeros, nonzero, array, atleast_2d, sqrt, mod, inf

from hv import HyperVolume

from pyDOE import lhs
from copy import copy, deepcopy


class MOO_HyperVolumeGradient:
    
    def __init__(self, dim_d, dim_o, lb, ub,
                 mu=40, fitness=None, gradient=None, ref=None, initial_step_size=0.1, 
                 maximize=True, sampling='uniform', adaptive_step_size=True, 
                 verbose=False, **kwargs):    
        """
        Hypervolume Indicator Gradient Ascent Algortihm class

        Parameters
        ----------

        dim_d : integer
            dimensionality of the decision space

        dim_o : integer
            dimensionality of the objective space 

        lb : array
            lower bound of the search domain

        ub : array
            upper bound of the search domain

        mu :  integer
            the size of the Pareto approxiamtion set

        fitness : callable or list of callable (function) (vector-evaluated) objective 
            function

        gradient : callable or list of callable (function)
            the gradient (Jacobian) of the objective function

        ref : array or list 
            the reference point

        initial_step_size : numeric or string
            the inital step size, it could be a string subject to evaluation
            
        maximize : boolean or list of boolean
            Is the objective functions subject to maximization. If it is a list, it specifys the maximization option per objective dimension

        sampling : string
            the method used in the initial sampling of the approximation set

        adaptive_step_size : boolean
            whether to enable to adaptive control for the step sizes, enabled by default

        verbose : boolean
            controls the verbosity

        kwargs: additiional parameters, including:
            steer_dominated : string
                the method to steer (move) the dominated points. Available options: are
                'NDS', 'M1', 'M2', 'M3', 'M4', 'M5'. 'NDS' stands for Non-dominated 
                Sorting and enabled by default. For the detail of the methods here, 
                please refer to paper [2] below.

            enable_dominated : boolen, whether to include dominated points population, for 
                test purpose only

            normalize : boolean
                if the gradient is normalized or not

        References:

        .. [1] Wang H., Deutz A., Emmerich T.M. & Bäck T.H.W., Hypervolume Indicator 
            Gradient Ascent Multi-objective Optimization. In Lecture Notes in Computer 
            Science 10173:654-669. DOI: 10.1007/978-3-319-54157-0_44. In book: 
            Evolutionary Multi-Criterion Optimization, pp.654-669.

        .. [2] Wang H., Ren Y., Deutz A. & Michael T. M.Emmerich (2016), On Steering 
            Dominated Points in Hypervolume Indicator Gradient Ascent for Bi-Objective 
            Optimization. In: Schuetze O., Trujillo L., Legrand P., Maldonado Y. (Eds.) 
            NEO 2015: Results of the Numerical and Evolutionary Optimization Workshop NEO 
            2015 held at September 23-25 2015 in Tijuana, Mexico. no. Studies in 
            Computational Intelligence 663. International Publishing: Springer.
                

        """
        self.dim_d = dim_d                
        self.dim_o = dim_o               
        self.mu = mu                   
        self.verbose = verbose

        assert self.mu > 1               # single point is not allowed 
        assert sampling in ['uniform', 'LHS', 'grid']
        self.sampling = sampling

        # step-size settings
        self.step_size = eval(initial_step_size) if isinstance(initial_step_size, 
            basestring) else initial_step_size
        self.individual_step_size = np.repeat(self.step_size, self.mu)
        self.adaptive_step_size = adaptive_step_size

        # setup boundary in decision space
        lb, ub = atleast_2d(lb), atleast_2d(ub)
        self.lb = lb.T if lb.shape[1] != 1 else lb
        self.ub = ub.T if ub.shape[1] != 1 else ub
        
        # are objective functions subject to maximization  
        if hasattr(maximize, '__iter__') and len(maximize) != self.dim_o:
            raise ValueError('maximize should have the same length as fitnessfuncs')
        elif isinstance(maximize, bool):
            maximize = [maximize] * self.dim_o
        self.maximize = np.atleast_1d(maximize)
        
        # setup reference point
        self.ref = np.atleast_2d(ref).reshape(-1, 1)  
        self.ref[~self.maximize, 0] = -self.ref[~self.maximize, 0]
        
        # setup the fitness functions
        if isinstance(fitness, (list, tuple)):
            if len(fitness) != self.dim_o:
                raise ValueError('fitness_grad: shape {} is inconsistent with dim_o:{}'.format(len(fitness), self.dim_o))
            self.fitness_func = fitness
            self.vec_eval_fitness = False
        elif hasattr(fitness, '__call__'):
            self.fitness_func = fitness
            self.vec_eval_fitness = True
        else:
            raise Exception('fitness should be either a list of functions or \
                a vector evaluated function!')
        
        # setup fitness gradient functions
        if isinstance(gradient, (list, tuple)):
            if len(gradient) != self.dim_o:
                raise ValueError('fitness_grad: shape {} is inconsistent with dim_o: {}'.format(len(gradient), self.dim_o))
            self.grad_func = gradient
            self.vec_eval_grad = False
        elif hasattr(gradient, '__call__'):
            self.grad_func = self.__obj_dx(gradient)
            self.vec_eval_grad = True
        else:
            raise Exception('fitness_grad should be either a list of functions or \
                a matrix evaluated function!')
        
        # setup the performance metric functions for convergence detection
        try:
            self.performance_metric_func = kwargs['performance_metric']
            self.target_perf_metric = kwargs['target']
        except KeyError:
            self.performance_metric_func = None
            self.target_perf_metric = None
        
        self.normalize = kwargs['normalize'] if kwargs.has_key('normalize') else True
        self.enable_dominated = True if not kwargs.has_key('enable_dominated') else kwargs['enable_dominated']
        self.maxiter = kwargs['maxiter'] if kwargs.has_key('maxiter') else inf
        
        # dominated_steer for moving non-differentiable or zero-derivative points
        try:
            self.dominated_steer = kwargs['dominated_steer']
            assert self.dominated_steer in ['M'+str(i) for i in range(1, 6)] + ['NDS']
        except KeyError:
            self.dominated_steer = 'NDS'
            
        self.pop = None
        self.pop_old = None    # for potential rollback
        
        # create some internal variables
        self.gradient = zeros((self.dim_d, self.mu))
        self.gradient_norm = zeros((self.dim_d, self.mu))
        
        # list recording on which condition the optimizer terminates
        self.stop_list = []
        
        # iteration counter        
        self.itercount = 0
        
        # step-size control mechanism
        self.hv_history = np.random.rand(10)
        self.hv = HyperVolume(-self.ref[:, 0])
        
        # step-size control mechanism
        self.path = zeros((self.dim_d, self.mu))
        self.inner_product = zeros(self.mu)
        
        # Assuming on the smooth landspace that is differentiable almost everywhere
        # We need to record the extreme point that is non-differentiable
        self.non_diff_point = []
        
        # dynamic reference points
        self.dynamic_ref = []
        
        # dominance rank of the points
        self.dominance_track = zeros(self.mu, dtype='int')
        
        # record the working states of all search points
        self._states = array(['NOT-INIT'] * self.mu, dtype='|S5')
        
    def __str__(self):
        # TODO: implement this 
        pass
    
    def __obj_dx(self, gradient):
        def obj_dx(x):
            dx = np.atleast_2d(gradient(x))
            dx = dx.T if dx.shape[0] != self.dim_o else dx
            return dx
        return obj_dx
        
    def init_sample(self, dim, n_sample, x_lb, x_ub, method=None):
        if method == None:
            method = self.sampling
            
        if method == 'LHS':
            # Latin Hyper Cube Sampling: Get evenly distributed sampling in R^dim
            samples = lhs(dim, samples=n_sample).T * (x_ub - x_lb) + x_lb
            
        elif method == 'uniform':
            samples = np.random.rand(dim, n_sample) * (x_ub - x_lb) + x_lb
        
        elif method == 'grid':
            n_sample_axis = np.ceil(sqrt(self.mu))
            self.mu = int(n_sample_axis ** 2)
            x1 = np.linspace(x_lb[0] + 0.05, x_ub[0]-0.05, n_sample_axis)
            x2 = np.linspace(x_lb[1] + 0.05, x_ub[1]-0.05, n_sample_axis)
            X1, X2 = np.meshgrid(x1, x2)
            samples = r_[X1.reshape(1, -1), X2.reshape(1, -1)]
        
        return samples
        
    def hypervolume_dx(self, positive_set, ref=None):
        if ref is None:
            ref = self.ref
            
        n_point = len(positive_set)
        pop = self.pop[:, positive_set]
        gradient_decision = zeros((self.dim_d, n_point))
        gradient_objective = self.hypervolume_df(positive_set, ref)
        
        for k in range(n_point):
            jacobian = self.grad_func(pop[:, k]) if self.vec_eval_grad else \
                array([grad(pop[:, k]) for grad in self.grad_func])
                
            # gradient vectors need to be reverted under minimization
            jacobian *= (-1) ** np.atleast_2d(~self.maximize).T
            gradient_decision[:, k] = np.dot(gradient_objective[:, k], jacobian)
            
#            if inner(jacobian[0, :], jacobian[1, :]) / (norm(jacobian[0, :]) * norm(jacobian[1, :])) == -1:
#                self._states[positive_set[k]] = 'INCO'
#            else:
#                if self._states[positive_set[k]] == 'INCO':
#                    pdb.set_trace()
#                self._states[positive_set[k]] = 'COM'
            
        return gradient_decision
        
    def hypervolume_df(self, positive_set, ref):
        if self.dim_o == 2:
            gradient = self.__2D_hypervolume_df(positive_set, ref)
        else:
            gradient = self.__ND_hypervolume_df(positive_set, ref)
    
        return gradient
    
    def __2D_hypervolume_df(self, positive_set, ref):
        n_point = len(positive_set)
        gradient = zeros((self.dim_o, n_point))
        _grad = zeros((self.dim_o, n_point))
        
        # sort the pareto front with repsect to y1
        fitness = self._fitness[:, positive_set]
        idx = argsort(fitness[0, :])
        
        # sorted Pareto front         
        sorted_fitness = fitness[:, idx]
        
        y1 = sorted_fitness[0, :]
        y2 = sorted_fitness[1, :]
        _grad[0, :] = y2 - r_[y2[1:], ref[1]]
        _grad[1, :] = y1 - r_[ref[0], y1[0:-1]]
        
        gradient[:, idx] = _grad
        return gradient
        
        
    def __ND_hypervolume_df(self, positive_set):
        # TODO: implement hypervolume gradient larger than 3D
        pass
    
    def check_population(self, fitness):
        n_point = fitness.shape[1]
        # find the pareto front, weakly and strictly dominated set
        weakly_dom_count = zeros(n_point)
        strictly_dom_count = zeros(n_point)
        
        for i in range(n_point-1):
            p = fitness[:, i]
            for j in range(i+1, n_point):
                q = fitness[:, j]
                if all(p > q):
                    strictly_dom_count[j] += 1
                elif all(p >= q) and not all(np.isclose(p, q)):
                    weakly_dom_count[j] += 1
                elif all(p < q):
                    strictly_dom_count[i] += 1
                elif all(p <= q) and not all(np.isclose(p, q)):
                    weakly_dom_count[i] += 1
        
        pareto_front = set(nonzero(np.bitwise_and(weakly_dom_count == 0, 
                                                  strictly_dom_count == 0))[0])
        
        # strictly dominated set
        S = set(nonzero(strictly_dom_count != 0)[0])
        # weakly dominated set
        W = set(nonzero(np.bitwise_and(strictly_dom_count == 0, 
                                       weakly_dom_count != 0))[0])
                                       
        # find the subset of pareto front with duplicated components                               
        if self.dim_o == 2:      # simple case in 2-D: duplication is impossible
            N, D = set(pareto_front), set([])
        else:                    # otherwise...
            D = set([])
            front_size = len(pareto_front)
            for i in range(front_size - 1):
                p = fitness[:, i]
                for j in range(i+1, front_size):
                    q = fitness[:, j]
                    if np.any(p == q):
                        D |= set([i, j])
            N = set(pareto_front) - D
        
        # exterior set
        E = set(nonzero(np.any(fitness < self.ref, axis=0))[0])
        # interior set
        I = set(nonzero(np.all(fitness > self.ref, axis=0))[0])
        # boundary set
        B = set(nonzero(np.any(fitness == self.ref, axis=0))[0]) - E
        
        Z = E | S                 # zero derivative set
        U = D | (W - E) | (B - S) # undefined derivative set
        P = N & I                 # positive derivative set
        
        return pareto_front, Z, U, P
    
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
        return fitness, _fitness
    
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
        
    def steering_dominated(self, idx_ZU):
        # The rest points move along directions aggregated from function
        # gradients by scalarization dominated_steers
        gradient_ZU = zeros((self.dim_d, len(idx_ZU)))
        
        if self.dominated_steer in ['M4', 'M5']:
            mid_gap, slope = self.__mid_gap_pareto(self.pareto_front)
            __ = list(set(range(self.mu)) - set(self.pareto_front))
            nearst_gap_idx = self.__nearst_gap(mid_gap, __)
                
        if self.dominated_steer == 'M2' and self.itercount == 0:
            self.weights = np.random.rand(len(idx_ZU))
            self.weights_mapping = {t: i for i, t in enumerate(idx_ZU)}
        
        for i, k in enumerate(idx_ZU): 
            # calculate the objective function gradients
            grads = np.atleast_2d(self.grad_func(self.pop[:, k])).T if self.vec_eval_grad \
                else array([grad(self.pop[:, k]) for grad in self.grad_func]).T
            # gradient vectors need to be reverted under minimization
            grads *= (-1) ** np.atleast_2d(~self.maximize)
            
            # simple objective scalarization with equal weights 
            if self.dominated_steer == 'M1':
                gradient_ZU[:, i] = np.sum(grads, axis=1)
            
            # objective scalarization with random (uniform) weights 
            elif self.dominated_steer == 'M2':
                try:
                    idx = self.weights_mapping[k]
                    w = self.weights[idx]
                except:
                    w = np.random.rand()
                    self.weights_mapping[k] = len(self.weights)
                    self.weights = np.append(self.weights, w)
                # random weights generation per iteration
                if 11 < 2:
                    w = np.random.rand()
                gradient_ZU[:, i] = np.sum(grads * array([w, 1-w]), axis=1)
            
            # Lara's method: scalarization after gradient normalization
            elif self.dominated_steer == 'M3':
                
                length = sqrt(np.sum(grads ** 2.0, axis=0))
                idx, = nonzero(length != 0)
                grads[:, idx] /= length[idx]
                gradient_ZU[:, i] = np.sum(grads, axis=1)
            
            # converge to the tangential point on the pareto front with the same
            # slope as the secant of the nearst gap
            elif self.dominated_steer == 'M4':
                assert self.dim_o == 2       # only work in 2-D
                
                gap_idx = nearst_gap_idx[i]
                m = slope[gap_idx]
                w = -m / (1 - m)
                gradient_ZU[:, i] = np.sum(grads * array([w, 1-w]), axis=1)
            
            # converge to the middel point of the chord of the nearst gap
            elif self.dominated_steer == 'M5':
                assert self.dim_o == 2       # only work in 2-D
                
                if len(nearst_gap_idx) == 0:
                    break
                gap_idx = nearst_gap_idx[i]
                mid = mid_gap[:, gap_idx]
                p = self._fitness[:, k]
                tmp = 2*array([mid - p])    # remember we need a gradient descent here
                gradient_ZU[:, i] = np.sum(grads * tmp, axis=1)
            
        return gradient_ZU
        
    def __nearst_gap(self, mid_gap, idx):
        """
        assigning non-differential points to the nearst gap on the pareto front 
        evenly
        """
        nearst_gap_idx = []
        
        if len(mid_gap) != 0:
            if mid_gap.shape[1] > 1:
                if 1 < 2:
                    avail_gap = range(mid_gap.shape[1])
                    for i in idx:
                        if len(avail_gap) == 0:
                            avail_gap = range(len(mid_gap))
                        p = self._fitness[:, i].reshape(-1, 1)
                        dis = np.sum((mid_gap[:, avail_gap] - p) ** 2.0, axis=0)
                        
                        if len(dis) != 0:
                            nearst_idx = avail_gap[nonzero(dis == min(dis))[0][0]]
                            nearst_gap_idx.append(nearst_idx)
                        
                        avail_gap = list(set(avail_gap) - set(nearst_gap_idx))
                else:
                    for i in idx:
                        p = self._fitness[:, i].reshape(-1, 1)
                        dis = np.sum((mid_gap - p) ** 2.0, axis=0)
                        if len(dis) != 0:
                            nearst_gap_idx.append(nonzero(dis == min(dis))[0][0])
                
        return nearst_gap_idx

    def __mid_gap_pareto(self, idx_P):
        front = self._fitness[:, idx_P]
        
        # sort the front according to the first axis
        idx = argsort(front[0, :])
        sorted_front = front[:, idx]
        n_point = len(idx_P)
        slope = np.zeros(n_point)
        
        A = sorted_front[:, 0:-1]
        B = sorted_front[:, 1:]
        mid_gap = (A + B) / 2.0
        
        for i in range(n_point-1):
            if np.isclose(B[0, i] - A[0, i], 0):
                slope[i] = np.inf
            else:
                slope[i] = (B[1, i] - A[1, i]) / (B[0, i] - A[0, i])
        
        return mid_gap, slope
    
    def check_stop_criteria(self):
        # TODO: implemement more stop criteria 
        if self.itercount >= self.maxiter:
            self.stop_list += ['maxiter']
            
        if self.itercount > 0:
            # check for stationary necessary condition
            grad_len = sqrt(np.sum(self.gradient ** 2.0, axis=0))
            if np.all(grad_len < 1e-5):
                self.stop_list += ['stationary']
            
            # check if the step-size is too small
            if np.all(self.individual_step_size < 1e-5 * np.max(self.ub - self.lb)):
                self.stop_list += ['step-size']
                
        # check if the performance metric target is reached
        if self.itercount > 0 and self.performance_metric_func is not None:
            PF = self.fitness[:, self.pareto_front].T
            self.perf_metric = self.performance_metric_func(PF)
            if self.perf_metric <= self.target_perf_metric:
                self.stop_list += ['target']
            
        return self.stop_list
        
    def __backtracing_line_search(self, idx):
        point = self.pop[:, idx]
        idx_point = self.pareto_front.index(idx)
        pareto = self.fitness[:, self.pareto_front]
        alpha = self.step_size
        beta = 1
        tau = 0.7
        
        gradient = self.gradient[:, idx]        
        f0 = self.hv.compute(pareto.T)
        while True:
            new_point = point + alpha * gradient
            new_fitness, _ = self.evaluate(new_point[:, np.newaxis])
            pareto[:, idx_point] = new_fitness[:, 0]
            fnew = self.hv.compute(pareto.T)
            
            if fnew >= f0 + alpha * beta * np.linalg.norm(gradient)**2:
                break
            alpha *= tau
            
        return alpha

    def step_size_control(self):
        # ------------------------------ IMPORTANT -------------------------------
        # The step-size adaptation is of vital importance here!!!
        # general control rule to improve the stability: when a point tries to merge
        # into a front that dominates it in the last teration, the step-size 
        # of this particular point is set to the mean step-size of the front.
        if hasattr(self, 'dominance_track_old'):
            for i, rank in enumerate(self.dominance_track):
                if rank != self.dominance_track_old[i]:
                    idx = self.fronts[rank]
                    mean_step_size = np.median(self.individual_step_size[idx])
                    self.individual_step_size[i] = mean_step_size
                        
        self.dominance_track_old = deepcopy(self.dominance_track)
        
        #==============================================================================
        # step-size control method 1: detection of oscillating HV
        # works in very primitive case, needs more test
        # It requires hypervolume indicator computation, which is time-consuming
        #==============================================================================
        if 11 < 2:
            front = self.fitness[:, self.pareto_front]
            hv = HyperVolume(-self.ref[:, 0])
            self.hv_history[self.itercount%10] = hv.compute(front.T.tolist())
            
            if (self.dominated_steer == 'NDS' and len(self.fronts) == 1) or \
                (not self.dominated_steer == 'NDS' and len(self.idx_ZU) == 0):
                    if len(np.unique(self.hv_history)) == 2:
                        self.step_size *= 0.8
        
        #==============================================================================
        # Step size control method 2: cumulative step-size control
        # It works reasonably good among many tests and does not require too much 
        # additional computational time
        #==============================================================================
        if self.itercount != 0:
            # the learning rate setting is largely afffected by the situation of 
            # oscillation. The smaller this value is, the larger oscilation it 
            # could handle. However, the smaller this value implies slower learning rate
            alpha = 0.7       # used for general purpose
            # alpha = 0.5     # currently used for Explorative Landscape Analysis
            c = 0.2
            if 11 < 2:
                if self.dominated_steer == 'NDS':
                    control_set = range(self.mu)
                else:
                    control_set = self.idx_P
            else: 
            # TODO: verify this: applying the cumulative step-size control to all 
            # the search points 
                control_set = range(self.mu)
            
            if 1 < 2:
                from scipy.spatial.distance import cdist 
                for idx in control_set:
                    self.inner_product[idx] = (1 - c) * self.inner_product[idx] + \
                        c * np.inner(self.path[:, idx], self.gradient_norm[:, idx])
                    
                    if 11 < 2:
                        # step-size control rule similar to 1/5-rule in ES
                        if self.inner_product[idx] < 0:
                            self.individual_step_size[idx] *= alpha
                        else:
                            step_size_ = self.individual_step_size[idx] / alpha
                            self.individual_step_size[idx] = np.min([np.inf*self.step_size,                                     step_size_])
                    
                    # control the change rate of the step-size by passing the cumulative 
                    # dot product into the exponential function   
                    step_size_ = self.individual_step_size[idx] * \
                        np.exp((self.inner_product[idx])*alpha)
                    
                    # put a upper bound on the adaptive step-size to avoid it becoming two 
                    # large! The upper bound is calculated as the distance from one point 
                    # to its nearest neighour in the decision space.
                    _ = [i for i, front in enumerate(self.fronts) if idx in front][0]
                    front = self.fronts[_]
                    if len(front) != 1:
                        __ = list(set(front) - set([idx]))
                        dis = cdist(np.atleast_2d(self.pop[:, idx]), self.pop[:, __].T)
                        step_size_ub = 0.7 * np.min(dis)
                    else:
                        step_size_ub = 4.*self.step_size
                        
                    self.individual_step_size[idx] = np.min([step_size_ub, step_size_])
                    self.path[:, idx] = self.gradient_norm[:, idx]   

        #==============================================================================
        # step-size control method 3: exploit the backtracing Line search to find 
        # the optimal step-size setting. works but requires much more function evaluations
        #==============================================================================   
        if 11 < 2:
            for idx in self.idx_P:
                self.individual_step_size[idx] = self.__backtracing_line_search(idx)
                
    def constraint_handling(self, pop):
        # handling the simple box constraints
        lb, ub = self.lb[:, 0], self.ub[:, 0]
        for i in range(self.mu):
            p = pop[:, i]
            idx1, idx2 = p <= lb, p >= ub
            p[idx1] = lb[idx1]
            p[idx2] = ub[idx2]
            
    def restart_check(self):
        """
        In general, this function check if a subset of the approximation set should
        be re-sampled. The re-sampling condition is:
            stationary or non-differetiable but not on the global PF
        This function also checks if the hypervolume indicator gradient at a point
        should be projected with respect to a voilated box constraint.
        Perhaps put those in another function?
        """
        pareto_front = set(self.pareto_front)
        
        # check for non-differentiable points
        non_diff = set(nonzero(np.any(np.bitwise_or(np.isnan(self.gradient), 
                                        np.isinf(self.gradient)), axis=0))[0])
                                        
        # check for boundary points
        on_bound = set(nonzero(np.any(np.bitwise_or(self.pop == self.lb, 
                                                    self.pop == self.ub), axis=0))[0])
                                                    
        # TODO: gradient projection
        # if: 
        #   1) the point is on the boundary
        #   2) it tries to violate the boundary constraint
        # then set the corresponding gradient component to zero
        if 1 < 2:                                           
            ind = list(on_bound)
            for i in ind:
                point = self.pop[:, i]
                for j in range(self.dim_d):
                    if point[j] == self.lb[j] or point[j] == self.ub[j]:
                        self.gradient[j, i] = 0
                                                    
        # check for zero gradient points
        zero_grad = set(nonzero(np.all(np.isclose(self.gradient, 0), axis=0))[0])
        
        # make the non-differentiable point that is on the pareto front stationary
        stationary_ind = zero_grad & pareto_front
        # resample the point that 1) on the boundart 2) has zero gradient
        # 3) has invalid gradient
        restart_ind = (zero_grad | non_diff) - pareto_front - stationary_ind
        
        return list(restart_ind), list(stationary_ind)
    
    def duplication_handling(self):
        # check for the potential duplication of stationary point in the popualation
        # add move such point to prevent duplication of points
        # TODO: verify this! possibly better dominated_steer exists
        fitness = self.fitness[:, self.pareto_front]
        checked_list = []

        for i, ind in enumerate(self.pareto_front):
            if i in checked_list:
                continue
            
            p = self.fitness[:, [ind]]
            bol = np.all(p == fitness, axis=0)
            bol[i] = False
            tmp = nonzero(bol)[0]
            
            # when duplication happens, move one duplicated point slightly
            if len(tmp) != 0:
                for k in tmp:
                    duplication_ind0 = self.pareto_front[i]
                    duplication_ind = self.pareto_front[k]
                    
                    if sum(self.gradient[:, duplication_ind]) == 0 or \
                        sum(self.gradient[:, duplication_ind0]) == 0:
                    
                        # compute its nearst neighour in decision space
                        point = self.pop[:, [duplication_ind]]
                        dis = np.sum((self.pop[:, self.pareto_front] - point) ** 2.0, axis=0)
                        dis[dis == 0] = np.inf
                        res = nonzero(dis == min(dis))[0][0]
                        nearst_ind = self.pareto_front[res]
                       
                        # move to the half way to its nearst neighour and penalize its stepsize
                        self.pop[:, duplication_ind] = (self.pop[:, nearst_ind] + \
                            self.pop[:, duplication_ind]) / 2.
                        
                        self.individual_step_size[duplication_ind] = self.step_size / 5.
                        checked_list += [k]

    def update_ref_point(self):
        # TODO: verify the performance difference of this to the fixed reference point
        self.dynamic_ref = []
        for front in self.fronts:
            front_fitness = self._fitness[:, front]
            ref_front = np.repeat(np.min(front_fitness), self.dim_o)
            ref_front += 0.1 * ref_front
            self.dynamic_ref += [ref_front]
            
    def step(self):
        
        # population initialization
        if self.pop is None:
            self.pop = self.init_sample(self.dim_d, self.mu, self.lb, self.ub)
            self._states = array(['INIT'] * self.mu)
            
        # evaluation
        self.fitness, self._fitness = self.evaluate(self.pop)
        
        # Combine non dominated sorting with HyperVolume gradient ascend
        if self.dominated_steer == 'NDS':
            self.fronts = self.fast_non_dominated_sort(self._fitness)
            self.pareto_front = self.fronts[0]
            
            # TODO: implement the dynamic reference point update:
#            self.update_ref_point()
            
            # compute the hypervolume gradient for each front layer
            for i, front in enumerate(self.fronts):
                self.gradient[:, front] = self.hypervolume_dx(front)
        
        # partition the collection of vectors according to Hypervolume indicator 
        # differetiability
        else:
            pareto_front, Z, U, P = self.check_population(self._fitness)
            self.fronts = self.fast_non_dominated_sort(self._fitness)
            self.idx_P, self.idx_ZU = list(P), list(Z | U)
            self.pareto_front = list(pareto_front)
            
            __ = list(set(range(self.mu)) - set(self.pareto_front))
            
            # compute the hypervolume gradient for differentiable points
            # TODO: check why I abandon the idx_P here
#            gradient_P = self.hypervolume_dx(self.idx_P)
            gradient_P = self.hypervolume_dx(self.pareto_front)
            gradient_ZU = self.steering_dominated(__) \
                if self.enable_dominated and len(__) != 0 else []
            
            self.gradient[:, self.pareto_front] = gradient_P
            self.gradient[:, __] = gradient_ZU
        
        # check for stationary points and points needs to be resampled
        restart_ind, stationary_ind = self.restart_check()
        
        if 11 < 2:
            # do not allow restart... for debug purpose
            restart_ind = []
        
        for j, f in enumerate(self.fronts):
            self.dominance_track[f] = j
        
        # index set for gradient ascent
        ind = list(set(range(self.mu)) - set(restart_ind) - set(stationary_ind))
        
        # normalization should be performed after gradient correction
        self.gradient_norm[:, :] = self.gradient
        if self.normalize:
            length = sqrt(np.sum(self.gradient_norm ** 2.0, axis=0))
            idx, = nonzero(length != 0)
            self.gradient_norm[:, idx] /= length[idx]
        
        # call the step-size control function
        if self.adaptive_step_size:
            self.step_size_control()
        
        # Prior to gradient ascent, copy the population for potential rollback
        self.pop_old = copy(self.pop)
        
        # re-sample the point uniformly and reset the corresponding step-size to 
        # the initial step-size
        # TODO: better restarting action rules and better implementation
        self.individual_step_size[restart_ind] = self.step_size
        self.pop[:, restart_ind] = self.init_sample(self.dim_d, len(restart_ind),
                                                    self.lb, self.ub, 'uniform')
        
        # gradient ascending along the normalized gradient
        self.pop[:, ind] += self.individual_step_size[ind] * self.gradient_norm[:, ind]
        
        # constraint handling methods
        self.constraint_handling(self.pop)
        
        # repair the duplicated points
        self.duplication_handling()
        
        # incremental...
        self.itercount += 1
        
        return self.pareto_front, self.fitness
        
    def optimize(self):
        
        # Main iteration        
        while not self.check_stop_criteria():
            self.step()
        
        return self.pareto_front, self.itercount

