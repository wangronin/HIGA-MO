# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:43:09 2015

@author: wangronin
"""


import pdb
import numpy as np
from numpy import zeros, array, ones, cos, sin, sum, pi, sqrt, exp

import warnings


class MOP:
    
    def __init__(self):
        pass
    
    def __check_input(self, x):
        x = np.atleast_1d(x)
        assert len(x) == self.dim
        return x
    
    def pareto_front(self):
        pass
     
    def objfun(self, x):
        pass
    
            
    def hypervolume_indicator(self, non_dominated):
        pass
    
    
    def convergence_metric(self, non_dominated, n_approximation=1000):
        
#        non_dominated = self.check_input(non_dominated)
    
        # discretized Pareto front to compute convergence measure
        f1_interval, pareto_f2 = self.pareto_front()
        
        f1_len = np.sum(f1_interval[:, 1] - f1_interval[:, 0])
        f1 = array([])
        for i, interval in enumerate(f1_interval):
            len_ = interval[1] - interval[0]
            _ = int(np.floor(len_ / f1_len * n_approximation))
            f1 = np.r_[f1, np.linspace(interval[0], interval[1], _)]
        
        f2 = pareto_f2(f1)
        pareto_front = np.vstack([f1, f2]).T
            
        # compute the convergence metric
        metric = 0
        for p in non_dominated:
            dis = np.sqrt(np.sum((pareto_front - p) ** 2.0, axis=1))
            metric += np.min(dis)
        
        metric /= len(non_dominated)
        
        return metric
        

class zdt(MOP):
    """
    Bi-objective optimization problem by E. Zitzler, K. Deb, and L. Thiele (ZDT). 
    The objective functions have the following form:
        f1(x) = f1(x1)
        f2(x) = g(x2,...,xn) * h(f1(x1), g(x2,...,xn))
    The gradient of the objective functions are calculated 
    TODO: implement the Hessian matrices computation
    """
    
    def __init__(self, index, dim):
        self.index = index
        self.dim = dim
        self.f_dim = 2
        
        # setup boundary constraints in the decision space 
        if self.index in [1, 2, 3, 6]:
            self.lb = [0.] * self.dim
            self.ub = [1.] * self.dim
        elif self.index == 4:
            self.lb = [0.] + [-5.] * (self.dim - 1)
            self.ub = [1.] + [5.] * (self.dim - 1)

                        
    def __str__(self):
        # TODO: add some the class string here..
        pass        
        
        
    def __check_input(self, x):
        x = np.atleast_1d(x)
        assert len(x) == self.dim
        return x
        
        
    def objfun(self, x):
        x = self.__check_input(x)
        f1 = self.f1(x)
        g = self.g(x)
                
        return array([f1, self.g(x) * self.h(f1, g)])
        
        
    def f1(self, x):
        if self.index == 6:
            x1 = x[0]
            return 1. - exp(4. * x1) * sin(6. * pi * x1) ** 6.
        else:
            return x[0]
            
            
    def f1_dx(self, x):
        if self.index in range(1, 5):
            dx = zeros(self.dim)
            dx[0] = 1
    
        elif self.index == 6:
            dx = zeros(self.dim)
            dx[0] = -16.0 * x[0] * np.exp(-4.0 * x[0]) * np.sin(6.0*pi*x[0]) ** 6.0 - \
                6.0 * (np.sin(6.0*pi*x[0]) ** 5.0) * np.cos(6.0*pi*x[0])* 6.0 * pi * np.exp(-4.0 * x[0])
        
        return dx
    
    
    def g(self, x):
        x = x[1:]
        
        if self.index in [1, 2, 3]:
            return 1. + 9. * sum(x) / (self.dim - 1.)
            
        elif self.index == 4:
            return 1. + 10. * (self.dim - 1.) + sum(x ** 2. - 10. * cos(4. * pi * x))
        elif self.index == 6:
            return 1. + 9. * (sum(x) / (self.dim - 1.)) ** 0.25 
            
            
    def __g_dx(self, x):
        x = x[1:]
        
        if self.index in [1, 2, 3]:
            dx = 9.0 / (self.dim - 1.0) * ones(self.dim)
            dx[0] = 0
        elif self.index == 4:
            return 1. + 10. * (self.dim - 1.) + sum(x ** 2. - 10. * cos(4. * pi * x))
        elif self.index == 6:
            return 1. + 9. * (sum(x) / (self.dim - 1.)) ** 0.25 
            dx = zeros(self.dim)
            dx[1:] = 9.0 * 0.25 * (sum(x) / (self.dim - 1)) ** (0.25 - 1.) / (self.dim - 1.0)
        
        return dx
            
            
    def h(self, f1, g):
        
        if self.index in [1, 4]:
            return 1. - sqrt(f1 / g)
        elif self.index in [2, 5]:
            return 1. - (f1 / g) ** 2.
        elif self.index == 3:
            return 1. - sqrt(f1 / g) - (f1 / g) * sin(10. * pi * f1)
            
            
    def __h_dx(self, f1, g, f1_dx, g_dx):
        
        if self.index in [1, 4]:
            return - 0.5 * sqrt(f1 / g) ** -1. * (g * f1_dx - f1 * g_dx) / g ** 2.0
        elif self.index in [2, 5]:
            return - 2. * (f1 / g) * (g * f1_dx - f1 * g_dx) / g ** 2.0
        elif self.index == 3:
            return - 0.5 * sqrt(f1 / g) ** -1. * (g * f1_dx - f1 * g_dx) / g ** 2.0 - \
                (f1_dx * g - f1 * g_dx) / (g ** 2.0) * sin(10. * pi * f1) - \
                10. * pi * (f1 / g) * cos(10. * pi * f1) * f1_dx
            
            
    def objfun_dx(self, x):
        x = self.__check_input(x)
        f1 = self.f1(x)
        
        with warnings.catch_warnings(): 
            warnings.filterwarnings('error')
            try:
                f1_dx = self.f1_dx(x)
                g_dx = self.__g_dx(x)
                g = self.g(x)
                h = self.h(f1, g)
                f2_dx = g_dx * h + g * self.__h_dx(f1, g, f1_dx, g_dx)
            # use 'warnings' to catch possible non-differentiable points
            except Warning:
                f2_dx = np.zeros(self.dim)
        
        return array([f1_dx, f2_dx])
        
    def pareto_front(self):
        
        if self.index in [1, 4]:
            return array([[0, 1]]), lambda f1: 1. - sqrt(f1)
        elif self.index == 2:
            return array([[0, 1]]), lambda f1: 1. - f1 ** 2.
        elif self.index == 3:
            return array([[0, 0.0830015349],
                    [0.1822287280, 0.2577623634],
                    [0.4093137648, 0.4538821041],
                    [0.6183967944, 0.6525117038],
                    [0.8233317938, 0.8518328654]]), \
                    lambda f1: 1. - sqrt(f1) - f1 * sin(10. * pi * f1)
        elif self.index == 6:
            return array([[0.2807753191, 1]]), lambda f1: 1. - f1 ** 2.
        

class simple(MOP):
    
    def __init__(self, problem_index, dim, alpha=0.25):
        
        # problem indices encode different multi-objective problems
        # Bi-objective problems
        # ---------------------------------------------------------
        # 1 : double spheres problems
        # 2 : Funnel-like bi-objective problems provided by Muenster.
        # 3 : TODO: need a name for this problem.
        # 4 : TODO: verify EBN problem with Michael. It could be the same with GSP 
        # 5 : Generalized Schaffer problem
        if problem_index not in [1, 2, 3, 4, 5]:
            raise Exception('Invalid problem instance')
            
        self.problem_index = problem_index
        
        # problem 3 is only defined on 2-D decision space
        if problem_index == 3:
            self.dim = 2
        else:
            self.dim = dim
        
        # Bi-objective problems
        if self.problem_index in [1, 2, 3, 4, 5]:
            self.lb = [0] * self.dim
            self.ub = [1] * self.dim
            self.f_dim = 2
            self.ref = [1.1, 1.1]
            
        # Generalized Schaffer Problem
        if self.problem_index == 5:
            self.alpha = alpha
            
    
    def __str__(self):
        pass
    
    def __check_input(self, x):
        x = np.atleast_2d(x) 
        x = x.T if x.shape[0] != 1 else x
                
        return x
        
    def objfun(self, x):
        x = self.__check_input(x)
        
        # double sphere problem
        if self.problem_index == 1:
            center1 = zeros(self.dim)
            center2 = zeros(self.dim)
            center1[1] = 0.5
            center2[0:2] = [1, 0.5]
        
            return (_Fsphere(x, center1), _Fsphere(x, center2)) 
            
        elif self.problem_index == 2:
            center1 = zeros(self.dim)
            center2 = zeros(self.dim)
            center3 = zeros(self.dim)
            center1[1] = 0.5
            center2[0:2] = [1, 0.5]
            center3[0:2] = [0.5, 1] 
        
            return (_Fsphere(x, center1) + _Fsphere(x, center2), _Fsphere(x, center3)) 
            
        elif self.problem_index == 3:
            x0, x1 = x.flatten()
            f1 = 1.0 - (1 - x0 ** 2.0) * (x1 **2.0)
            f2 = 1.0 - (1 - x1 ** 2.0) * (x0 **2.0)
            return f1, f2
        
        # EBN problem
        elif self.problem_index == 4:
            pass
        
        # Generalized Schaffer problem
        elif self.problem_index == 5:
            center1 = zeros(self.dim)
            center2 = ones(self.dim)
            return (_Fsphere(x, center1) ** self.alpha / (self.dim ** self.alpha), 
                    _Fsphere(x, center2) ** self.alpha / (self.dim ** self.alpha))
    
    
    def objfun_dx(self, x):
        x = self.__check_input(x)
        
        if self.problem_index == 1:
            center1 = zeros(self.dim)
            center2 = zeros(self.dim)
            center1[1] = 0.5
            center2[0:2] = [1, 0.5]
            
            return np.r_[_Fsphere_gradient(x, center1), _Fsphere_gradient(x, center2)]
            
        elif self.problem_index == 2:
            center1 = zeros(self.dim)
            center2 = zeros(self.dim)
            center3 = zeros(self.dim)
            center1[1] = 0.5
            center2[0:2] = [1, 0.5]
            center3[0:2] = [0.5, 1]
            
            return np.r_[_Fsphere_gradient(x, center1) + _Fsphere_gradient(x, center2),
                         _Fsphere_gradient(x, center3)]
                         
        elif self.problem_index == 3:
            x0, x1 = x.flatten()
            return array([[2.0 * x0 * x1 ** 2.0, -2.0 * x1 + 2.0 * x1 * x0 ** 2.0],
                         [-2.0 * x0 + 2.0 * x0 * x1 ** 2.0, 2.0 * x1 * x0 ** 2.0]])
                         
        elif self.problem_index == 5:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                center1 = zeros(self.dim)
                center2 = ones(self.dim)
                try:
                    c1 = self.alpha * _Fsphere(x, center1) ** (self.alpha - 1) / \
                        (self.dim ** self.alpha)
                except Warning:
                    c1 = 0
                try:
                    c2 = self.alpha * _Fsphere(x, center2) ** (self.alpha - 1) / \
                        (self.dim ** self.alpha)
                except Warning:
                    c2 = 0
                
            return np.r_[_Fsphere_gradient(x, center1) * c1,
                         _Fsphere_gradient(x, center2) * c2]
                         
                         
    def pareto_front(self):
        
        if self.problem_index == 1:
            return array([[0, 1]]), lambda f1: (sqrt(f1) - 1.) ** 2.0
        elif self.problem_index == 2:
            pass
        elif self.problem_index == 3:
            pass
        elif self.problem_index == 5:
            gamma = 0.5 / self.alpha
            return lambda f1: (1.0 - f1 ** gamma) ** (1.0 / gamma)
            
            
    def check_input(self, x):
        x = np.atleast_2d(x)
        x = x.T if x.shape[1] != self.f_dim else x
        
        return x
            




def _FEBN(x, gamma):
    pass
        
def _Fsphere(x, center):
    return np.sum((x - center) ** 2.0)
    
def _Fsphere_gradient(x, center):
    return 2.0 * (x - center)
