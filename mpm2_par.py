# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:23:23 2016

@author: wangronin
"""

import numpy as np
from pandas import read_csv
from mpm2 import createProblem

def vector_to_latex(v):
    latex_code = '$\\begin{pmatrix}'
    latex_code += ' \\\\ '.join(['{:.4f}'.format(_) for _ in v])
    latex_code += '\end{pmatrix}$'
    return latex_code
    
def matrix_to_latex(matrix):
    latex_code = '$\\begin{bmatrix}'
    latex_code += ' \\\\ '.join([' & '.join(['{:.4f}'.format(__) \
        for __ in line]) for line in matrix])
    latex_code += '\end{bmatrix}$'
    return latex_code


def MPM2_parameter_to_latex(dim, n_peaks1, n_peaks2, seed1, seed2, 
                            shape1='ellipse', shape2='ellipse'):
                                
    f1 = createProblem(n_peaks1, dim, 'random', seed1, True, shape1)
    f2 = createProblem(n_peaks2, dim, 'random', seed2, True, shape2)
    
    peaks_all = [f1.peaks, f2.peaks]
    n_obj = 2
    
    latex_code = '\\begin{table}[H]\n'
    latex_code += '\caption{Parameters of bi-objective problem based on \\textbf{spherical} MPM2 ' + \
        'functions: n\_peaks1={}, n\_peaks2={}, seed1={}, seed2={}}}\n'.format(n_peaks1, n_peaks2, seed1, seed2)
    latex_code += '\centering\n'
    latex_code += '\\begin{tabular}[h]{c|c||c|c|c|c|r}\n'
    latex_code += 'function &  peak  &  center $\mathbf{c}$ & height $h$ & radius $R$ & s & $\mathbf{D}$ \\\\ \n'
    latex_code += '\hline \hline \n'
    
    for i in range(n_obj):
        peaks = peaks_all[i]
        peak_code_list = []
        
        for j, p in enumerate(peaks):
            peak_code = '\\rule{0pt}{.75cm}'
            if j == 0:
                peak_code += '$f_{}$ & '.format(i+1)
            else:
                peak_code += ' & '
            peak_code += 'peak{} & '.format(j+1)
            peak_code += vector_to_latex(p) + ' & '
            peak_code += '{:.4f} & '.format(p.height)
            peak_code += '{:.4f} & '.format(p.radius)
            peak_code += '{:.4f} & '.format(p.shape)
            peak_code += matrix_to_latex(p.D) + ' \\\\[.5cm] \n'
            peak_code_list.append(peak_code)
        
        latex_code += '\cline{2-7} \n'.join(peak_code_list)
        latex_code += '\hline \n'
        
    latex_code += '\end{tabular} \n'
    latex_code +='\end{table} \n\n'

    return latex_code
    
    
def MPM2_parameter_extract(dim, n_peak, seed, shape='ellipse'):
    
    f = createProblem(n_peak, dim, 'random', seed, True, shape)
    
    f_pars = {'dim': dim}
    
    for i, peak in enumerate(f.peaks):
        prefix = 'p' + str(i+1)
        
        f_pars[prefix + '.height'] = peak.height
        f_pars[prefix + '.radius'] = peak.radius
        f_pars[prefix + '.shape'] = peak.shape
        f_pars[prefix + '.center'] = np.array(peak)
        f_pars[prefix + '.D'] = peak.D
    
    return f_pars
    

if __name__ == '__main__':
    
    df = read_csv('./setup2.csv')
    n_problem = df.shape[0]
    dim = 2
    
    par = df.loc[5]
    res = MPM2_parameter_extract(dim, par.n2, par.s2)
    
    # output MPM2 function parameters to Latex tables
    with open('./latex/MPM2.tex', 'w') as f:        
        
        for i in range(n_problem):
            par = df.loc[i]
            table = MPM2_parameter_to_latex(dim, par.n1, par.n2, par.s1, par.s2, 
                                            'sphere', 'sphere')
            f.write(table)
    