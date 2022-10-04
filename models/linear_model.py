import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from sympy.parsing.sympy_parser import parse_expr
from sympy import latex 

from sympy import Symbol, latex, diff, sympify
from sympy.core.numbers import Float, Integer

class LinModel():
    def __init__(self, dataframe, xlabel='x', ylabel='y', label='', x_units='', y_units='', round_figures=3, rename_x=''):
        self.model_label = label 
        self.x_units = x_units
        self.y_units = y_units
        self.x_var = dataframe[xlabel].copy()
        self.y_var = dataframe[ylabel].copy()
        
        self.fit()
        self.lhs = sympify(ylabel + '(' + xlabel + ')')
        self.rhs = sympify('m *' + xlabel + ' + b') if rename_x == '' else sympify('m *' + rename_x + ' + b')
        self.pars = self.vars_stat_summary[['Parámetro', 'Valor estimado']].set_index('Parámetro')
        self.rhs_subs = self.rhs.subs(self.pars.transpose().round(round_figures).items())
        self.expression = str(self.lhs) + ' = ' + str(self.rhs_subs)
        self.ln_expression = latex(self.lhs) + ' = ' + latex(self.rhs_subs)
    
    def fit(self):
        def stat_error(std, dof):
            t_95 = t.isf(0.025, dof)
            return std*t_95

        coefficients, covariance_matrix = np.polyfit(self.x_var, self.y_var, 1, cov=True)
        self.slope, self.intercept = coefficients
        
        dof = len(self.x_var)-2 # degrees of freedom
        std_slope, std_intercept = np.abs(np.sqrt(np.diag(covariance_matrix)))
        unc_slope, unc_intercept = (stat_error(std_slope, dof), stat_error(std_intercept, dof))
        cov = covariance_matrix[0,1]
        
        x_min = self.x_var.min()
        x_max = self.x_var.max()
        #interval_length = x_max - x_min
        prediction_func = lambda x: self.slope*x + self.intercept
        
        y_predicted = prediction_func(self.x_var) # f_i = f(x_i)
        ss_res = np.sum((self.y_var - y_predicted)**2)
        ss_tot = np.sum((self.y_var - self.y_var.mean())**2)
                
        pearson_correlation = cov/std_intercept/std_slope
        coef_of_determination = 1 - ss_res/ss_tot
        
        # Falta añadir el error experimental, solo se reporta el estadístico!
        
        self.fit_summary = pd.Series(
            data={
                'Pearson correlation r': pearson_correlation,
                'Coefficient of determination R^2': coef_of_determination
            })
        
        
        self.vars_stat_summary = pd.DataFrame(
            data={
                'Parámetro' : ['m', 'b'], 
                'Valor estimado':[self.slope, self.intercept],
                'Error estándar': [std_slope, std_intercept], 
                'Incertidumbre':[unc_slope, unc_intercept]
            })
        
    def f_model(self, x):
        return self.slope*x + self.intercept
    
    def plot_model(self, ax=None, xlabel='', ylabel='', color = (0,0,0.8), legend=True, show_expression=True, data_label=''):
        label = self.model_label
        if type(ax) == type(None):
            _, ax = plt.subplots(1,1)
            
        x = self.x_var
        y = self.y_var

        x_min = x.min()
        x_max = x.max()
        
        x_grid = np.linspace(x_min, x_max, 100)
        y_model = self.f_model(x_grid)
        
        
        if xlabel == '':
            xlabel += x.name
        if ylabel == '':
            ylabel += y.name
        curve_label = str(ylabel) + ' = ' +str(self.rhs_subs) if show_expression is True else ''
        
        if self.x_units != '':
            xlabel +=  ' (' + self.x_units + ')'
        if self.y_units != '':
            ylabel += ' (' + self.y_units + ')'
                
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        ax.errorbar(x=x, y=y, marker = '.', ls='', ms='5', color = color, label=data_label)
        ax.plot(x_grid, y_model, color=(*color, 0.3), ls='-', label=curve_label)
        if legend is True:
            ax.legend()
