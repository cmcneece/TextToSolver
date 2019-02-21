#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:09:56 2018

@author: cmcneece
"""
from pandas import DataFrame
import numpy as np
import sympy as sp
from scipy.sparse import spdiags, vstack, hstack, csr_matrix
from scipy.sparse.linalg import  norm, spsolve
import pdb as pdb


class TextToSolver:
    """TextToSolver class, instatiates the system of equations object.

    This class accept a list of strings representing the residual of a system
    of equations. The class uses sympy to generate the analytic Jacobian and
    the system and appropriate lambdas for use in slope based solvers.

    Args:
        dep_vars (list): A list of strings representing the dependent variables
        residual (list): A list of strings representing the residual of the
            system of equations.

    Optional Args:
        indep_vars (list): A list of strings representing the independent
            variables.
        parameters: (dict): A dictionary holding the name of parameters in the
            system of equations as a key, and the value to be subsituted as the
            value.

    Returns:
        object: The TextToSolver object with tools for the solution of the
            system of equations.

    Attributes:
        residual_string (list): A list of strings representing the system of
            equations after parameter values have been substituted.
        variables_string (list): A list of strings representing the dependent
            variables of the system of equations.
        indep_variables_string (list): A list of strings representing the
            independent variables of the system of equations.
        parameters (dict): A dictionary holding the name of parameters in the
            system of equations as a key, and the value to be subsituted as the
            value.
        residual_symb (list): A list of the residual as symbolic functions.
        residual_func (list): A list of the residual as lambda functions
        jacoabian_string (DataFrame): A DataFrame of the Jacobian as a strings.
        jacobian_symb (DataFrame): A DataFrame of the Jacobian as a symbolic
            functions.
        jacobian_func (DataFrame): A DataFrame of the Jacobian as a lambda
            functions.
    """
    def __init__(self, dep_vars, residual, indep_vars=None, parameters=None):
        if indep_vars is None:
                indep_vars = []
        if parameters is None:
                parameters = []
        self.residual_string = residual
        self.variables_string = dep_vars
        self.indep_vars = indep_vars
        self.parameters = parameters
        self.generate_symbolic()
        self.functionalize()

    def get_variables(self):
        '''Return the variables as a list of strings.'''
        return self.variables_string

    def get_residual(self):
        '''Resturn the residual as a list of strings.'''
        return self.residual_symb

    def get_jacobian(self):
        '''Return the Jacobian as a DataFrame of strings.'''
        return self.jacobian_string

    def get_parameters(self):
        '''Return the parameters as a dictionary.'''
        return self.parameters

    def functionalize(self):
        '''Turn the symbolic residual and Jacobian into lambda functions.'''

        residual = self.residual_string
        variables = self.variables_string
        jacobian = self.jacobian_string

        # functionalize the residual
        residual_func = [sp.lambdify(self.indep_vars + variables, eq, np)
                         for eq in self.residual_string]

        # functionalize the jacobian
        jacobian_func = DataFrame(index=residual, columns=variables)
        for row in residual:
            for col in variables:
                jacobian_func.loc[row, col] = sp.lambdify(
                    self.indep_vars + variables, jacobian.loc[row, col], np)

        self.jacobian_func = jacobian_func
        self.residual_func = residual_func

    def generate_symbolic(self):
        '''Turn the string of residual and Jacobian into symbolic functions.'''

        # pull out residual information from self
        sys_string = self.residual_string
        var_string = self.variables_string
        parameters = self.parameters

        # generate symbolic variables
        # turn the variable string list into a list of symbolics
        var_symb = [sp.symbols(var) for var in var_string]

        # substitute the parameter values into the equation string
        num_eqs = len(sys_string)
        for key, val in parameters.items():
            for index in range(num_eqs):
                sys_string[index] = sys_string[index].replace(key, str(val))

        # turn the equation strings into symbolic functions
        sys_symb = [sp.sympify(eq) for eq in sys_string]

        # build the jacobian
        jacobian_symb = DataFrame(index=sys_string, columns=var_string)
        for var_ind, var in enumerate(var_symb):
            for eq_ind, eq in enumerate(sys_symb):
                jacobian_symb.loc[sys_string[eq_ind], var_string[var_ind]] = \
                 sp.diff(eq, var)

        # convert symbolic jacobian into strings
        jacobian_string = DataFrame(index=sys_string, columns=var_string)
        for row in sys_string:
            for col in var_string:
                jacobian_string.loc[row, col] = \
                 str(jacobian_symb.loc[row, col])

        # put information back into self
        self.jacobian_string = jacobian_string
        self.jacobian_symb = jacobian_symb
        self.residual_string = sys_string
        self.variables_string = var_string
        self.variables_symb = var_symb
        self.residual_symb = sys_symb

    def solve(self, guess, indep_var_val={}, input_options={}):
        '''Pass the independent variable values to the Newton solver.

        Solves the system of equations given an initial guess for the dependent
        variables. If independent variables are specified, the solver will
        evaluate the system of equations with the provided values.

        Args:
            guess (dict): A dictionary of the dependent variables as keys and
                a value of the initial guess as the value. If independent
                variables are specified, and are arrays, the guess value must
                be an array of the same shape.

        Optional Args:
            indep_vars (dict): A dictionary of the independent variables as
                keys and the values at which they are to be evaulated as
                values. The solver accepts single values of arrays.
            options (dict): A dictionary with options for the Newton solver.
                'DISPLAY': boolean, togles printing iter information (True)
                'TOL_X': tolerance in the Netwon step (1e-8)
                'TOL_FUN': tolerance of the residual function (1e-6)
                'MAX_ITER': maximum number of iterations before exiting (20)
                'ALPHA': parameter for decrease in step size
                'MIN_LAMBDA': minimum value of lamdba (0.1)
                'MAX_LAMBDA': maximum values of lambda (0.5)

        Returns:
            solution (dict): A dictionary containing the depedent variable
                names as keys, with the solution as values.
            report (dict): A dictionary with the diagnostic report of the
                solver including total iterations, value of the residual norm,
                the exit flag, and the resiprical condition of the Jacobian on
                the final step.
                    exit_flag:
                        2: step is too close to previous guess
                        1: solver converged
                        -1: Jacobian matrix may be singular
            '''

        x, report = self.newton_solver(guess, indep_var_val=indep_var_val,
                                       input_options=input_options)
        x_dict = self.package_x(x)

        return x_dict, report

    def package_x(self, x):
        '''
        Turn solution array into a dictionary sorted by variable.
        '''

        x_list = self.resegment_x(x)
        temp_x = {var: x_list[ind] for ind, var in
                  enumerate(self.variables_string)}

        return temp_x

    def newton_solver(self, guess, indep_var_val, input_options):
        '''Newton solver for the system of equations.'''

        options = {'DISPLAY': True, 'MAX_ITER': 20,
                   'TOL_FUN': 1e-6, 'TOL_X': 1e-8,
                   'ALPHA': 1e-4, 'MIN_LAMBDA': 0.1,
                   'MAX_LAMBDA': 0.5}
        for key, val in input_options.items():
            options[key] = val
        TOL_X = options['TOL_X']
        TOL_FUN = options['TOL_FUN']
        DISPLAY = options['DISPLAY']
        MAX_ITER = options['MAX_ITER']
        MIN_LAMBDA = options['MIN_LAMBDA']

        # get values out of the inputs
        guess, indep_var_val = self.unpack_inputs(guess, indep_var_val)
        x = np.vstack(guess).flatten()

        # evaluate residual and jacobian
        res, jac = self.evaluate_res_jac(guess, indep_var_val)
        f = np.dot(res.T, res)
        f2 = f

        # check if defined
        jac_sum = csr_matrix.sum(jac.tocsr())
        if np.isnan(jac_sum) or np.isinf(jac_sum):
            exit_flag = -1  # matrix may be singular
        else:
            exit_flag = 1  # normal exit

        # check the reciprocal condition, approximation to condition number
        resip_cond = 1/norm(jac)

        # check norm of residual
        res_norm = np.linalg.norm(res)

        # dummy values
        dx = np.zeros(res.shape)
        convergence = np.inf
        step_norm = np.inf

        # start counter
        iter_count = 0
        # backtracking
        lam = 1
        lam_1 = 1
        lam_2 = 1

        if DISPLAY:
            self.__print_newton_step(iter_count, res_norm, convergence,
                                     resip_cond, step_norm, first=True)
        # run the solver
        while (res_norm > TOL_FUN or lam < 1) and \
                exit_flag >= 0 and iter_count <= MAX_ITER:
            if lam == 1:
                # Newton-Raphson solver
                iter_count += 1  # increment counter
                dx = self.calculate_step(res, jac)  # calculate step
                g = res.T @ jac  # gradient of res_norm
                slope = np.dot(g, dx)  # slope of gradient
                f_old = np.dot(res.T, res)  # objective function
                x_old = x  # initial value
                lam_min = TOL_X/np.amax(np.abs(dx)/np.maximum(np.abs(x_old),
                                                              1))

            if lam < lam_min:
                exit_flag = 2  # x is too close to X_OLD
                break
            elif np.any(np.isnan(dx)) or np.any(np.isinf(dx)):
                exit_flag = -1  # matrix may be singular
                break

            x = x_old + dx * lam  # next guess
            x_list = self.resegment_x(x)
            res, jac = self.evaluate_res_jac(x_list, indep_var_val)
            f = np.dot(res.T, res)  # new objective function

            lam, lam_1 = self.__calculate_lambda(lam, f, f_old, slope, options,
                                                 f2, lam_1, lam_2)

            if lam < 1:
                lam_2 = lam_1
                f2 = f  # save 2nd most previous value
                lam = np.amax([lam, MIN_LAMBDA*lam_1])  # minimum step length
                continue

            res_norm_0 = res_norm  # old resnorm
            res_norm = np.linalg.norm(res)  # calculate new res_norm
            # calculate convergence rate
            convergence = np.log(res_norm_0/(res_norm + np.finfo(float).eps))
            step_norm = np.linalg.norm(dx)  # norm of the step

            # display
            if DISPLAY:
                self.__print_newton_step(iter_count, res_norm, convergence,
                                         resip_cond, step_norm, first=False)

            jac_sum = csr_matrix.sum(jac.tocsr())
            if np.isnan(jac_sum) or np.isinf(jac_sum):
                exit_flag = -1  # matrix may be singular
                break
            # check the reciprocal condition, approximation to condition number
            resip_cond = 1/norm(jac)
        if DISPLAY:
            print('\n')

        report = {'exit_flag': exit_flag, 'res_norm': res_norm,
                  'iter_count': iter_count, 'resip_cond': resip_cond}
        return x, report

    def __calculate_lambda(self, lam, f, f_old, slope, options, f2, lam_1,
                           lam_2):
        # check for convergence
        lam_1 = lam  # save previous lambda
        if f > f_old + options['ALPHA']*lam*slope:
            if lam == 1:
                lam = -slope/2/(f-f_old-slope)  # calculate lambda
            else:
                A = 1/(lam_1 - lam_2)
                B = np.asarray([[1/lam_1 ** 2, -1/lam_2 ** 2],
                               [-lam_2/lam_1 ** 2, lam_1/lam_2 ** 2]],
                               dtype=float)
                C = np.asarray([f-f_old-lam_1*slope, f2-f_old-lam_2*slope],
                               dtype=float)
                coeff = A * np.dot(B, C)
                a, b = coeff[0], coeff[1]
                if a == 0:
                    lam = -slope/2/b
                else:
                    discriminant = b ** 2 - 3*a*slope
                    if discriminant < 0:
                        lam = options['MAX_LAMBDA']*lam_1
                    elif b <= 0:
                        lam = (-b+np.sqrt(discriminant))/3/a
                    else:
                        lam = -slope/(b+np.sqrt(discriminant))

            # minimum step length
            lam = np.amin([lam, options['MAX_LAMBDA']*lam_1])
        elif np.isnan(f) or np.isinf(f):
            # limit undefined evaluation or overflow
            lam = options['MAX_LAMBDA']*lam_1
        else:
            lam = 1  # fraction of Newton step
        
        return lam, lam_1

    def __print_newton_step(self, iter_count, res_norm, convergence,
                            resip_cond, step_norm, first):
            if first:
                print('{:<10}{:>20}{:>20}{:>20}{:>20}'.format('iter_count',
                                                              'res_norm',
                                                              'convergence',
                                                              'resip_cond',
                                                              'step_norm'))
                print('-'*90)

            print('{:<10d}{:>20.2e}{:>20.2e}{:>20.2e}{:>20.2e}'.format(
                                                                iter_count,
                                                                res_norm,
                                                                convergence,
                                                                resip_cond,
                                                                step_norm))

    def unpack_inputs(self, guess, indep_var_val):
        '''Turn a list of variable value arrays into a single array.'''

        # unpack input
        guess_list = [np.array(guess[var]) for var in self.variables_string]
#        pdb.set_trace()
        ind_var_list = [np.array(indep_var_val[var]) for
                        var in self.indep_vars]

        return guess_list, ind_var_list

    def resegment_x(self, x):
        '''
        Turn an array variable value arrays a list of arrays segmented by
        variable.
        '''

        len_res = len(self.variables_string)
        len_input = len(x)/len_res
        x = np.reshape(x, (int(len_res), int(len_input)))
        temp_x = [x[i, :] for i in range(len_res)]

        return temp_x

    def evaluate_res_jac(self, guess, ind_var_val):
        '''Evaluate the residual and Jacobian at specified values.'''

        # determine size of system
        num_res = len(self.residual_string)
        num_val = 1 if (not ind_var_val) else len(ind_var_val[0])

        # build empty df for jacobian
        jac = DataFrame(index=self.residual_string,
                        columns=self.variables_string)

        # evaluate residual
        res = [eq(*ind_var_val, *guess) for eq in self.residual_func]
        residual = np.asarray(res).flatten()

        # evaluate jacobian
        for eq in self.residual_string:
            for var in self.variables_string:
                temp = self.jacobian_func.loc[eq, var](*ind_var_val,
                                                       *guess)*np.ones(num_val)
                jac.loc[eq, var] = spdiags(temp, 0, num_val, num_val)
        jac_mat = jac.values
        jacobian = [hstack(jac_mat[ind, :]) for ind in range(num_res)]
        jacobian = vstack(jacobian[:])

        return residual, jacobian

    def calculate_step(self, residual, jacobian):
        '''Calculate the Newton iteration step by inverting the Jacobian.'''

        dx = spsolve(-jacobian.tocsr(), residual)
        return dx
