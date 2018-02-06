import sys
sys.path.append("..")
from TextToSolver import TextToSolver as tts
import numpy as np
import matplotlib.pyplot as plt

residual = ['x ** 2 + y ** 2 - z ** 2', 'y - 5 * x']
dep_vars = ['x', 'y']
ind_vars = ['z']
options = {'TOL_FUN': 1e-10, 'TOL_X': 1e-12}

sys = tts(dep_vars, residual, indep_vars=ind_vars)

ind_var = {'z': np.array([1, 2, 3, 4, 5])}
guess = {'x': -np.ones(ind_var['z'].shape),
        'y': np.ones(ind_var['z'].shape)}

solution, report = sys.solve(guess, indep_var_val=ind_var, input_options=options)

fig, ax = plt.subplots()
ax.plot(solution['x'], solution['y'],'or')
for ind, val in enumerate(ind_var['z']):
    ax.annotate('z = ' + str(val), (solution['x'][ind], solution['y'][ind]), ha='center')

plt.show()
