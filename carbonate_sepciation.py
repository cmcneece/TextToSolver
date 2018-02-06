import sys
sys.path.append("..")
from TextToSolver import TextToSolver as tts
import numpy as np
import matplotlib.pyplot as plt
import time

residual = ['-pKco2 + pH2CO3 - pPco2',
            '-pK1 + pHCO3 + pH - pH2CO3',
            '-pK2 + pCO3 + pH - pHCO3',
            '-pKw + pH + pOH',
            '-log(10 ** pCO2T) + log(10 ** pH2CO3 + 10 ** pHCO3 + 10 ** pCO3)']

ind_vars = ['pH']
dep_vars = ['pPco2', 'pHCO3', 'pCO3', 'pH2CO3', 'pOH']
parameters = {'pKw': -14, 'pKco2': -1.46, 'pK1': -6.35,
              'pK2': -10.33, 'pCO2T': -5}
options = {'DISPLAY': False}

sys = tts(dep_vars, residual, indep_vars=ind_vars,
          parameters=parameters)
n = 100
pH = np.linspace(-14, 0, n)
ind_var = {'pH': pH}

guess = {var: np.ones(pH.shape) for var in dep_vars}

solution, report = sys.solve(guess, indep_var_val=ind_var, input_options=options)

fig, ax = plt.subplots()
for var in solution:
    ax.plot(pH, solution[var])
plt.show()
