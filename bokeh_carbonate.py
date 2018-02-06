import sys
sys.path.append("..")
from TextToSolver import TextToSolver as tts
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure, show


residual = ['-pK1 + pCO3 + 2 * pH - pCO2',
            '-pK2 + pCO3 + pH - pHCO3',
            '-pKw + pH + pOH',
            '-log(10 ** pCO2T) + log(10 ** pCO2 + 10 ** pHCO3 + 10 ** pCO3)']

ind_vars = ['pH', 'pCO2T']
dep_vars = ['pHCO3', 'pCO3', 'pCO2', 'pOH']
parameters = {'pKw': -14, 'pK1': -16.681,
              'pK2': -10.33}
delta_H = {'pKw': -14, 'pKco2': -1.46, 'pK1': -5.738,
              'pK2': -3.561}

options = {'DISPLAY': False}

system = tts(dep_vars, residual, indep_vars=ind_vars,
             parameters=parameters)
n = 100
pH = np.linspace(-14, 0, n)
pco2t = -3
ind_var = {'pH': pH, 'pCO2T': pco2t*np.ones(pH.shape)}

guess = {var: np.ones(pH.shape) for var in dep_vars}

solution, report = system.solve(guess, indep_var_val=ind_var,
                                input_options=options)
solution['pH'] = pH

source = ColumnDataSource(data=solution)

plot = figure(plot_height=400, plot_width=600, title="carbonate speciation",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[-14, 0])
colors = ['red', 'cyan', 'green', 'blue', 'magenta']
for ind, var in enumerate(dep_vars):
    plot.line('pH', var, source=source, line_color=colors[ind], line_width=3,
              line_alpha=0.6)

plot.xaxis.axis_label = 'log10 [H^+]'
plot.yaxis.axis_label = 'log10 concentration'

# Set up widgets
text = TextInput(title="title", value='carbonate_speciation')
CO2T = Slider(title="log10 total carbon", value=pco2t, start=-5.0, end=5.0, step=0.1)

# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):

    # Get the current slider values
    ind_var_val = {'pH': pH, 'pCO2T': CO2T.value*np.ones(pH.shape)}
    new_sol, report = system.solve(guess, indep_var_val=ind_var_val,
                           input_options=options)
    new_sol['pH'] = pH
    source.data = new_sol

CO2T.on_change('value', update_data)

# Set up layouts and add to document
inputs = widgetbox(text, CO2T)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "bokeh_carbonate"

# show the plot
show(plot)