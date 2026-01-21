import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


iteration = "iteration_"        #iteration to analyze
avg_flow_rates = []             #data collection array
dia = 0.001                     #particle diameter
dia_spacing = 0.25              #spacing of measurements
num_sims = 9                    #number of simulations in this iteration
min_dia = 4.75                  #minimum diameter


os.makedirs(f"plots/{iteration}/", exist_ok=True)


diameters = np.arange(min_dia * dia, (min_dia + (num_sims - 1) * dia_spacing + 0.000000001) * dia, dia * dia_spacing)
big_flow = []

#calculate average flow rates for each simulation
for i in range(1, 1 + num_sims):
    flow_rates = np.genfromtxt(f"csv_outputs/{iteration}/sim_{i}.csv", delimiter=",", skip_header=1)
    n = 0
    sum = 0
    #range 1-15 as the simulation is currently set only to run until 15 datapoints are collected for faster data collection
    for j in range(1,15):
        sum += flow_rates[j]
        n += 1
    avg_flow_rates.append(sum / n)

#array for plotting flow rate over time
for i in range(1,1 + num_sims):
    flow_rates = np.genfromtxt(f"csv_outputs/{iteration}/sim_{i}.csv", delimiter=",", skip_header=1)
    big_flow.append(flow_rates)


#Beverloo equation-- use 0.65 * 2500 as density because the Beverloo equation uses bulk density
def func(D, exp, k, C):
    return C * 0.65 * 2500 * ((9.81)**0.5) * ((D - k * dia)**exp)


avg_flow_rates = np.array(avg_flow_rates)


#Perform curve fit
popt, pcov = curve_fit(func, diameters, avg_flow_rates, maxfev=10000)
perr = np.sqrt(np.diag(pcov))
exp, k, C = [round(popt[0],3),round(popt[1],3),round(popt[2],3)]
print('exp = ', exp, '+/-', round(perr[0],3))
print('k = ', k, '+/-', round(perr[1],3))
print('C = ', C, '+/-', round(perr[2],3))
xa = np.arange(0,8 * dia,0.05 * dia)

#Plot graph
plt.plot(diameters * 1000, avg_flow_rates * 1000, 'o')
plt.plot(xa * 1000, func(xa,*popt) * 1000, '-', label=f"Trendline: ${C}ρg^{{0.5}}(D{'+' if k < 0 else '-'} {abs(k)})^{{{exp}}}$")
plt.legend()
plt.xlabel("Aperture diameter (mm)")
plt.ylabel("Mass flow rate (g/s)")
plt.savefig(f"plots/{iteration}/W_vs_D.png", dpi=300, bbox_inches="tight")
plt.show()


#Plot second graph
t = np.arange(0,30,0.2)
for i in range(len(big_flow)):
    plt.plot(t[:len(big_flow[i])],1000 * big_flow[i],'o')
plt.xlabel("Time (s)")
plt.ylabel("Mass flow rate for each aperture size (g/s)")
plt.savefig(f"plots/{iteration}/W_vs_t.png",dpi=300, bbox_inches="tight")
plt.show()
