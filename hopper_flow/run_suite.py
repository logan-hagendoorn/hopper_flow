import os
import subprocess

#reset counter2, the current simulation within the suite
with open("counter2.txt", "w") as f:
    f.write("0")

#increment counter1, the current suite iteration
with open("counter1.txt", "r") as f:
    current_counter1 = int(f.read())
current_counter1 += 1
with open("counter1.txt", "w") as f:
    f.write(str(current_counter1))


#make a new directory for csv and video outputs for this iteration
os.makedirs("csv_outputs/iteration_" + str(current_counter1), exist_ok=False)
os.makedirs("video_outputs/iteration_" + str(current_counter1), exist_ok=False)
os.makedirs("plots/iteration_" + str(current_counter1), exist_ok=False)


pi = 3.141592653589793


dt_fixed = 1 / 36000                                    #timestep
substeps = int((1 / 60) // dt_fixed)                    #number of substeps per frame, using 60 fps
dia = 0.001                                             #grain diameter
den = 2500.0                                            #grain density
cellParam = 100                                         #scales the size of the spatial partitioning grid (cells are dia x dia x dia)
n = 200000                                              #number of grains
k = 50.0                                                #this simulation uses a spring-dashpot model for normal forces; this is the spring constant
muK = 0.125                                             #kinetic friction coefficient between grains
muKSurface = 0.15                                       #kinetic friction coefficient between grains and surfaces
muSSurface = 0.4                                        #static friction coefficient between grains and surfaces
elastic = float(False)                                  #0.0 for inelastic collisions, 1.0 for elastic collisions
grain_mass = (4/3)*pi*(dia/2)**3 * den                  #mass of each grain
k_surface = k                                           #spring constant for surfaces
g = 9.81                                                #gravitational acceleration
cone_angle = 45 * pi / 180                              #pi / 2 - half angle of the cone in radians
critical_damping = 2 * (grain_mass * k_surface)**0.5    #dashpot damping coefficient for normal forces
k_t = 0.5 * k                                           #tangential spring constant for friction model
xi = 0.5                                                #damping ratio for tangential dashpot
c_t = 2 * xi * (grain_mass * k_t)**0.5                  #tangential damping


#combinations of parameters to run
param_combinations = []

dia_spacing = 0.25 #how closely spaced the datapoints should be
num_sims = 14      #how many simulations should be run
min_dia = 4.75     #the minimum diameter that should be used

for i in range(num_sims):
    param_combinations.append((dt_fixed, substeps, dia, den, cellParam, n, k, muK, muKSurface, muSSurface, elastic, grain_mass, k_surface, g, cone_angle, critical_damping, k_t, xi, c_t, dia * (min_dia + i * dia_spacing)))

#run the simulation for each combination
for i in range(len(param_combinations)):
    subprocess.run(["python", "main.py"] + [str(p) for p in param_combinations[i]])
