import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
import numpy.typing as npt
import csv
import os
import cv2
import sys

os.system("cls")


#increment iteration counter
with open("counter2.txt", "r") as f:
    current_counter2 = int(f.read())
current_counter2 += 1
with open("counter2.txt", "w") as f:
    f.write(str(current_counter2))

#find iteration number of suite
with open("counter1.txt", "r") as f:
    current_counter1 = int(f.read())


#initialize Taichi, the JIT compiler, using the gpu backend
ti.init(arch=ti.gpu)


#initialize video writer
video_path = os.path.join("video_outputs/iteration_" + str(current_counter1), "sim_" + str(current_counter2) + ".mp4")
video = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    60.0,
    (1800, 1080)
)


pi = 3.141592653589793


#data collection arrays and variables
flow_rates = [] #stores the flow rate at each second of the simulation
count = [0, 0]  #count[0] is the total number of particles that have flowed out at the previous second, count[1] is the current second

#simulation parameters (all in SI units)
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
critical_damping = 2 * ti.sqrt(grain_mass * k_surface)  #dashpot damping coefficient for normal forces
k_t = 0.5 * k                                           #tangential spring constant for friction model
xi = 0.5                                                #damping ratio for tangential dashpot
c_t = 2 * xi * ti.sqrt(grain_mass * k_t)                #tangential damping
aperture_diameter = 8 * dia                             #diameter of the cone's aperture
time_to_run = 6                                         #Time that each simulation should run before shutting down

#if simulation was run with parameters, override defaults
if len(sys.argv) > 1:
    args = [float(a) for a in sys.argv[1:]]
    (dt_fixed, substeps, dia, den, cellParam, n, k, muK, muKSurface, muSSurface, elastic, grain_mass, k_surface, g, cone_angle, critical_damping, k_t, xi, c_t, aperture_diameter) = args

#cast int parameters from float to int
substeps = int(substeps)
cellParam = int(cellParam)
n = int(n)


#derived quantities
z_cutoff = 1.0 * dia + aperture_diameter * (ti.tan(cone_angle)) / 2         #z-coordinate of the cone's aperture
num_triangles = (cellParam - 1) * (cellParam - 1) * 2                       #number of triangles making up the cone surface mesh
v_crit = 0.5 * muSSurface * g * (grain_mass / k_t)**0.5                     #velocity at which a slipping particle transitions to sticking



#taichi fields
cellsArr = ti.Vector.field(8, dtype=int, shape=(cellParam,cellParam,2*cellParam))   #spatial partitioning grid, each cell holds up to 8 particle indices
cell_counts = ti.field(dtype=int, shape=(cellParam,cellParam,2*cellParam))          #number of particles in each cell
propertiesArr = ti.Vector.field(2, dtype=float, shape=(n))                          #stores particle properties (not used currently, but if different sizes/densities
                                                                                    #are used this will be needed)
positions = ti.Vector.field(3, dtype=float, shape=(n))                              #particle positions
velocities = ti.Vector.field(3, dtype=float, shape=(n))                             #particle velocities
a_old = ti.Vector.field(3, dtype=float, shape=(n))                                  #particle accelerations for the first half of velocity verlet
a_new = ti.Vector.field(3, dtype=float, shape=(n))                                  #particle accelerations for the second half of velocity verlet
cone_surface = ti.Vector.field(3, dtype=float, shape=cellParam*cellParam)           #cone surface mesh vertices
color = ti.Vector.field(3, dtype=float, shape=cellParam*cellParam)                  #cone surface mesh vertex colors
indices = ti.field(int, shape=num_triangles*3)                                      #cone surface mesh triangle indices
disp_positions = ti.Vector.field(3, dtype=float, shape=(n))                         #positions of displayed particles (for visualization with some particles skipped)
particle_colors = ti.Vector.field(3, dtype=float, shape=(n))                        #particle colors for visualization (green = slow --> red = fast)
integrated_v_perp_surface = ti.Vector.field(3, dtype=float, shape=(n,2))            #integrated tangential velocity of each particle relative to surface 
                                                                                    #for friction model (0=cone, 1=cylinder)
old_positions = ti.Vector.field(3,dtype=float,shape=(n))                            #stores previous positions to detect leaving contact
kinetic_surface = ti.field(dtype=ti.i32, shape=(n))                                 #1 if particle n is currently slipping, 0 otherwise


#initializing the cone surface mesh
@ti.kernel
def initConeSurface():
    for i in cone_surface:
        cone_surface[i] = ti.Vector([i % cellParam * dia, i // cellParam * dia, (1 + ti.tan(cone_angle) * ti.sqrt((i % cellParam - 50)**2 + (i // cellParam - 50)**2)) * dia])
    for i, j in ti.ndrange(int(cellParam) - 1, int(cellParam) - 1):
        quad_id = (i * (int(cellParam) - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * int(cellParam) + j
        indices[quad_id * 6 + 1] = (i + 1) * int(cellParam) + j
        indices[quad_id * 6 + 2] = i * int(cellParam) + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * int(cellParam) + j + 1
        indices[quad_id * 6 + 4] = i * int(cellParam) + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * int(cellParam) + j


initConeSurface()




#initializing:
#  the particles (currently, they start in a cylindrical column above the cone)
#  the spatial partitioning grid (cellsArr)
@ti.kernel
def initArrs():
    count = 0
    z = 0.0
    while count < int(n):
        for y in range(12, 90):
            for x in range(12, 90):
                if (x*dia - 50*dia)**2 + (y*dia - 50*dia)**2 <= (33.5 * dia)**2:
                    if count < int(n):
                        positions[count] = ti.Vector([x*dia, y*dia, 42*dia + z* dia])
                        velocities[count] = ti.Vector([0.0, 0.0, 0.0])
                        count += 1
        z += 1.0
    for I in ti.grouped(cellsArr):
        cellsArr[I] = [-1,-1,-1,-1,-1,-1,-1,-1]
    for i in positions:
        for j in range(8):
            gx = int(positions[i][0] // dia)
            gy = int(positions[i][1] // dia)
            gz = int(positions[i][2] // dia)
            if (cellsArr[gx,gy,gz][j] == -1):
                cellsArr[gx,gy,gz][j] = i
                break



initArrs()




#mod of a vector
@ti.func
def mod(v):
    return ti.sqrt(v.dot(v))


#updates the spatial partitioning grid
@ti.func
def updateCells():
    # clear counts and cell slots
    for I in ti.grouped(cell_counts):
        cell_counts[I] = 0
        for s in range(8):
            cellsArr[I][s] = -1 #fix

    # insert particles into cells using an atomic counter per cell (prevents race conditions)
    for i in range(int(n)):
        gx = int(positions[i][0] // dia)
        gy = int(positions[i][1] // dia)
        gz = int(positions[i][2] // dia)

        # bounds check
        if gx < 0 or gy < 0 or gz < 0:
            continue
        nx = cellParam
        if gx >= nx or gy >= nx or gz >= 2*nx:      #if particle i is out of bounds, ignore it
            continue

        idx = ti.atomic_add(cell_counts[gx,gy,gz], 1)
        if idx < 8:
            cellsArr[gx,gy,gz][idx] = i





@ti.kernel
def fixed_update(): #deleted parameters

    #FIRST PASS: compute a(t) = a_old
    for i in range(int(n)): #n is number of particles
        #stores the old position to detect if a particle has left contact with a surface so its friction state can be reset
        old_positions[i] = positions[i]


        #initializes the total force on this particular particle to zero
        f_tot = ti.Vector([0.0,0.0,0.0])
        

        #calculates color of particle based on velocity magnitude (green=slow --> red=fast)
        speeds = 350 * mod(velocities[i])
        particle_colors[i] = ti.Vector([
            ti.min(1.0, ti.max(0.0, ti.select(
                speeds < 7, speeds / 7,
                ti.select(speeds <= 9, 1.0, 1.0)
            ))),
            ti.min(1.0, ti.max(0.0, ti.select(
                speeds < 7, 1.0,
                ti.select(speeds <= 9, 1.0,
                        ti.select(speeds <= 120, 1.0 - (speeds - 9) / (120 - 9), 0.0))
            ))),
            0.0
        ])


        #gravity
        f_tot -= ti.Vector([0.0,0.0,grain_mass * g])




        #cone interaction (friction + normal forces)
        if (cone_distance(positions[i]) < dia/2):
            # cone parameters
            apex = ti.Vector([50.0 * dia, 50.0 * dia, 1.0 * dia])
            tan_theta = ti.tan(cone_angle)

            # particle position
            pos = positions[i]
            r = ti.sqrt((pos[0]-apex[0])**2 + (pos[1]-apex[1])**2)

            # analytical gradient
            grad = ti.Vector([
                - (pos[0] - apex[0]) * tan_theta / r if r != 0 else 0.0,
                - (pos[1] - apex[1]) * tan_theta / r if r != 0 else 0.0,
                1.0
            ])

            # normalized normal vector
            normal = grad / mod(grad)

            # penetration depth
            penetration = dia/2 - cone_distance(pos)

            # normal force
            f_normal = ti.Vector([0.0,0.0,0.0])
            f_normal_elastic = k_surface * penetration * normal
            if (elastic == 0):
                v = velocities[i]
                f_normal = f_normal_elastic - critical_damping * (v.dot(normal)) * normal

                #friction:
                #tangential velocity relative to surface
                v_perp = v - normal * v.dot(normal)
                integrated_v_perp_surface[i,0] = ti.Vector([integrated_v_perp_surface[i,0][0]+v_perp[0]*dt_fixed * 0.5, 
                                                            integrated_v_perp_surface[i,0][1]+v_perp[1]*dt_fixed * 0.5, 
                                                            integrated_v_perp_surface[i,0][2]+v_perp[2]*dt_fixed * 0.5])
                f_friction = - k_t * integrated_v_perp_surface[i,0] - c_t * v_perp
                
                #if static friction is too high, then switch to kinetic friction
                if (mod(f_friction) > muSSurface * mod(f_normal_elastic)):
                    kinetic_surface[i] = 1
                    if (mod(v_perp) != 0):
                        f_friction = (-v_perp.normalized()) * muKSurface * mod(f_normal_elastic)
                    else:
                        f_friction = ti.Vector([0.0, 0.0, 0.0])
                f_tot += f_friction

            f_tot += f_normal




        #cylinder interaction (friction + normal forces)
        if (cylinder_distance(positions[i]) < dia/2):
            # cylinder parameters (use same units as positions)
            center = ti.Vector([50.0 * dia, 50.0 * dia])
            
            normal = ti.Vector([positions[i][0] - center[0], positions[i][1] - center[1], 0.0]).normalized() # outward normal

            # penetration depth
            penetration = dia/2 - cylinder_distance(positions[i])

            #normal/friction calculations are the same as above
            f_normal = ti.Vector([0.0, 0.0, 0.0])
            f_normal_elastic = - k_surface * penetration * normal
            if (elastic == 0):
                f_normal = f_normal_elastic - critical_damping * (velocities[i].dot(normal)) * normal #same as above
                v = velocities[i]
                v_perp = v - normal * v.dot(normal)
                integrated_v_perp_surface[i,1] = ti.Vector([integrated_v_perp_surface[i,1][0]+v_perp[0]*dt_fixed * 0.5, 
                                                          integrated_v_perp_surface[i,1][1]+v_perp[1]*dt_fixed * 0.5, 
                                                          integrated_v_perp_surface[i,1][2]+v_perp[2]*dt_fixed * 0.5])
                f_friction = - k_t * integrated_v_perp_surface[i,1] - c_t * v_perp
                if (mod(f_friction) > muSSurface * mod(f_normal_elastic)):
                    kinetic_surface[i] = 1
                    if (mod(v_perp) != 0):
                        f_friction = (-v_perp.normalized()) * muKSurface * mod(f_normal_elastic)
                    else:
                        f_friction = ti.Vector([0.0, 0.0, 0.0])
                f_tot += f_friction
            f_tot += f_normal




        #next 3 for loops are for the first three indices of cellsArr, xyz. these give the xyz coordinates of the 27
        #cells around grain i (including the cell of grain i)
        for x in range(int(positions[i][0] // dia) - 1, int(positions[i][0] // dia) + 2):
            for y in range(int(positions[i][1] // dia) - 1, int(positions[i][1] // dia) + 2):
                for z in range(int(positions[i][2] // dia) - 1, int(positions[i][2] // dia) + 2):

                    #this for loop loops through all the grains contained in cell xyz.
                    for j in range(8):
                        
                        #if grain j is both assigned (not negative one) and not the same grain as grain i,
                        #then this tests if grain i and j are in contact, and if so, calculates the force
                        #from grain j on grain i
                        if ((x >= 0 and x < cellParam and y >= 0 and y < cellParam and z >= 0 and z < 2*cellParam) and cellsArr[x,y,z][j] != -1 and cellsArr[x,y,z][j] != i):
                            xDiff = positions[i][0] - positions[cellsArr[x,y,z][j]][0]
                            yDiff = positions[i][1] - positions[cellsArr[x,y,z][j]][1]
                            zDiff = positions[i][2] - positions[cellsArr[x,y,z][j]][2]
                            
                            #vector connecting the centers of grains i and j
                            d = ti.Vector([xDiff,yDiff,zDiff])
                            if (mod(d) < dia):
                                #normal force
                                f_normal = -k* (mod(d) - dia) * d.normalized()
                                if (elastic == 0):
                                    #normal damping
                                    v = velocities[i]
                                    f_normal -= critical_damping * (v - velocities[cellsArr[x,y,z][j]]).dot(d.normalized()) * d.normalized()

                                    #friction (kinetic only)
                                    v_perp = v - f_normal * v.dot(f_normal / mod(f_normal)) / mod(f_normal)
                                    f_friction = ti.Vector([0.0,0.0,0.0])
                                    if (mod(v_perp) != 0):
                                        f_friction = (-v_perp.normalized()) * muK * mod(f_normal)
                                    f_tot += f_friction
                                f_tot += f_normal
                        elif ((x >= 0 and x < cellParam and y >= 0 and y < cellParam and z >= 0 and z < 2*cellParam) and cellsArr[x,y,z][j] == -1):
                            break
        

        #update acceleration of grain i
        a_old[i] = f_tot / grain_mass
        

        #if particle i was slipping but its velocity has dropped below v_crit, set it to sticking
        if (mod(velocities[i] + a_old[i] * 0.5 *dt_fixed) < v_crit and mod(velocities[i]) >= v_crit and kinetic_surface[i] == 1):
            integrated_v_perp_surface[i,0] = ti.Vector([0.0,0.0,0.0])
            integrated_v_perp_surface[i,1] = ti.Vector([0.0,0.0,0.0])
            kinetic_surface[i] = 0
    



    #v_new = v_old + 0.5 * a_old * dt <-- 0.5 because velocity is updated twice per timestep
    for i in range(int(n)):
        velocities[i] = velocities[i] + 0.5 * a_old[i] * dt_fixed
    

    #x_new = x_old + v_half * dt
    for i in range(int(n)):
        positions[i] = positions[i] + velocities[i] * dt_fixed
    

    updateCells()

    


    #SECOND PASS: compute a(t+dt/2) = a_new
    for i in range(int(n)): #n is number of particles
        if (cone_distance(old_positions[i]) < dia/2 and cone_distance(positions[i]) >= dia/2):
            integrated_v_perp_surface[i,0] = ti.Vector([0.0,0.0,0.0])
            kinetic_surface[i] = 0
        if (cylinder_distance(old_positions[i]) < dia/2 and cylinder_distance(positions[i]) >= dia/2):
            integrated_v_perp_surface[i,1] = ti.Vector([0.0,0.0,0.0])
            kinetic_surface[i] = 0


        f_tot = ti.Vector([0.0,0.0,0.0])
        

        #gravity
        f_tot -= ti.Vector([0.0,0.0,grain_mass * g])




        #cone interaction (friction + normal forces)
        if (cone_distance(positions[i]) < dia/2):
            # cone parameters
            apex = ti.Vector([50.0 * dia, 50.0 * dia, 1.0 * dia])
            tan_theta = ti.tan(cone_angle)

            # particle position
            pos = positions[i]
            r = ti.sqrt((pos[0]-apex[0])**2 + (pos[1]-apex[1])**2)

            # analytical gradient
            grad = ti.Vector([
                - (pos[0] - apex[0]) * tan_theta / r if r != 0 else 0.0,
                - (pos[1] - apex[1]) * tan_theta / r if r != 0 else 0.0,
                1.0
            ])

            # normalized normal vector
            normal = grad / mod(grad)

            # penetration depth
            penetration = dia/2 - cone_distance(pos)

            # normal force
            f_normal = ti.Vector([0.0,0.0,0.0])
            f_normal_elastic = k_surface * penetration * normal
            if (elastic == 0):
                v = velocities[i]
                f_normal = f_normal_elastic - critical_damping * (v.dot(normal)) * normal

                #friction:
                #tangential velocity relative to surface
                v_perp = v - normal * v.dot(normal)
                integrated_v_perp_surface[i,0] = ti.Vector([integrated_v_perp_surface[i,0][0]+v_perp[0]*dt_fixed * 0.5, 
                                                            integrated_v_perp_surface[i,0][1]+v_perp[1]*dt_fixed * 0.5, 
                                                            integrated_v_perp_surface[i,0][2]+v_perp[2]*dt_fixed * 0.5])
                f_friction = - k_t * integrated_v_perp_surface[i,0] - c_t * v_perp
                
                #if static friction is too high, then switch to kinetic friction
                if (mod(f_friction) > muSSurface * mod(f_normal_elastic)):
                    kinetic_surface[i] = 1
                    if (mod(v_perp) != 0):
                        f_friction = (-v_perp.normalized()) * muKSurface * mod(f_normal_elastic)
                    else:
                        f_friction = ti.Vector([0.0, 0.0, 0.0])
                f_tot += f_friction

            f_tot += f_normal




        #cylinder interaction (friction + normal forces)
        if (cylinder_distance(positions[i]) < dia/2):
            # cylinder parameters (use same units as positions)
            center = ti.Vector([50.0 * dia, 50.0 * dia])
            
            normal = ti.Vector([positions[i][0] - center[0], positions[i][1] - center[1], 0.0]).normalized() # outward normal

            # penetration depth
            penetration = dia/2 - cylinder_distance(positions[i])

            #normal/friction calculations are the same as above
            f_normal = ti.Vector([0.0, 0.0, 0.0])
            f_normal_elastic = - k_surface * penetration * normal
            if (elastic == 0):
                f_normal = f_normal_elastic - critical_damping * (velocities[i].dot(normal)) * normal #same as above
                v = velocities[i]
                v_perp = v - normal * v.dot(normal)
                integrated_v_perp_surface[i,1] = ti.Vector([integrated_v_perp_surface[i,1][0]+v_perp[0]*dt_fixed * 0.5, 
                                                          integrated_v_perp_surface[i,1][1]+v_perp[1]*dt_fixed * 0.5, 
                                                          integrated_v_perp_surface[i,1][2]+v_perp[2]*dt_fixed * 0.5])
                f_friction = - k_t * integrated_v_perp_surface[i,1] - c_t * v_perp
                if (mod(f_friction) > muSSurface * mod(f_normal_elastic)):
                    kinetic_surface[i] = 1
                    if (mod(v_perp) != 0):
                        f_friction = (-v_perp.normalized()) * muKSurface * mod(f_normal_elastic)
                    else:
                        f_friction = ti.Vector([0.0, 0.0, 0.0])
                f_tot += f_friction
            f_tot += f_normal




        #next 3 for loops are for the first three indices of cellsArr, xyz. these give the xyz coordinates of the 27
        #cells around grain i (including the cell of grain i)
        for x in range(int(positions[i][0] // dia) - 1, int(positions[i][0] // dia) + 2):
            for y in range(int(positions[i][1] // dia) - 1, int(positions[i][1] // dia) + 2):
                for z in range(int(positions[i][2] // dia) - 1, int(positions[i][2] // dia) + 2):

                    #this for loop loops through all the grains contained in cell xyz.
                    for j in range(8):
                        
                        #if grain j is both assigned (not negative one) and not the same grain as grain i,
                        #then this tests if grain i and j are in contact, and if so, calculates the force
                        #from grain j on grain i
                        if ((x >= 0 and x < cellParam and y >= 0 and y < cellParam and z >= 0 and z < 2*cellParam) and cellsArr[x,y,z][j] != -1 and cellsArr[x,y,z][j] != i):
                            xDiff = positions[i][0] - positions[cellsArr[x,y,z][j]][0]
                            yDiff = positions[i][1] - positions[cellsArr[x,y,z][j]][1]
                            zDiff = positions[i][2] - positions[cellsArr[x,y,z][j]][2]
                            
                            #vector connecting the centers of grains i and j
                            d = ti.Vector([xDiff,yDiff,zDiff])
                            if (mod(d) < dia):
                                #normal force
                                f_normal = -k* (mod(d) - dia) * d.normalized()
                                if (elastic == 0):
                                    #normal damping
                                    v = velocities[i]
                                    f_normal -= critical_damping * (v - velocities[cellsArr[x,y,z][j]]).dot(d.normalized()) * d.normalized()

                                    #friction (kinetic only)
                                    v_perp = v - f_normal * v.dot(f_normal / mod(f_normal)) / mod(f_normal)
                                    f_friction = ti.Vector([0.0,0.0,0.0])
                                    if (mod(v_perp) != 0):
                                        f_friction = (-v_perp.normalized()) * muK * mod(f_normal)
                                    f_tot += f_friction
                                f_tot += f_normal
                        elif ((x >= 0 and x < cellParam and y >= 0 and y < cellParam and z >= 0 and z < 2*cellParam) and cellsArr[x,y,z][j] == -1):
                            break


        #update acceleration of grain i
        a_new[i] = f_tot / grain_mass


        #if particle i was slipping but its velocity has dropped below v_crit, set it to sticking
        if (mod(velocities[i] + 0.5 * a_new[i] * dt_fixed) < v_crit and mod(velocities[i]) >= v_crit and kinetic_surface[i] == 1):
            integrated_v_perp_surface[i,0] = ti.Vector([0.0,0.0,0.0])
            integrated_v_perp_surface[i,1] = ti.Vector([0.0,0.0,0.0])
            kinetic_surface[i] = 0
        



    # v_new = v_half + 0.5 * a_new * dt
    for i in range(int(n)): #do i even need v_half as a separate variable????
        velocities[i] = velocities[i] + 0.5 * a_new[i] * dt_fixed

        
    updateCells()



#distance functions for cone and cylinder
@ti.func
def cone_distance(v):
    #cone centered at (50,50,1)
    d = 1000.0
    #if below the cone's aperture, 1000 is returned (i.e., no interaction)
    if (v [2] > z_cutoff):
        d = ((v[2]) - ti.tan(cone_angle) * ti.sqrt((v[0] - 50*dia)**2 + (v[1] - 50*dia)**2)) * ti.sin(cone_angle)
    return d

@ti.func
def cylinder_distance(v):
    #cylinder centered at (50,50) w radius 35
    d = ti.sqrt((v[0]-50*dia)**2 + (v[1]-50*dia)**2) - 35.0*dia
    return -d


#coutns the number of particles that have gone below z = 1.0 * dia (i.e., have flowed out of the hopper)
@ti.kernel
def count_flowed_particles() -> int:
    count = 0
    for i in range(int(n)):
        if positions[i][2] < 1.0 * dia:
            count += 1
    return count

#updates the displayed particle positions for visualization (currently, it shows a cross section of the hopper)
@ti.kernel
def updateDispPositions():
    for i in range(int(n)):
        #'''
        if positions[i][1] > 50*dia:
            disp_positions[i] = positions[i]
        else:
            disp_positions[i] = ti.Vector([0,0,0])

#captures the current frame from the Taichi window and writes it to the video
def capture_frame():
    img = window.get_image_buffer_as_numpy()
    img = (img * 255).astype(np.uint8)
    img = img[:, :, :3]
    img = img[:, :, ::-1]

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    video.write(img)


window = ti.ui.Window(name="hourglass_draft",res = (1800, 1080), fps_limit=60,pos=(0,50))
canvas = window.get_canvas()
scene = ti.ui.Scene()

camera = ti.ui.Camera()
camera.position(0.05,-0.1,0.14)   # start back from the particles
camera.lookat(0.05,0.05,0.05)
current_t = 0.0

with open("csv_outputs/iteration_" + str(current_counter1) + "/" + "sim_" + str(current_counter2) + ".csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Flow rate (kg / s)"])  # optional header

#main simulation loop, runs every frame
while window.running:

    #simulation substeps
    for i in range(int(substeps)):
        fixed_update()
    current_t += 1/60

    #each 0.4s, calculate and store the flow rate
    if (current_t % 0.3999999999999 < 0.0001):
        count[0] = count[1]
        count[1] = count_flowed_particles()
        flow_rates.append((count[1] - count[0]) * grain_mass / 0.4)
        print("Current flow rate (kg/s):", flow_rates[-1])
        with open("csv_outputs/iteration_" + str(current_counter1) + "/" + "sim_" + str(current_counter2) + ".csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([flow_rates[-1]])
    

    #if the simulation has run for the desired amount of time, end it.
    if (len(flow_rates) >= time_to_run // 0.4):
        video.release()
        exit()
    
    
    #camera
    camera.track_user_inputs(window, movement_speed=0.001, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    #lighting
    scene.point_light(pos=(140,60,210), color=(0.6,0.6,0.6))
    scene.ambient_light((0.2, 0.2, 0.2))

    # draw particles
    updateDispPositions()
    scene.particles(disp_positions, radius=dia * 0.7 / 2,per_vertex_color=particle_colors)

    #draw cone surface (not currently in use)
    #scene.mesh(cone_surface, indices, color=(0.5, 0.5, 0.5))


    capture_frame()
    canvas.scene(scene)
    window.show()


