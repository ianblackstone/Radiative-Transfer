import matplotlib.pyplot as plt
import math
import random as rand

num_photons = 100000
num_bins = 10

escaped_mu = [0]*num_bins

tau_atm = 10

albedo = 1

def NewPhoton():
    x = 0
    y = 0
    z = 0
    
    return x, y, z

def scatter():
    cost = 2*rand.random()-1
    sint = math.sqrt(1-cost**2)
    phi = 2*math.pi*rand.random()
    cosp = math.cos(phi)
    sinp = math.sin(phi)
    
    return cost, sint, cosp, sinp

x, y, z = NewPhoton()

cost, sint, cosp, sinp = scatter()

for i in range(0,num_photons):
    
    abs_flag = 0
    
    while z < 1:
        tau = -math.log(rand.random())
        step = tau/tau_atm
        
        if rand.random() > albedo:
            z = 2
            abs_flag = 1
        
        cost, sint, cosp, sinp = scatter()
        
        x += step*sint*cosp
        y += step*sint*sinp
        z += step*cost
       
        if z < 0:
            x, y, z = NewPhoton()
        
    if abs_flag == 0:
        mu_bin = math.floor(cost*num_bins)
        escaped_mu[mu_bin] += 1
    
    x, y, z = NewPhoton()
    
dtheta = 1/num_bins
half = dtheta/2

midpoints = [0]*num_bins
norm_intensity = [0]*num_bins

for i in range(0,num_bins):
    midpoints[i] = math.acos(i*dtheta+half)
    norm_intensity[i] = escaped_mu[i]/(2*num_photons*math.cos(midpoints[i]))*num_bins

thetas = [18.2,31.8,41.4,49.5,56.6,63.3,69.5,75.3,81.4,87.1]
intensity = [1.224,1.145,1.065,0.9885,0.9117,0.8375,0.7508,0.6769,0.5786,0.5135]

thetas_rad = [0]*num_bins

for i,angle in enumerate(thetas):
    thetas_rad[i] = math.radians(thetas[i])

midpoints_deg = [0]*num_bins

for i in range(0,num_bins):
    midpoints_deg[i] = math.degrees(midpoints[i])

plt.scatter(midpoints,norm_intensity,color='blue')
plt.plot(thetas_rad,intensity,color='red')
plt.axes().set_aspect('equal')
plt.xlabel(r'$\theta$')
plt.xticks([0,math.pi/9,2*math.pi/9,3*math.pi/9,4*math.pi/9],labels=['0','20','40','60','80'])
plt.ylabel('intensity')
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
plt.title(r'escaping photons vs $\theta$ using the reemit boundary condition')