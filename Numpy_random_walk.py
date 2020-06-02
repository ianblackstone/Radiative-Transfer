# Imports
import numpy as np
import matplotlib.pyplot as plt
import time
import math

start_time = time.time()

# Declare the number of bins, photons, thickness of the atmosphere, and albedo.
num_photons = 1000000
num_bins = 10
tau_atm = 10
albedo = 1

# Flag for the boundary.  it can be 'absorb', 'reemit', or 'reflect'.
boundary = 'reemit'

# Initial values, defined this way to make the photon state more explicit.
# Once this is working I want to incorporate pandas so the variables can be more
# explicitly called.
x_init = 0
y_init = 0
z_init = 0
phi_init = 0
mu_init = 1
lambda_init = 1

# randomize mu from 0 to 1 for new photons instead of uniformly upward moving photons.
rand_mu = 1

# Define the initial state of each photon.
#                            0        1      2       3          4         5
initial_state = np.array([x_init, y_init, z_init, phi_init, mu_init, lambda_init])

def CreatePhotons(num_photons,initial_state,rand_mu):
    ##########################################################################
    # Generates a number of new photons.
    #
    # num_photons -- An integer giving number of photons to be spawned during this operation.
    # initial_state -- A numpy array giving the state the photons should be spawned in.
    # rand_mu -- Boolean toggle to keep mu as it's initial value or randomize it between 0 and 1.
    ##########################################################################
    
    # Generate the new photons.
    photon_state = np.array([initial_state,]*num_photons,dtype=float)
    
    # Randomly set mu
    if rand_mu == 1:
        photon_state[:,4] = np.random.rand(num_photons)
        
    return photon_state

def IsoScatter(num_photons):
    ##########################################################################
    # Scatter photons isotropically.
    #
    # num_photons -- An integer giving number of photons to be scattered
    ##########################################################################
    
    # Generate a random phi and mu    
    phi = 2*np.pi*np.random.rand(num_photons)
    mu = 2*np.random.rand(num_photons) - 1
    
    return phi, mu

def TakeStep(num_photons,photon_state):
    
    # Generate a random step size
    step = -np.log(np.random.rand(num_photons))
    
    # find the sine of theta
    # sin_theta = np.sqrt(1-photon_state[:,4]**2)
    
    # Update x, y, and z.
    # For now let's just focus on z.
    # photon_state[:,0] += step*np.cos(photon_state[:,3])*sin_theta
    # photon_state[:,1] += step*np.sin(photon_state[:,3])*sin_theta
    photon_state[:,2] += step*photon_state[:,4]
    
    return photon_state

def RemovePhotons(photon_state,removed_photons):
    ########################################################################################
    # A function to remove selected photons from the list so they no longer have to be tracked.
    #
    # photon_state -- A numpy array consisting of [x,y,z,phi,theta,lambda] for each photon.
    # removed_photons -- a numpy array giving the index of the photons to be removed. eg [0,4,82]
    ########################################################################################
    
    photon_state = np.delete(photon_state,removed_photons, axis=0)
    
    return photon_state

def CheckBoundary(photon_state,boundary):
    #######################################################################################
    # This function handles the behavior at the emitting boundary.
    #
    # photon_state -- A Numpy array consisting of [x,y,z,phi,mu,lambda] for each photon.
    # boundary -- A string containing 'reemit', 'absorb', or 'reflect' which tells us what to
    #               do when a photon reaches the boundary.  See each case for a description.
    #######################################################################################
    
    # Find the photons that have bounced back under the origin.
    boundary_photons = np.where(photon_state[:,2]<=0)[0]
   
    if len(boundary_photons)>0:
        # A photon that reaches the boundary is absorbed and replaced with a new photon in the initial state.
        # The photon momentum is counted as its initial momentum plus the downward momentum it is carrying.
        # This method DOES NOT have a set amount of momentum for the system like the other two so its use is
        # mostly for comparison with Wood, et al.
        if boundary == 'reemit':
            
            # We need to reference the initial state of the photons to emit a new one.
            global initial_state
                    
            # Reset the absorbed photons
            photon_state[boundary_photons] = CreatePhotons(len(boundary_photons), initial_state, rand_mu)
        
        # A photon that reaches the boundary is absorbed, contributing both its initial momentum and its current
        # downward momentum.  The photon is then removed from the list.
        elif boundary =='absorb':
            
            # Remove the absorbed photons
            photon_state = RemovePhotons(photon_state, boundary_photons)
        
        # A photon that reaches the boundary is reflected back.  The z position and velocity are both flipped.
        elif boundary =='reflect':
            
            # Reflect the photon position
            photon_state[boundary_photons,2] = -photon_state[boundary_photons,2]
            
            # Reflect the photon velocity
            photon_state[boundary_photons,4] = -photon_state[boundary_photons,4]
            
        else:
            print('No valid boundary condition specified, enter a boundary condition.')
    
    return photon_state

def CheckForEscape(photon_state,tau_atm):
    #######################################################################################
    # This function checks for photons past the atmospheric boundary, then removes those photons
    # and logs the mu at escape.
    #
    # photon_state -- A Numpy array consisting of [x,y,z,phi,mu,lambda] for each photon.
    # tau_atm -- a number giving the height of the atmosphere.
    #######################################################################################
    
    escaped_photons = np.where(photon_state[:,2]>tau_atm)
    
    escaped_mu = photon_state[escaped_photons,4]
    
    photon_state = RemovePhotons(photon_state, escaped_photons)
    
    return photon_state, escaped_mu

def RunPhotons(tau_atm,num_photons,boundary,albedo):
    #######################################################################################
    # This function creates the initial photons and loops until the photons have all either
    # escaped or been absorbed.
    #
    # tau_atm -- a number giving the optical depth of the atmosphere.
    # num_photons -- The number of photons to start with
    # boundary -- The boundary condition to use. 'absorb', 'reemit', or 'reflect'.
    # albedo -- A number between 0 and 1 that represents the chance to scatter vs absorb.
    #######################################################################################
    
    # Set up an empty array to track the mu of escaped photons.
    escaped_mu = np.empty(0)
    
    # A flag to end the loop if the photon is absorbed (not used right now)
    # abs_flag = 0
    
    # Generate new photons
    photon_state = CreatePhotons(num_photons,initial_state,rand_mu)
    
    # Loop until all photons have escaped or been absorbed.
    while num_photons > 0:
        
        num_photons = len(photon_state)
        
        # Take a step
        photon_state = TakeStep(num_photons,photon_state)
        
        # Check if any photons escaped
        photon_state, mu_list = CheckForEscape(photon_state,tau_atm)
        
        # Add the mu list 
        escaped_mu = np.append(escaped_mu, mu_list[0], axis = 0)
        
        num_photons = len(photon_state)
        
        # If all photons escape or absorb finish the run.
        # This must be done after each step where photons could dissapear
        # to avoid errors.
        if num_photons == 0:
            break
        
        # Check for photons that bounce under the origin.
        photon_state = CheckBoundary(photon_state,boundary)
        
        num_photons = len(photon_state)
        
        # If all photons escape or absorb finish the run.
        # This must be done after each step where photons could dissapear
        # to avoid errors.
        if num_photons == 0:
            break
        
        # Check for absorption goes here when we enable it.
        
        # Scatter the photons
        phi, mu = IsoScatter(num_photons)
        
        photon_state[:,3] = phi
        photon_state[:,4] = mu
    
    return escaped_mu

# Run the photons and return the mu at escape
escaped_mu = RunPhotons(tau_atm, num_photons, boundary, albedo)

# get counts in each bin
bins, counts = np.unique(np.floor(num_bins*escaped_mu),return_counts=True)

dtheta = 1/num_bins
half = dtheta/2

midpoints = [0]*num_bins
norm_intensity = [0]*num_bins

for i in range(0,num_bins):
    midpoints[i] = math.acos(i*dtheta+half)
    norm_intensity[i] = counts[i]/(2*num_photons*math.cos(midpoints[i]))*num_bins


thetas = [18.2,31.8,41.4,49.5,56.6,63.3,69.5,75.3,81.4,87.1]
intensity = [1.224,1.145,1.065,0.9885,0.9117,0.8375,0.7508,0.6769,0.5786,0.5135]

thetas = np.radians(thetas)

plt.figure(2)
plt.axes().set_aspect('equal')
plt.scatter(midpoints,norm_intensity)
plt.plot(thetas,intensity,color='red')
plt.xlabel(r'$\theta$')
plt.xticks([0,np.pi/9,2*np.pi/9,3*np.pi/9,4*np.pi/9],labels=['0','20','40','60','80'])
plt.ylabel('intensity')
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
plt.title(r'escaping photons vs $\theta$ using the ' + boundary + ' boundary condition')

finish_time = time.time()
print('time taken: ' + str(finish_time - start_time))