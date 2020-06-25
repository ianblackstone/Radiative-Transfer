# Imports
import numpy as np
import matplotlib.pyplot as plt
import time
# import math

start_time = time.time()

# Declare the number of bins, photons, thickness of the atmosphere, and albedo.
num_photons = 100000
num_bins = 10
num_atm = 100
albedo = 0.9
h = 1
hg = np.linspace(-0.9,0.9,num=5)
alpha = 0.3

tau_list = np.logspace(-2,2,num_atm, base=10)

# Flag for the boundary.  it can be 'absorb', 'reemit', or 'reflect'.
boundary = 'reflect'

# Flag for the scattering type.  'hg', 'iso', 'draine'
scatter = 'draine'

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
rand_mu = 0

# Create counters to track the 3 areas momentum will be in.
initial_momentum = 0
boundary_momentum = 0
escaped_momentum = 0

# Define the initial state of each photon.
#                            0        1      2       3          4         5
initial_state = np.array([x_init, y_init, z_init, phi_init, mu_init, lambda_init])

def GetZMomentum(mu_list,lambda_list):
    #######################################################################################
    # Calculate the z momentum of a group of photons.
    #
    # mu_list -- numpy array containing the values of mu for each photon
    # # lambda_list -- numpy array containing the wavelength of each photon.
    #######################################################################################
    momentum = np.sum(np.multiply(mu_list,lambda_list)*h)
    
    return momentum

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

def HGScatter(num_photons,g):
    ##########################################################################
    # Scatter photons using the Henyey-Greenstein scattering function.
    #
    # num_photons -- An integer giving number of photons to be scattered
    # g -- a number from -1 to 1 (0 excluded), that represents the "average"
    #      direction the photon is scattered into.
    ##########################################################################

    phi = 2*np.pi*np.random.rand(num_photons)
    s = 2*np.random.rand(num_photons) - 1  
    # This is the Henyey Greenstein relation for anisotropic scattering.
    mu = 1/(2*g)*(1+g**2-((1-g**2)/(1+g*s))**2)
    
    return phi, mu

def DraineScatter(num_photons,g,alpha):
    ##########################################################################
    # Scatter photons using the Henyey-Greenstein scattering function.
    #
    # num_photons -- An integer giving number of photons to be scattered
    # g -- a number from -1 to 1 (0 excluded), that represents the "average"
    #      direction the photon is scattered into.
    ##########################################################################
    
    phi = 2*np.pi*np.random.rand(num_photons)
    mu = np.empty(0)
    
    remaining_photons = num_photons
    
    while remaining_photons > 0:
        y = np.random.rand(remaining_photons) 
        mu_candidate = 2*np.random.rand(remaining_photons)-1
    
        pdf = 1/(4*np.pi)*(1 - g**2)/(1 + alpha*(1 + 2*g**2)/3)*(1+alpha*mu_candidate**2)/(1+g**2-2*g*mu_candidate)**(3/2)
        
        accepted_mu = np.where(pdf > y)[0]
        if len(accepted_mu)>0:
            remaining_photons -= len(accepted_mu)
            mu = np.append(mu,mu_candidate[accepted_mu])
    return phi, mu

# def Absorb(photon_state,albedo,collision_momentum):
def Absorb(photon_state,albedo):
    #######################################################################################
    # Determine whether any photons are absorbed.
    #
    # photon_state -- A numpy array consisting of [x,y,z,phi,theta,lambda] for each photon.
    # albedo -- a number giving the chance of scattering instead of absorbing at any interaction.
    #######################################################################################
    
    absorbed_photons = np.where(np.random.rand(len(photon_state)) > albedo)[0]
    
    if len(absorbed_photons) > 0:
        
        # collision_momentum += GetZMomentum(photon_state[absorbed_photons,4], photon_state[absorbed_photons,5])
        
        # Remove the absorbed photons
        photon_state = RemovePhotons(photon_state, absorbed_photons)
    
    # return photon_state, collision_momentum
    return photon_state

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

# def CheckBoundary(photon_state,boundary,boundary_momentum,collision_momentum):
def CheckBoundary(photon_state,boundary,boundary_momentum):
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
            
            boundary_momentum += GetZMomentum(photon_state[boundary_photons,4],photon_state[boundary_photons,5])
            
            # We need to reference the initial state of the photons to emit a new one.
            global initial_state
                    
            # Reset the absorbed photons
            photon_state[boundary_photons] = CreatePhotons(len(boundary_photons), initial_state, rand_mu)
        
        # A photon that reaches the boundary is absorbed, contributing both its initial momentum and its current
        # downward momentum.  The photon is then removed from the list.
        elif boundary =='absorb':
            
            boundary_momentum += GetZMomentum(photon_state[boundary_photons,4],photon_state[boundary_photons,5])
            
            # Remove the absorbed photons
            photon_state = RemovePhotons(photon_state, boundary_photons)
        
        # A photon that reaches the boundary is reflected back.  The z position and velocity are both flipped.
        elif boundary =='reflect':
            
            boundary_momentum -= GetZMomentum(photon_state[boundary_photons,4],photon_state[boundary_photons,5])
            # collision_momentum += GetZMomentum(photon_state[boundary_photons,4],photon_state[boundary_photons,5])
            
            # Reflect the photon position
            photon_state[boundary_photons,2] = -photon_state[boundary_photons,2]
            
            # Reflect the photon velocity
            photon_state[boundary_photons,4] = -photon_state[boundary_photons,4]
            
        else:
            print('No valid boundary condition specified, enter a boundary condition.')
    
    # return photon_state, boundary_momentum, collision_momentum
    return photon_state, boundary_momentum

def CheckForEscape(photon_state,tau_atm,escaped_momentum):
    #######################################################################################
    # This function checks for photons past the atmospheric boundary, then removes those photons
    # and logs the mu at escape.
    #
    # photon_state -- A Numpy array consisting of [x,y,z,phi,mu,lambda] for each photon.
    # tau_atm -- a number giving the height of the atmosphere.
    #######################################################################################
    
    escaped_photons = np.where(photon_state[:,2]>tau_atm)
    
    escaped_mu = photon_state[escaped_photons,4]
    
    escaped_momentum += GetZMomentum(escaped_mu,photon_state[escaped_photons,5])
    
    photon_state = RemovePhotons(photon_state, escaped_photons)
    
    return photon_state, escaped_mu, escaped_momentum

def RotateIntoLabFrame(photon_state,mu_prime):
    #######################################################################################
    # This takes the newly generated mu and rotates it into the lab frame for propogation.
    # Special cases have to be made for photons traveling along the z axis, complicating this
    # function somewhat.
    # 
    # photon_state -- A Numpy array consisting of [x,y,z,phi,mu,lambda] for each photon.
    # mu_prime -- A Numpy array containing the cosine of the angle of scattering in the photon frame
    #######################################################################################
    
    upward_moving = np.where(photon_state[:,4]==1)
    downward_moving = np.where(photon_state[:,4]==-1)
    all_others = np.where(np.abs(photon_state[:,4])!=1)
    
    if len(upward_moving[0]) > 0:
        photon_state[upward_moving,4] = mu_prime[upward_moving]
    if len(downward_moving[0]) > 0:
        photon_state[downward_moving,4] = -mu_prime[downward_moving]
    if len(all_others[0]) > 0:
        sin_theta = np.sqrt(1-mu_prime[all_others]**2)
        cos_phi = np.cos(photon_state[all_others,3])
        photon_state[all_others,4] = -np.sqrt(1-photon_state[all_others,4]**2) * sin_theta * cos_phi + photon_state[all_others,4]*mu_prime[all_others]
    
    return photon_state

def RunPhotons(tau_atm,num_photons,boundary,albedo,scatter,g):
    #######################################################################################
    # This function creates the initial photons and loops until the photons have all either
    # escaped or been absorbed.
    #
    # tau_atm -- a number giving the optical depth of the atmosphere.
    # num_photons -- The number of photons to start with
    # boundary -- The boundary condition to use. 'absorb', 'reemit', or 'reflect'.
    # albedo -- A number between 0 and 1 that represents the chance to scatter vs absorb.
    # scatter -- A flag for the type of scattering to use.
    # g -- a number from -1 to 1 (0 excluded), that represents the "average"
    #      direction the photon is scattered into.
    #######################################################################################
    
    
    # Set up an empty array to track the mu of escaped photons.
    escaped_mu = np.empty(0)
    
    # Generate new photons
    photon_state = CreatePhotons(num_photons,initial_state,rand_mu)
    
    # Create counters for the momentum.
    initial_momentum = GetZMomentum(photon_state[:,4],photon_state[:,5])
    escaped_momentum = 0
    boundary_momentum = 0
    
    # Create a tracker to count up the momentum transferred only in collisions.  This SHOULD
    # give the same number as the other method.
    # collision_momentum = 0
    
    # print(str(photon_state[:,2][0]) + ', ' + str(photon_state[:,4][0]))
    
    # Loop until all photons have escaped or been absorbed.
    while num_photons > 0:
        
        num_photons = len(photon_state)
        
        # Take a step
        photon_state = TakeStep(num_photons,photon_state)
        
        # print(str(photon_state[:,2][0]) + ', ' + str(photon_state[:,4][0]))
        
        # Check for photons that bounce under the origin.
        # photon_state, boundary_momentum, collision_momentum = CheckBoundary(photon_state,boundary,boundary_momentum,collision_momentum)
        photon_state, boundary_momentum = CheckBoundary(photon_state,boundary,boundary_momentum)

        num_photons = len(photon_state)
        
        # If all photons escape or absorb finish the run.
        # This must be done after each step where photons could dissapear
        # to avoid errors.
        if num_photons == 0:
            # print('boundary')
            break
        
        # Check if any photons escaped
        photon_state, mu_list, escaped_momentum = CheckForEscape(photon_state,tau_atm,escaped_momentum)
        
        # Add the mu list 
        escaped_mu = np.append(escaped_mu, mu_list[0], axis = 0)
        
        num_photons = len(photon_state)
        
        # If all photons escape or absorb finish the run.
        # This must be done after each step where photons could dissapear
        # to avoid errors.
        if num_photons == 0:
            # print('escaped')
            break
        
            
        # Check for absorption
        # photon_state, collision_momentum = Absorb(photon_state, albedo, collision_momentum)
        photon_state = Absorb(photon_state, albedo)
        
                
        num_photons = len(photon_state)
        
        # If all photons escape or absorb finish the run.
        # This must be done after each step where photons could dissapear
        # to avoid errors.
        if num_photons == 0:
            # print('absorbed')
            break
        
        # Get the initial momentum before scattering (final - initial).
        # collision_momentum += GetZMomentum(photon_state[:,4],photon_state[:,5])
        
        # Check if we want isotropic scattering, set it to hg scattering with a parameter of 0.
        if scatter=='iso':
            scatter = 'hg'
            g = 0
        
        # print('pre')
        # print(GetZMomentum(photon_state[:,4],photon_state[:,5]))
        # Scatter the photons
        if scatter == 'hg':
            if g == 0:
                phi, mu = IsoScatter(num_photons)
                photon_state[:,3] = phi
                photon_state[:,4] = mu
                
            else:
                phi, mu_prime = HGScatter(num_photons,g)
                photon_state[:,3] = phi
                photon_state = RotateIntoLabFrame(photon_state, mu_prime)
                
        if scatter == 'draine':
            if g == 0 and alpha == 0:
                phi, mu = IsoScatter(num_photons)
                photon_state[:,3] = phi
                photon_state[:,4] = mu
                
            else:
                phi, mu_prime = DraineScatter(num_photons,g,alpha)
                photon_state[:,3] = phi
                photon_state = RotateIntoLabFrame(photon_state, mu_prime)

        
        
        
        
        # Get the final momentum after scattering.
        # collision_momentum -= GetZMomentum(photon_state[:,4],photon_state[:,5])
        # print('post')
        # print(GetZMomentum(photon_state[:,4],photon_state[:,5]))
    
    # return escaped_mu, boundary_momentum, escaped_momentum, initial_momentum, collision_momentum
    return escaped_mu, boundary_momentum, escaped_momentum, initial_momentum

momentum_method_1 = np.zeros(num_atm)
momentum_method_2 = np.zeros(num_atm)

number_transferred = np.zeros(num_atm)

for g in hg:
    alb = albedo
    for atm_i, tau_atm in enumerate(tau_list):
        # Run the photons and return the mu at escape
        # escaped_mu, boundary_momentum, escaped_momentum, initial_momentum, collision_momentum = RunPhotons(tau_atm, num_photons, boundary, alb)
        escaped_mu, boundary_momentum, escaped_momentum, initial_momentum = RunPhotons(tau_atm, num_photons, boundary, alb, scatter,g)

        momentum_method_1[atm_i] = (initial_momentum - escaped_momentum + boundary_momentum)/initial_momentum
        # momentum_method_2[atm_i] = collision_momentum/initial_momentum
    
    plt.figure(1)
    plt.plot(np.log10(tau_list),np.log10(momentum_method_1),label=f"{g:.2f}")
    # plt.plot(np.log10(tau_list),momentum_method_2, label=alb)
    
plt.legend(title = "HG parameter")
plt.xlabel("Log Atmospheric depth")
plt.ylabel("Log Momentum Transferred")
plt.title("Momentum transferred")


# # get counts in each bin
# bins, counts = np.unique(np.floor(num_bins*escaped_mu),return_counts=True)
    
# dtheta = 1/num_bins
# half = dtheta/2

# midpoints = [0]*num_bins
# norm_intensity = [0]*num_bins

# for i in range(0,num_bins):
#     midpoints[i] = math.acos(i*dtheta+half)
#     norm_intensity[i] = counts[i]/(2*num_photons*math.cos(midpoints[i]))*num_bins

# thetas = [18.2,31.8,41.4,49.5,56.6,63.3,69.5,75.3,81.4,87.1]
# intensity = [1.224,1.145,1.065,0.9885,0.9117,0.8375,0.7508,0.6769,0.5786,0.5135]

# thetas = np.radians(thetas)

# plt.figure(2)
# plt.axes().set_aspect('equal')
# plt.scatter(midpoints,norm_intensity)
# plt.plot(thetas,intensity,color='red')
# plt.xlabel(r'$\theta$')
# plt.xticks([0,np.pi/9,2*np.pi/9,3*np.pi/9,4*np.pi/9],labels=['0','20','40','60','80'])
# plt.ylabel('intensity')
# plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
# plt.title(r'escaping photons vs $\theta$ using the ' + boundary + ' boundary condition')

finish_time = time.time()
print('time taken: ' + str(finish_time - start_time))

# print('integral method: ' + str(momentum_method_1))
# print('delta method: ' + str(momentum_method_2))