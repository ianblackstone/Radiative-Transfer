import cupy as cp
import matplotlib.pyplot as plt
import time

start_time = time.time()

# atmosphere depth, numpy array from 0 to 10 with N_atm evenly log spaced samples
num_atm = 1
tau_atm = cp.logspace(1,1,num_atm,base=10)

# The number of photons to simulate for each optical depth
num_photons = 1000

# An array to track the mu at escape.
escaped_mu = []

# Henyeye-Greenstein parameters to use.  Eventually these will need to be calculated from the grain size and dust composition.
g = [-1,-0.5,0.001,0.5,1]

# Define an initial state for each photon.  They are upward moving and start at (0,0,0).  Wavelength currently not used but
# included for future use, not currently used.
initial_state = cp.array([0,0,0,0,0,1])

# Initialize the containing list for all the photons.
photon_state = cp.array([initial_state,]*num_photons,dtype=float)

# Planck's constant, for momentum calculations when we care about wavelength.
h = 1

# The albedo of the cloud, defined as the probability of scattering.  A photon that does not scatter will absorb.
albedo = 1

# Flag for the boundary.  it can either 'absorb', 'reemit', or 'reflect'.
boundary = 'reflect'

total_momentum = float(cp.sum(h/photon_state[:,5]))

# def AcceptReject(dist):
    #####################################################################################
    # A function to perform the acceptance/rejection of points for anisotropic scattering

    #####################################################################################

def GetStepSize(photon_state):
    step = -cp.log(1-cp.random.rand(len(photon_state)))
    return step


def TakeStep(photon_state, step):
    ####################################################################################
    # This function steps each of the photons forward in the direction they scattered.
    #
    # photon_state -- A CuPy array consisting of [x,y,z,phi,theta,lambda] for each photon.
    # theta -- an angle in radians, measuring the angle between the z axis and the photon
    #           velocity in the lab frame.
    # phi -- The azimuthal angle in the lab frame.
    ####################################################################################

    # Give friendly names to theta and phi for ease of use here.
    phi = photon_state[:,3]
    theta = photon_state[:,4]
    
    # Update the x coordinate
    photon_state[:,0] += step*cp.sin(theta)*cp.cos(phi)
    
    # Update the y coordinate
    photon_state[:,1] += step*cp.sin(theta)*cp.sin(phi)
    
    # Update the z coordinate
    photon_state[:,2] += step*cp.cos(theta)
    
    return photon_state
    
def IsoScatter(photon_state):
    ####################################################################################
    # This function takes the photons and scatters them off the dust in the atmosphere 
    # using an isotropic scattering method.
    # 
    # photon_state -- A CuPy array consisting of [x,y,z,phi,theta,lambda] for each photon.
    ####################################################################################
    
    # Theta is the polar angle with respect to the lab frame z-axis.
    theta = 2*cp.pi*cp.random.rand(len(photon_state))
    
    # Phi is the azimuthal angle, scattering should always be isotropic in phi.
    phi = 2*cp.pi*cp.random.rand(len(photon_state))
    
    return theta, phi


# def HGScatter(photon_state,g):
#     ####################################################################################
#     # This function takes the photons and scatters them off the dust in the atmosphere 
#     # using the Heneye-Greenstein phase function and the inverse CDF sampling method.
#     # 
#     # photon_state -- A CuPy array consisting of [x,y,z,phi,theta,lambda] for each photon.
#     # g -- A float giving the HG parameter to use.
#     ####################################################################################
    
    
#     # mu is the cosine of the polar angle with respect to the incoming photon direction before scattering.
#     # For testing we will just use isotropic scattering for now where mu will be determined by a random angle theta.
#     theta = 2*cp.pi*cp.random.rand(len(photon_state))
#     mu = cp.cos(theta)
    
#     # phi is the azimuthal angle, scattering should always be isotropic in phi.
#     phi = 2*cp.pi*cp.random.rand(len(photon_state))
    
def CheckIfEscaped(photon_state,tau_atm,escaped_mu):
    ######################################################################################
    # This checks if a photon has escaped, calculates the momentum that photon takes with it
    # then removes that photon from the list.
    #
    # photon_state -- A CuPy array consisting of [x,y,z,phi,theta,lambda] for each photon.
    # tau_atm -- The depth of the atmosphere.
    ######################################################################################
    
    # Looks for all the photons above the atmosphere.    
    escaped_photons = cp.where(photon_state[:,2]>=tau_atm)[0]
    momentum_transfer = 0
    
    if len(escaped_photons) > 0:   
        # Find the angles the particles escaped at
        escaped_mu += cp.cos(photon_state[escaped_photons,4]).tolist()
        
        # Calculates the momentum transferred, this is the difference between the initial z momentum and the final z momentum
        momentum_transfer = float(cp.sum(h*(-cp.cos(photon_state[escaped_photons,4]))/photon_state[escaped_photons,5]))
        
        # Remove the escaped photons
        photon_state = RemovePhotons(photon_state, escaped_photons)
    
    return photon_state, momentum_transfer, escaped_mu

def CheckBoundary(photon_state,boundary):
    #######################################################################################
    # This function handles the behavior at the emitting boundary.
    #
    # photon_state -- A CuPy array consisting of [x,y,z,phi,theta,lambda] for each photon.
    # boundary -- A string containing 'reemit', 'absorb', or 'reflect' which tells us what to
    #               do when a photon reaches the boundary.  See each case for a description.
    #######################################################################################
    
    # Find the photons that have bounced back under the origin.
    boundary_photons = cp.where(photon_state[:,2]<=0)[0]
    
    # The sum of the initial momentum and the extra downward momentum.
    momentum_transfer = float(cp.sum(h*(-cp.cos(photon_state[boundary_photons,4]))/photon_state[boundary_photons,5]))
   
    if len(boundary_photons)>0:
        # A photon that reaches the boundary is absorbed and replaced with a new photon in the initial state.
        # The photon momentum is counted as its initial momentum plus the downward momentum it is carrying.
        # This method DOES NOT have a set amount of momentum for the system like the other two so its use is
        # mostly for comparison with Wood, et al.
        if boundary == 'reemit':
            
            # We need to reference the initial state of the photons to emit a new one.
            global initial_state
                    
            # Reset the absorbed photons
            photon_state[boundary_photons] = cp.array([initial_state,]*len(boundary_photons),dtype=float)
        
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
            print('No valid boundary condition specified, enter a boundary condition.  Code is likely to give errors.')
    
    return photon_state, momentum_transfer
        
        
def Absorb(photon_state,albedo):
    ########################################################################################
    # This compares the albedo against a random number.  If the random number is larger the
    # photon is absorbed, contributing it's initial momentum to the cloud.  If it's smaller
    # the photon scatters
    #
    # photon_state -- A CuPy array consisting of [x,y,z,phi,theta,lambda] for each photon.
    # albedo -- A float from 0 to 1 that gives the probability of scatterring.  Using the 
    #           definition given in section 2.2 of Wood et al.
    ########################################################################################
    
    # Generate the list of absorbed photons.
    absorbed_photons = cp.where(cp.random.rand(len(photon_state))>albedo)[0]
    
    # Remove the absorbed photons
    photon_state = RemovePhotons(photon_state, absorbed_photons)
    
    return photon_state

def RemovePhotons(photon_state,removed_photons):
    ########################################################################################
    # A function to remove selected photons from he list so they no longer have to be tracked.
    # If this seems roundabout it's because cupy does not support the delete function of numpy.
    #
    # photon_state -- A CuPy array consisting of [x,y,z,phi,theta,lambda] for each photon.
    # removed_photons -- a python list giving the index of the photons to be removed. eg [0,4,82]
    ########################################################################################
    
    surviving_photons = []
    
    for photon in cp.arange(len(photon_state)):
        if ~cp.isin(photon,removed_photons):
            surviving_photons += [int(photon),]
    
    photon_state = photon_state[surviving_photons]
    
    return photon_state

while 1:
    # Determine the step size and take a step.
    # Commented out print commands were for troubleshooting.
    step = GetStepSize(photon_state)
    photon_state = TakeStep(photon_state,step)
    # print('After step, length is ' + str(len(photon_state)))
    
    # See if any of the photons escaped.
    photon_state, momentum_transfer, escaped_mu = CheckIfEscaped(photon_state,tau_atm,escaped_mu)
    total_momentum += momentum_transfer
    # print('After escape, length is ' + str(len(photon_state)))

    # We need to end the loop as soon as the last photon is removed, otherwise we get an error from CuPy trying
    # to get a logical index value from an empty array.
    if len(photon_state)==0:
        break
    
    # See if any of the photons went below the boundary.
    photon_state, momentum_transfer = CheckBoundary(photon_state,boundary)
    total_momentum += momentum_transfer
    # print('After boundary, length is ' + str(len(photon_state)))

    if len(photon_state)==0:
        break
   
    # Check for absorbtion.
    photon_state = Absorb(photon_state,albedo)
    # print('After absorb, length is ' + str(len(photon_state)))
    
    if len(photon_state)==0:
        break
    
    # 
    theta, phi = IsoScatter(photon_state)
    photon_state[:,3] += phi
    photon_state[:,4] += theta
    # print('After scatter, length is ' + str(len(photon_state)))


finish_time = time.time()
print('time taken: ' + str(finish_time - start_time))