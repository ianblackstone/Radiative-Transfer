# Imports
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from hoki import load

from read_BPASS import GetTauScaling


##############################################################################
# This code will start with a BPASS SED, downsample it, then generate N_photons
# sampled from the PDF of the SED.  Then it will release the photons to scatter
# and absorb with frequency dependent anisotropic scattering.
#
##############################################################################

## Flags and parameters
##----------------------------------------------------------------------------

# Set the reference wavelength
wl_ref = 0.665

# Set the BPASS timeslice.
time_slice = 6.0

# Declare the number of bins, photons, and atmospehres.
num_photons = 100000
num_bins = 10
num_atm = 300

wl_min = 0.001
wl_max = 10

grain_type = 'Sil'
grain_min = 0.001
grain_max = 1

tau_list = np.logspace(-3, 1, num_atm, base=10)

# Flag for the boundary.  it can be 'absorb', 'reemit', or 'reflect'.
boundary = 'reflect'

# Flag for the scattering type.  'hg', 'iso', 'draine'
scatter = 'hg'

# Boolean flags for randomizing mu at the start and turning off scattering.
rand_mu = 1
abs_only = 0

# monochromatic light, set to 0 to draw from an SED.
monochromatic = 0

# Select the BPASS file to use
BPASS_file = 'spectra-bin-imf135_300.z020.dat'

## Begin code to run the MC.
## -------------------------------------------------------------------------

start_time = time.time()

# Load the BPASS data
BPASS_data = load.model_output(BPASS_file)

# Convert the BPASS data to microns.
BPASS_data.WL *= 10**-4

BPASS_data = BPASS_data[ (BPASS_data.WL >= wl_min) & (BPASS_data.WL <= wl_max) ]

Grain_File = grain_type + ' Average Values.csv'

folder = 'Draine data ' + grain_type + '/'

kappa_data, kappa_av_RP, kappa_av_F = GetTauScaling(folder, grain_min, grain_max, BPASS_data.WL.to_numpy(), BPASS_data, str(time_slice), wl_ref)

kappa_data.WL = kappa_data.WL.round(4)

# kappa_file = 'lambda ' + str(wl_ref).replace('.', '_') + ' a ' + str(grain_min).replace('.', '_') + ' ' + str(grain_max).replace('.', '_') + ' ' + grain_type + ' time ' + str(time_slice) + '.csv'

Grain_data = pd.read_csv(Grain_File)
# Grain_data.WL = Grain_data.WL.round(4)

Grain_data.WL = Grain_data.WL.round(4)

# kappa_data = pd.read_csv(kappa_file)


Grain_data = pd.merge(Grain_data, kappa_data, left_on = 'WL', right_on = 'WL')


# Not doing absorption only right now.
# if abs_only == 1:
#     kappa = pd.read_csv('abs 1 kappa by wl.csv')
#     L_Edd = pd.read_csv('abs 1 L_Edd dataframe.csv')
# else:
    
# kappa = pd.read_csv('kappa by wl.csv', index_col=0)



## Generate photon counts
# -----------------------------------------------------

BPASS_data = BPASS_data.iloc[::100,:]

# BPASS_data.WL *= 10**-4

h = 6.6261*(10**(-27))

c = 2.99792458*(10**10)

# BPASS_data.iloc[:,1:-1] *= 10**8

photon = BPASS_data[str(time_slice)] * BPASS_data.WL**2 / (h*c)

# -----------------------------------------------------
    
## Find CDF
# -----------------------------------------------------

norm = np.trapz(photon, x = BPASS_data.WL)
CDF = np.zeros_like(BPASS_data.WL)

for i, _ in enumerate(BPASS_data.WL):
    phot = photon[0:i]
    WL = BPASS_data.WL[0:i]
    CDF[i] = np.trapz(phot, x = WL) / norm

# plt.plot(BPASS_data.WL, CDF)
# plt.ylabel('Continuous Distribution')
# plt.xlabel('Wavelength (micron)')
# plt.title('CDF by wavelength')
# plt.savefig('CDF.png', dpi = 200)

# -----------------------------------------------------



## Generate PDF from CDF.
# -----------------------------------------------------

PDF = np.gradient(CDF)

# Initial values, defined this way to make the photon state more explicit.
# Once this is working I want to incorporate pandas so the variables can be more
# explicitly called.
x_init = 0
y_init = 0
z_init = 0
phi_init = 0
mu_init = 1
lambda_init = 1
g_av = 0
albedo = 0
Q_ext = 0
tau_scale = 1


# Create counters to track the 3 areas momentum will be in.
initial_momentum = 0
boundary_momentum = 0
escaped_momentum = 0

# Define the initial state of each photon.
#                            0        1      2       3          4         5        6       7        8
initial_state = np.array([x_init, y_init, z_init, phi_init, mu_init, lambda_init, g_av, albedo, tau_scale])



# Grain_data['Kappa'] = kappa_data.Kappa
# Grain_data['Scale'] = kappa_data.Scale



def AcceptReject(N, PDF, Range):
    ##########################################################################
    # This is a generic accept / reject function.  It takes a PDF and generates
    # N values from a range.  It then generates a random number between 0 and 1
    # for each value and rejects that chosen value if it is larger than the
    # value of the PDF at that point.  It repeats for any rejected values until
    # N accepted values have been found.  This is done with a list of all values
    # that could be accepted to reduce the interpolation that needs to be done.
    # Functionally this means it will be one of 100,000 wavelengths or 81 grain
    # sizes that are in the BPASS or Draine data sets.
    #
    # N -- An integer representing the number of values to return.
    # PDF -- A numpy array describing a probability distribution function.
    # Range -- A numpy array containing all possible values that could be accepted.
    ##########################################################################
    
    # Create an empty array 
    Accepted = np.empty(0)
    
    while len(Accepted) < N:
        
        candidate_values = np.random.randint(0, len(Range), N - len(Accepted))
        
        random_values = np.random.rand(len(candidate_values))
        
        mask = random_values < PDF[candidate_values]
        
        Accepted = np.append(Accepted, Range[candidate_values[mask]])
    
    return Accepted

# def GetDraineAverage(folder, wavelengths):
#     ##########################################################################
#     # This function calculates the MRN weighted average for <cos> and the albedo.
#     #
#     # folder -- string containing the folder where the Draine files are located.
#     # wavelengths -- a numpy array containing the wavelengths this will be ran at.
#     ##########################################################################
    
#     # Get the list of all the files in the Draine folder
#     Draine_list = os.listdir(folder)
    
#     # Create an empty array for holding the grain sizes
#     a = np.zeros(len(Draine_list))
#     g = np.zeros((len(wavelengths),len(Draine_list)))
#     Q_abs = np.zeros((len(wavelengths),len(Draine_list)))
#     Q_scatt = np.zeros((len(wavelengths),len(Draine_list)))
    
#     # Loop over each file in the folder.
#     for j, file in enumerate(Draine_list):
        
#         # Get full path to file
#         Draine_file = os.path.join(folder,file)
        
#         # Import the data from the file.
#         Draine_data = np.genfromtxt(Draine_file, skip_header=1)
        
#         # Read the grain size from the file name.
#         grain_size = file.split('_')[-2] + '.' + file.split('_')[-1]
#         # Store the grain size.
#         a[j] = float(grain_size)
    
#         Q_abs[:,j] = np.interp(wavelengths, np.flip(Draine_data[:,0]), np.flip(Draine_data[:,1]))
#         Q_scatt[:,j] = np.interp(wavelengths, np.flip(Draine_data[:,0]), np.flip(Draine_data[:,2]))
#         g[:,j] = np.interp(wavelengths, np.flip(Draine_data[:,0]), np.flip(Draine_data[:,3]))
    
#     # Get the norm of the MRN distribution integral
#     norm = np.trapz(a**(-3.5), x = a)
    
#     Sigma_abs_av = np.trapz( a**(-1.5) * np.pi * Q_abs, x = a, axis = 1)/norm
#     Sigma_scatt_av = np.trapz( a**(-1.5) * np.pi * Q_scatt, x = a, axis = 1)/norm
#     g_av = np.trapz( a**(-3.5) * g, x = a, axis = 1)/norm

#     Q_ext = np.trapz( a**(-3.5) * (Q_abs + Q_scatt), x = a, axis = 1)/norm

#     return Sigma_abs_av, Sigma_scatt_av, g_av, Q_ext

# folder = 'C:/Users/Ian/Documents/GitHub/Radiative-Transfer/Draine data Gra'

# Sigma_abs_av, Sigma_scatt_av, g_av, Q_ext = GetDraineAverage(folder, BPASS_data.WL.to_numpy())

# albedo = Sigma_scatt_av / (Sigma_scatt_av + Sigma_abs_av)

# DF = pd.DataFrame({'WL':BPASS_data.WL, 'Sigma-Abs':Sigma_abs_av, 'Sigma-Scatt':Sigma_scatt_av, 'g':g_av, 'Albedo':albedo, 'Q_ext': Q_ext})

# DF.to_csv('Gra Average Values.csv', index = False)

# # Plot the average sigma and g values.
# ###########################################################################
# fig, axs = plt.subplots(3,1, sharex = True)

# axs[0].plot(BPASS_data.WL.to_numpy(), g_av)
# axs[0].set_ylabel('<cos>')
# axs[2].set_xlabel('Wavelength (microns)')
# axs[0].set_xscale('log')
# # axs[0].set_title('g weighted by MRN distribution')

# axs[1].plot(BPASS_data.WL.to_numpy(), Sigma_abs_av)
# axs[1].set_ylabel(r'<$\sigma_{abs}$>')
# axs[1].set_yscale('log')
# # axs[1].xlabel('Wavelength (microns)')
# # axs[1].xscale('log')
# # axs[1].set_title(r'\sigma_{abs} weighted by MRN distribution')

# axs[2].plot(BPASS_data.WL.to_numpy(), Sigma_scatt_av)
# axs[2].set_ylabel(r'<$\sigma_{scatt}$>')
# axs[2].set_yscale('log')
# # axs[2].xlabel('Wavelength (microns)')
# # axs[2].xscale('log')
# # axs[2].set_title(r'\sigma_{scatt} weighted by MRN distribution')

# fig.tight_layout()
# plt.savefig('MRN averaged values.png', dpi = 200)
# ###########################################################################

# plt.plot(BPASS_data.WL.to_numpy(), Sigma_abs_av)
# plt.yscale('log')
# plt.xscale('log')



def GetZMomentum(mu_list,lambda_list):
    #######################################################################################
    # Calculate the z momentum of a group of photons.
    #
    # mu_list -- numpy array containing the values of mu for each photon
    # # lambda_list -- numpy array containing the wavelength of each photon.
    #######################################################################################
    momentum = np.sum(np.multiply(mu_list,1/lambda_list)*h)
    
    return momentum

def CreatePhotons(num_photons,initial_state,rand_mu, PDF, Range):
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
    
    if monochromatic == 0:
        # Draw random wavelengths using the accept/reject method and the PDF.
        photon_state[:,5] = AcceptReject(num_photons, PDF, Range)
    else:
        photon_state[:,5] = monochromatic
    
    # There's some weird machine error in here so we round to 4 decimal places
    photon_state[:,5] = photon_state[:,5].round(4)
    
    # # Set the data from the premade csv file for average g and albedo
    # for i, wl in enumerate(photon_state[:,5]):
    #     photon_state[i,6] = Grain_data.g[Grain_data.WL == Grain_data.WL[(Grain_data.WL - wl).abs().argsort()[0]]]
    #     photon_state[i,7] = Grain_data.Albedo[Grain_data.WL == Grain_data.WL[(Grain_data.WL - wl).abs().argsort()[0]]]
    #     photon_state[i,8] = Grain_data.Q_ext[Grain_data.WL == Grain_data.WL[(Grain_data.WL - wl).abs().argsort()[0]]]
    #     photon_state[i,9] = kappa_data.Scale[kappa_data.WL == kappa_data.WL[(kappa_data.WL - wl).abs().argsort()[0]]]
    
    lookup = pd.DataFrame({'WL': photon_state[:,5]})
    DF = pd.merge(lookup, Grain_data, left_on = 'WL', right_on = 'WL', how = 'left')
    
    photon_state[:,6] = DF.g.to_numpy()
    photon_state[:,7] = DF.Albedo.to_numpy()
    photon_state[:,8] = DF.Scale.to_numpy()
    
    # photon_state[:,6] = 0.9
    
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
    # g -- a pandas column containing a number from -1 to 1 (0 excluded), that 
    #      represents the "average" direction the photon is scattered into.
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
    # g -- a pandas column containing a number from -1 to 1 (0 excluded), that 
    #      represents the "average" direction the photon is scattered into.
    # alpha -- A number giving the Draine parameter.
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
def Absorb(photon_state):
    #######################################################################################
    # Determine whether any photons are absorbed.
    #
    # photon_state -- A numpy array consisting of [x,y,z,phi,theta,lambda] for each photon.
    # albedo -- a number giving the chance of scattering instead of absorbing at any interaction.
    #######################################################################################
    
    if abs_only == 0:
        absorbed_photons = np.where(np.random.rand(len(photon_state)) > photon_state[:,7])[0]
    else:
        absorbed_photons = np.where(np.random.rand(len(photon_state)) > 0)[0]
    
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

# def CheckBoundary(photon_state,boundary,boundary_momentum,collision_momentum, PDF, Range):
def CheckBoundary(photon_state,boundary,boundary_momentum, PDF, Range):
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
            photon_state[boundary_photons] = CreatePhotons(len(boundary_photons), initial_state, rand_mu, PDF, Range)
        
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
    
    escaped_photons = np.where(photon_state[:,2]>tau_atm*photon_state[:,8])
    
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

def RunPhotons(tau_atm, num_photons, boundary, scatter, PDF, Range):
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
    photon_state = CreatePhotons(num_photons,initial_state,rand_mu, PDF, Range)
    
    # Create counters for the momentum.
    initial_momentum = GetZMomentum(photon_state[:,4],photon_state[:,5])
    escaped_momentum = 0
    boundary_momentum = 0
    
    scatter_count = 0
    
    # Create a tracker to count up the momentum transferred only in collisions.  This SHOULD
    # give the same number as the other method.
    # collision_momentum = 0
    
    # print(str(photon_state[:,2][0]) + ', ' + str(photon_state[:,4][0]))
    
    step = 0
    
    # Loop until all photons have escaped or been absorbed.
    while num_photons > 0:
        # fname = 'run/step ' + str(step) + '.csv'
        
        # np.savetxt(fname, photon_state, delimiter=',')
        
        num_photons = len(photon_state)
        
        # Take a step
        photon_state = TakeStep(num_photons,photon_state)
        
        # print(str(photon_state[:,2][0]) + ', ' + str(photon_state[:,4][0]))
        
        # Check if any photons escaped
        photon_state, mu_list, escaped_momentum = CheckForEscape(photon_state,tau_atm,escaped_momentum)
        
        if num_photons == 0:
            # print('boundary')
            break
        
        # Check for photons that bounce under the origin.
        # photon_state, boundary_momentum, collision_momentum = CheckBoundary(photon_state,boundary,boundary_momentum,collision_momentum)
        photon_state, boundary_momentum = CheckBoundary(photon_state,boundary,boundary_momentum, PDF, Range)

        num_photons = len(photon_state)
        
        # If all photons escape or absorb finish the run.
        # This must be done after each step where photons could dissapear
        # to avoid errors.
        if num_photons == 0:
            # print('boundary')
            break
        
        # Check if any photons escaped
        photon_state, mu_list, _ = CheckForEscape(photon_state,tau_atm,escaped_momentum)
        
        # Add the mu list 
        escaped_mu = np.append(escaped_mu, mu_list[0], axis = 0)
        
        # fname = 'run/step ' + str(step) + ' escape mu.csv'
        
        # np.savetxt(fname, mu_list, delimiter=',')
        
        # fname = 'run/step ' + str(step) + ' not escaped.csv'
        
        # np.savetxt(fname, photon_state, delimiter=',')
        
        num_photons = len(photon_state)
        
        # If all photons escape or absorb finish the run.
        # This must be done after each step where photons could dissapear
        # to avoid errors.
        if num_photons == 0:
            # print('escaped')
            break
        
            
        # Check for absorption
        photon_state = Absorb(photon_state)
        
                
        num_photons = len(photon_state)
        
        g = 1
        
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
                phi, mu_prime = HGScatter(num_photons,photon_state[:,6])
                photon_state[:,3] = phi
                photon_state = RotateIntoLabFrame(photon_state, mu_prime)
                
                
        # if scatter == 'draine':
        #     if g == 0 and alpha == 0:
        #         phi, mu = IsoScatter(num_photons)
        #         photon_state[:,3] = phi
        #         photon_state[:,4] = mu
                
        #     else:
        #         phi, mu_prime = DraineScatter(num_photons,photon_state[:,6],alpha)
        #         photon_state[:,3] = phi
        #         photon_state = RotateIntoLabFrame(photon_state, mu_prime)

        
        # print('Escaped momentum: ' + str(escaped_momentum))
        
        # print('Boundary momentum: ' + str(boundary_momentum))
        
        scatter_count += num_photons
        
        step += 1
        
        
        
        # Get the final momentum after scattering.
        # collision_momentum -= GetZMomentum(photon_state[:,4],photon_state[:,5])
        # print('post')
        # print(GetZMomentum(photon_state[:,4],photon_state[:,5]))
    
    # return escaped_mu, boundary_momentum, escaped_momentum, initial_momentum, collision_momentum
    return escaped_mu, boundary_momentum, escaped_momentum, initial_momentum, scatter_count

# PDF = PDF[np.where(PDF > 10**-4)]
# Range = BPASS_data.WL.to_numpy()[np.where(PDF > 10**-4)]
Range = BPASS_data.WL.to_numpy()

momentum_method_1 = np.zeros(num_atm)
# escaped_mu = np.zeros_like(momentum_method_1)
scatter_count = np.zeros_like(momentum_method_1)

# momentum_method_2 = np.zeros(num_atm)

# number_transferred = np.zeros(num_atm)

for atm_i, tau_atm in enumerate(tau_list):
    # Run the photons and return the mu at escape
    escaped_mu, boundary_momentum, escaped_momentum, initial_momentum, scatter_count[atm_i] = RunPhotons(tau_atm, num_photons, boundary, scatter, PDF, Range)

    momentum_method_1[atm_i] = (initial_momentum - escaped_momentum + boundary_momentum)/initial_momentum
    # momentum_method_2[atm_i] = collision_momentum/initial_momentum

# plt.figure(1)

# tau_list = pd.concat([old[old.columns[1:]],DF])['tau']
# momentum_method_1 = pd.concat([old[old.columns[1:]],DF])['momentum']

# plt.plot(tau_list, momentum_method_1, label='Sil')

# plt.plot(tau_list, 1-np.exp(-tau_list ), label = r'$1-e^{-\tau}$')

DF = pd.DataFrame({'tau':tau_list, 'momentum':momentum_method_1})

L_Edd = pd.read_csv('L_Edd dataframe.csv', index_col=0)

# DF.to_csv('momentum transfer 0_6 all.csv')

# new = pd.read_csv('momentum transfer.csv')

ratio_lambda_f_lambda_rp = Grain_data.Kappa_F[Grain_data.WL.round(decimals=4) == wl_ref].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL.round(decimals = 4) == wl_ref].to_numpy()

if grain_type == 'Gra':
    ratio_f_lambda_rp = L_Edd.kappa_av_F_Gra[L_Edd.time == time_slice].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
    ratio_rp_lambda_rp = L_Edd.kappa_av_RP_Gra[L_Edd.time == time_slice].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()

elif grain_type == 'Sil':
    
    ratio_f_lambda_rp = L_Edd.kappa_av_F_Sil[L_Edd.time == time_slice].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
    ratio_rp_lambda_rp = L_Edd.kappa_av_RP_Sil[L_Edd.time == time_slice].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()

elif grain_type == 'SiC':
    ratio_f_lambda_rp = L_Edd.kappa_av_F_SiC[L_Edd.time == time_slice].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
    ratio_rp_lambda_rp = L_Edd.kappa_av_RP_SiC[L_Edd.time == time_slice].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()

else:
    print('Unknown grain type, this will not plot.')


# plt.plot(DF['tau'],1-np.exp(-DF['tau']), label = r'$1-e^{-\tau_{RP, \, \lambda = 0.6 \, \mu m}}$')
# plt.plot(DF['tau'],1-np.exp(-DF['tau']*ratio_lambda_f_lambda_rp[0]), label = r'$1-e^{-\tau_{F, \, \lambda = 0.6 \, \mu m}}$')
plt.plot(DF['tau']*ratio_rp_lambda_rp[0],1-np.exp(-DF['tau']*ratio_rp_lambda_rp[0]), label = r'$1-e^{-\tau_{RP}}$')
# plt.plot(DF['tau'],1-np.exp(-DF['tau']*ratio_f_06_rp[0]), label = r'$1-e^{-\tau_{F}}$')
# plt.plot(DF['tau'], DF['momentum'], label = 'MC result')
plt.plot(DF['tau']*ratio_rp_lambda_rp[0], DF['momentum'], label = 'MC result')

# plt.xlabel(r"Atmospheric depth ($\tau_{RP, \, \lambda = 0.6 \, \mu m}$)")
plt.xlabel(r"Atmospheric depth ($\tau_{RP}$)")
plt.ylabel("Momentum Transferred (fraction of initial)")
plt.xscale('log')
plt.yscale('log')
plt.title("Momentum transferred")
plt.legend()
plt.grid(which = 'both', linestyle = ':')
plt.tick_params(labelright=True, right = True, grid_color = 'black', grid_alpha = 0.3)

image_name = 'Momentum transfer ' + str(wl_ref).replace('.','-') + ' ' + grain_type + ' time slice ' + str(time_slice).replace('.','_') + '.png'

plt.savefig(image_name, dpi = 200)
## ---------------------------------------------------------------------------

## Plot the ratio of momentum transfers
## ---------------------------------------------------------------------------
# plt.plot(DF['tau']*ratio_rp_06_rp[0], (1-np.exp(-DF['tau']*ratio_rp_06_rp[0]))/DF['momentum'])
# plt.xscale('log')
# # plt.yscale('log')
# plt.ylim(1.5,2)
# plt.ylabel('ratio')
# plt.xlabel(r'$\tau_{RP}$')
# plt.savefig('momentum ratio thin.png', dpi = 200)
## ----------------------------------------------------------------------------

## Plot the taus
## ---------------------------------------------------------------------------
# kappa = pd.read_csv('kappa by wl.csv')
# L_Edd = pd.read_csv('L_Edd dataframe.csv')

# wl_ref = 0.6
# time_slice = 6.0


# ratio_06_f_06_rp = kappa.kappa_F_Sil[kappa.WL.round(decimals=4) == wl_ref].to_numpy()/kappa.kappa_RP_Sil[kappa.WL.round(decimals = 4) == wl_ref].to_numpy()
# ratio_f_06_rp = L_Edd.kappa_av_F_Sil[L_Edd.time == time_slice].to_numpy()/kappa.kappa_RP_Sil[kappa.WL.round(decimals = 4) == wl_ref].to_numpy()
# ratio_rp_06_rp = L_Edd.kappa_av_RP_Sil[L_Edd.time == time_slice].to_numpy()/kappa.kappa_RP_Sil[kappa.WL.round(decimals = 4) == wl_ref].to_numpy()

# plt.plot(DF.tau, DF.tau*ratio_06_f_06_rp, label = r'$\tau_{F, \, \lambda = 0.6 \, \mu m}$')
# plt.plot(DF.tau, DF.tau*ratio_rp_06_rp, label = r'$\tau_{RP}$')
# plt.plot(DF.tau, DF.tau*ratio_f_06_rp, label = r'$\tau_{F}$')
# plt.legend()
# plt.xlabel(r'$\tau_{(RP, \, \lambda = 0.6 \, \mu m)}$')
# plt.ylabel(r'$\tau$')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('tau scales.png', dpi = 200)


## ---------------------------------------------------------------------------

# thing = 1-np.exp(-tau_list)

# plt.plot(tau_list,Sil/thing)
# plt.xscale('log')
# # plt.yscale('log')
# plt.title(r'ratio of momentum transfer / $(1-exp(-\tau))$')
# plt.xlabel(r'$\tau$')
# plt.ylabel('ratio')
# plt.savefig('ratio.png', dpi=200)


# # # get counts in each bin
# # bins, counts = np.unique(np.floor(num_bins*escaped_mu),return_counts=True)
    
# # dtheta = 1/num_bins
# # half = dtheta/2

# # midpoints = [0]*num_bins
# # norm_intensity = [0]*num_bins

# # for i in range(0,num_bins):
# #     midpoints[i] = math.acos(i*dtheta+half)
# #     norm_intensity[i] = counts[i]/(2*num_photons*math.cos(midpoints[i]))*num_bins

# # thetas = [18.2,31.8,41.4,49.5,56.6,63.3,69.5,75.3,81.4,87.1]
# # intensity = [1.224,1.145,1.065,0.9885,0.9117,0.8375,0.7508,0.6769,0.5786,0.5135]

# # thetas = np.radians(thetas)

# # plt.figure(2)
# # plt.axes().set_aspect('equal')
# # plt.scatter(midpoints,norm_intensity)
# # plt.plot(thetas,intensity,color='red')
# # plt.xlabel(r'$\theta$')
# # plt.xticks([0,np.pi/9,2*np.pi/9,3*np.pi/9,4*np.pi/9],labels=['0','20','40','60','80'])
# # plt.ylabel('intensity')
# # plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
# # plt.title(r'escaping photons vs $\theta$ using the ' + boundary + ' boundary condition')

finish_time = time.time()
print('time taken: ' + str(finish_time - start_time))

# # print('integral method: ' + str(momentum_method_1))
# # print('delta method: ' + str(momentum_method_2))

## Test out what reference wavelength we should use
## -----------------------------------------------------------
# phot = AcceptReject(100000, PDF, Range)



# plt.hist(phot, bins=np.logspace(np.log10(min(Range)),np.log10(max(Range)), 50))
# # plt.axvline(x=0.6, color = 'red', label = r'0.6 $\, \mu$ m')
# # plt.axvline(x = phot.mean(), color = 'black', label = 'mean photon')
# plt.xlabel('Wavelength (microns)')
# plt.ylabel('count')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.01)
# # plt.legend()
# plt.savefig('photon hist log 2.png', dpi = 200)
## -----------------------------------------------------------

## Plot the mu at escape
## ---------------------------------------------------------------
# plt.hist(escaped_mu, bins = 100)
# plt.xlabel(r'$\mu$')
# plt.ylabel('count')
# plt.title(r'distribution of $\mu$ at escape for $\tau = 1$')
# plt.tick_params(labelright=True)
# plt.savefig('escape mu 0_6 tau 1.png', dpi = 200)
## ---------------------------------------------------------------

## Testing time
## ---------------------------------------------------------------
# start_time = time.time()

# lookup = pd.DataFrame( {'WL': photon_state[:,5]})
# DF = pd.merge(lookup, Grain_data, left_on = 'WL', right_on = 'WL', how = 'left')

# finish_time = time.time()
# print('time taken: ' + str(finish_time - start_time))
## --------------------------------------------------------------

## Plotting escape and remaining mu
## ----------------------------------------------------------------
# escape_0 = pd.read_csv('run/step 0 escape mu.csv', header=None).T
# escape_0 = escape_0.rename(columns={0:'mu'})

# remain_0 = pd.read_csv('run/step 0 not escaped.csv', header = None)
# remain_0 = remain_0.rename(columns = {0:'x', 1:'y', 2:'z', 3:'phi', 4:'mu', 5:'wl', 6:'g', 7:'albedo', 8:'scale'})

# escape_1 = pd.read_csv('run/step 1 escape mu.csv', header=None).T
# escape_1 = escape_1.rename(columns={0:'mu'})

# fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
# ax1.hist(escape_0.mu, bins = 200)
# ax1.set_title(r'$\mu$ at escape first step')
# ax2.hist(remain_0.mu, bins = 200)
# ax2.set_title(r'remaining $\mu$ on first step')
# ax3.hist(escape_1.mu, bins = 200)
# ax3.set_title(r'$\mu$ at escape second step')
# fig.tight_layout()
# fig.savefig('0_1 mu comparison.png', dpi = 200)

## ----------------------------------------------------------------

## Plot kappa for diffferent grain types
## ----------------------------------------------------------------


# plt.plot(kappa.WL,kappa.kappa_RP_SiC, label = 'SiC')
# plt.plot(kappa.WL,kappa.kappa_RP_Sil, label = 'Sil')
# plt.plot(kappa.WL,kappa.kappa_RP_Gra, label = 'Gra')
# plt.legend()
# plt.xscale('log')
# plt.xlabel('Wavelength')
# plt.yscale('log')
# plt.ylabel(r'$\kappa_{RP}$')
# plt.title(r'$\kappa$ by wavelength')

## ----------------------------------------------------------------

## Plot the tau scaling factors
## ----------------------------------------------------------------

# SiC = pd.read_csv('lambda 0_6 0_001 1 SiC.csv')
# Sil = pd.read_csv('lambda 0_6 0_001 1 Sil.csv')
# Gra = pd.read_csv('lambda 0_6 0_001 1 Gra.csv')

# plt.plot(SiC.WL, SiC.Scale, label = 'SiC')
# plt.plot(Sil.WL, Sil.Scale, label = 'Sil')
# plt.plot(Gra.WL, Gra.Scale, label = 'Gra')
# plt.xscale('log')
# plt.legend()
# plt.xlabel('Wavelength')
# plt.ylabel(r'$\tau$ scale')
# plt.savefig('tau scales.png', dpi = 200)


## ----------------------------------------------------------------


## Plot g by wl
## ----------------------------------------------------------------
# plt.plot(Grain_data.WL, Grain_data.g)
# plt.xscale('log')
# plt.xlabel('Wavelength (microns)')
# plt.ylabel('g')
# plt.savefig('g by wl.png', dpi = 200)
## ----------------------------------------------------------------