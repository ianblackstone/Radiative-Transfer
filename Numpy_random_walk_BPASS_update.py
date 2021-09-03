# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import pandas as pd
from hoki import load
from scipy import integrate as inte

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
wl_ref = 0.656

# Set the BPASS timeslice.
time_slice = 6.0

# Declare the number of bins, photons, and atmospehres.
num_photons = 10000
num_bins = 10
num_atm = 100

wl_min = 0.001
wl_max = 10

grain_type = 'Sil'
grain_min = 0.001
grain_max = 1

tau_list = np.logspace(-4, 10, num_atm, base=10)

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

# cm per pc
cm_pc = 3.086*10**18

# g per solar mass
g_Msun = 1.988*10**33

h_old = 500
h_old = h_old * cm_pc

# Height of the gas column in parsecs
h_g = 5
# Convert to cgs
h_g = h_g * cm_pc

# Sigma g scaling factor
h_g_original = 5
h_g_original = h_g_original * cm_pc

# dust to gas ratio
f_dg = 1/100

# Define the files to read
gal_file_1 = 'NGC5194_Full_Aperture_table_2arc.dat'
gal_file_2 = 'M51_3p6um_avApertures_MjySr.txt'

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
G = 6.67259*(10**-8)

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
    # Old code style, needs changed before uncommenting.
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

# # PDF = PDF[np.where(PDF > 10**-4)]
# # Range = BPASS_data.WL.to_numpy()[np.where(PDF > 10**-4)]
# Range = BPASS_data.WL.to_numpy()

# momentum_method_1 = np.zeros(num_atm)
# # escaped_mu = np.zeros_like(momentum_method_1)
# scatter_count = np.zeros_like(momentum_method_1)

# # momentum_method_2 = np.zeros(num_atm)

# # number_transferred = np.zeros(num_atm)

# for atm_i, tau_atm in enumerate(tau_list):
#     # Run the photons and return the mu at escape
#     escaped_mu, boundary_momentum, escaped_momentum, initial_momentum, scatter_count[atm_i] = RunPhotons(tau_atm, num_photons, boundary, scatter, PDF, Range)

#     momentum_method_1[atm_i] = (initial_momentum - escaped_momentum + boundary_momentum)/initial_momentum
#     # momentum_method_2[atm_i] = collision_momentum/initial_momentum

# plt.figure(1)

# tau_list = pd.concat([old[old.columns[1:]],DF])['tau']
# momentum_method_1 = pd.concat([old[old.columns[1:]],DF])['momentum']

# plt.plot(tau_list, momentum_method_1, label='Sil')

# plt.plot(tau_list, 1-np.exp(-tau_list ), label = r'$1-e^{-\tau}$')

# DF = pd.DataFrame({'tau':tau_list, 'momentum':momentum_method_1})

L_Edd = pd.read_csv('L_Edd dataframe.csv')

# # DF.to_csv('momentum transfer 0_6 all.csv')

# # new = pd.read_csv('momentum transfer.csv')

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
# plt.plot(DF['tau']*ratio_rp_lambda_rp[0],1-np.exp(-DF['tau']*ratio_rp_lambda_rp[0]), label = r'$1-e^{-\tau_{RP}}$')
# # plt.plot(DF['tau'],1-np.exp(-DF['tau']*ratio_f_06_rp[0]), label = r'$1-e^{-\tau_{F}}$')
# # plt.plot(DF['tau'], DF['momentum'], label = 'MC result')
# plt.plot(DF['tau']*ratio_rp_lambda_rp[0], DF['momentum'], label = 'MC result')

# # plt.plot(1.086 * ratio_lambda_f_lambda_rp * DF['tau'], DF['momentum'], label = 'MC result')

# # # plt.xlabel(r"Atmospheric depth ($\tau_{RP, \, \lambda = 0.6 \, \mu m}$)")
# # # plt.xlabel(r"Atmospheric depth ($\tau_{RP}$)")
# plt.xlabel(r'$\frac{A_{\rm V}}{\rm mag}$')
# plt.ylabel("Momentum Transferred (fraction of initial)")
# plt.xscale('log')
# plt.yscale('log')
# plt.title("Momentum transferred")
# plt.legend()
# plt.grid(which = 'both', linestyle = ':')
# plt.tick_params(labelright=True, right = True, grid_color = 'black', grid_alpha = 0.3)

# image_name = 'Momentum transfer ' + str(wl_ref).replace('.','_') + ' ' + grain_type + ' time slice ' + str(time_slice).replace('.','_') + '.png'

# plt.savefig(image_name, dpi = 200)
## ---------------------------------------------------------------------------

## Plot the LEdd
## ---------------------------------------------------------------------------

# L_edd = Grain_data.Kappa_RP[Grain_data.WL.round(decimals=4) == wl_ref].to_numpy()/(100*DF['tau'] * DF['momentum'])*G*c

# A_lambda = 1.086 * ratio_lambda_f_lambda_rp * DF['tau']

# plt.plot(A_lambda, L_edd)
# plt.xscale('log')
# plt.yscale('log')

# plt.xlabel(r'$A_\lambda$ / mag')
# plt.ylabel(r'$L_{\rm Edd}$ / M$\pi \rm r^2$')

# # plt.savefig('L_Edd by extinction V-band.png', dpi = 200)

## ---------------------------------------------------------------------------

# ## Combine galaxy data
# ## ---------------------------------------------------------------------------


# Open and read the files
df1 = pd.read_csv(gal_file_1, delim_whitespace = True)
df2 = pd.read_csv(gal_file_2, delim_whitespace = True)

# Merge the files and delete the extra id column.
gal_data = pd.merge(df1,df2, left_on = 'ID', right_on = 'id')
gal_data = gal_data.drop(columns='id')

# Define new columns
# Find bolometric luminosty: L_Ha / 0.00724 (Kennicutt & Evans)
gal_data['L_Bol'] = gal_data.LHa/0.00724

# Find Sigma_old, area density of old stars.
# in solar masses/pc^2, then converted to cgs units.
gal_data['Sigma_star'] = 350*gal_data['3.6um_aperture_average'] / cm_pc**2 * g_Msun
gal_data['Sigma_g'] = gal_data.AHa/(1.086 * Grain_data.Kappa_F[Grain_data.WL.round(decimals=4) == wl_ref].to_numpy() * f_dg) * (h_g_original/h_g)**2


# ## ---------------------------------------------------------------------------


## Get Eddington ratios for galaxies
## ---------------------------------------------------------------------------

# gal_file = 'M51 MC Complete.csv'
# gal_file = 'M51.csv'

time_slice = 6.0

Range = BPASS_data.WL.to_numpy()

# gal_data = pd.read_csv(gal_file)

gal_data['Momentum'] = 0

ratio_lambda_f_lambda_rp = Grain_data.Kappa_F[Grain_data.WL.round(decimals=4) == wl_ref].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL.round(decimals = 4) == wl_ref]

# gal_data['tau'] = gal_data.AHa/(1.086 * ratio_lambda_f_lambda_rp.values[0])

gal_data['tau'] = gal_data['Sigma_g'] * Grain_data.Kappa_RP[Grain_data.WL.round(decimals=4) == wl_ref].to_numpy() * f_dg

# (ergs/s) / L_Sun
L_sun_conversion = 3.826*10**33

L_Edd_DF = pd.read_csv('L_Edd dataframe.csv')

L_over_M = L_Edd_DF[L_Edd_DF.time == time_slice].L_bol_BPASS/L_Edd_DF[L_Edd_DF.time == time_slice].Mass

# Mass of new stars, in M_sun.  
gal_data['Mass_new'] = gal_data.L_Bol/(L_over_M.to_numpy()[0] * L_sun_conversion  / g_Msun)

# Mass of gas in M_sun.
gal_data['Mass_g'] = gal_data.AHa/(1.086 * Grain_data.Kappa_F[Grain_data.WL.round(decimals=4) == wl_ref].to_numpy() * f_dg)*4*np.pi*h_g_original**2

# Mass of old stars in M_sun.
gal_data['Mass_old'] = gal_data.Sigma_star * 2/3 * np.pi * h_g**3 / h_old

gal_data['Mass_tot'] = (gal_data.Mass_g + gal_data.Mass_old + gal_data.Mass_new)

# for i, row in gal_data.iterrows():

#     tau_max = row.tau
    
#     escaped_mu, boundary_momentum, escaped_momentum, initial_momentum, scatter_count = RunPhotons(tau_max , num_photons, boundary, scatter, PDF, Range)
    
#     momentum = (initial_momentum - escaped_momentum + boundary_momentum)/initial_momentum
    
#     gal_data.loc[gal_data.ID == row.ID, 'Momentum'] = momentum
    
MC_data = pd.read_csv('detailed MC.csv')
momentum = np.interp(gal_data.tau, MC_data.tau.values, MC_data.momentum.values)
gal_data['Momentum'] = momentum

gal_data['L_Edd'] = (G * c * gal_data['Mass_g'] * gal_data.Mass_tot  / (h_g**2 * gal_data['Momentum']))

Edd_ratio = gal_data.L_Bol/gal_data.L_Edd-1
Edd_ratio[Edd_ratio < 0] = 0

gal_data['v_inf_1'] = np.sqrt(2*G*gal_data.Mass_tot/h_g)*np.sqrt(Edd_ratio)

gal_data['R_UV'] = (Grain_data.Kappa_RP[Grain_data.WL.round(decimals = 4) == wl_ref].to_numpy()[0] * f_dg * gal_data.Mass_g/(4*np.pi))**0.5
gal_data['v_inf_2'] = ((4 * gal_data.R_UV * gal_data.L_Bol)/(gal_data.Mass_g * c))**0.5




##  Plotting things!
## ---------------------------------------------------------------------------

# M_tot = gal_data.Mass_g + gal_data.Mass_new + gal_data.Mass_old

# ## --------------------------------------------
# plt.scatter(gal_data.Mass_g, gal_data.L_Edd/M_tot)
# plt.xscale('log')
# plt.yscale('log')

# plt.grid(which = 'both', linestyle = ':')
# plt.tick_params(labelright=True, right = True, grid_color = 'black', grid_alpha = 0.9)

# plt.ylabel(r'$L_{Edd}/M$ (solar luminosity / solar mass)')
# plt.xlabel(r'$M_{g}$ (Solar masses)')

# plt.savefig('L_Edd over M by M_g M51.png', dpi = 200)
# plt.close()
# # --------------------------------------------


# # --------------------------------------------
# plt.scatter(gal_data.Mass_new, gal_data.L_Edd/M_tot)
# plt.xscale('log')
# plt.yscale('log')

# plt.grid(which = 'both', linestyle = ':')
# plt.tick_params(labelright=True, right = True, grid_color = 'black', grid_alpha = 0.9)

# plt.ylabel(r'$L_{Edd}/M$ (solar luminosity / solar mass)')
# plt.xlabel(r'$M_{new}$ (Solar masses)')

# plt.savefig('L_Edd over M by M_new M51.png', dpi = 200)
# plt.close()
# # --------------------------------------------


# # --------------------------------------------
# plt.scatter(gal_data.Mass_old, gal_data.L_Edd/M_tot)
# plt.xscale('log')
# plt.yscale('log')

# plt.grid(which = 'both', linestyle = ':')
# plt.tick_params(labelright=True, right = True, grid_color = 'black', grid_alpha = 0.9)

# plt.ylabel(r'$L_{Edd}/M$ (solar luminosity / solar mass)')
# plt.xlabel(r'$M_{old}$ (Solar masses)')

# plt.savefig('L_Edd over M by M_old M51.png', dpi = 200)
# plt.close()
# # --------------------------------------------


# # --------------------------------------------
# plt.scatter(gal_data.L_Bol/(M_tot*L_sun_conversion), gal_data.L_Edd/M_tot)
# # plt.xscale('log')
# plt.yscale('log')

# plt.grid(which = 'both', linestyle = ':')
# plt.tick_params(labelright=True, right = True, grid_color = 'black', grid_alpha = 0.9)

# plt.ylabel(r'$L_{Edd}/M$')
# plt.xlabel(r'$L_{bol}/M$')

# plt.savefig('L_Edd over M by L_bol M51.png', dpi = 200)
# plt.close()
# # --------------------------------------------


# # --------------------------------------------
# plt.scatter(gal_data.AHa, gal_data.L_Edd/M_tot)
# # plt.xscale('log')
# plt.yscale('log')

# plt.grid(which = 'both', linestyle = ':')
# plt.tick_params(labelright=True, right = True, grid_color = 'black', grid_alpha = 0.9)

# plt.ylabel(r'$L_{Edd}/M$')
# plt.xlabel(r'$A_{\rm H\alpha}$')

# plt.savefig('L_Edd over M by Halpha extinction M51.png', dpi = 200)
# plt.close()
# # --------------------------------------------


# # --------------------------------------------

# plt.style.use('dark_background')
# plt.scatter(gal_data.xcenter, gal_data.ycenter, s = 1, c = gal_data.L_Bol/(L_sun_conversion*gal_data.L_Edd), norm = matplotlib.colors.DivergingNorm(1), cmap = 'coolwarm')
# plt.colorbar()
# plt.savefig('L_Bol over L_Edd for M51 cmap.png', dpi = 200)
# # --------------------------------------------

# # --------------------------------------------

# plt.hist(gal_data.L_Bol/(L_sun_conversion*gal_data.L_Edd), bins = 250)
# plt.xlabel(r'$L_{\rm bol}/L_{\rm Edd}$')
# plt.ylabel('count')
# plt.savefig('M51 L over L hist.png', dpi = 200)

## --------------------------------------------


## --------------------------------------------
# plt.hist(v_inf, bins = 200)
# plt.xlabel(r'$v_{\infty}$ km/s')
# plt.ylabel('count')
# plt.savefig('V_inf for M51.png', dpi = 200)
## --------------------------------------------

# # --------------------------------------------

# plt.style.use('dark_background')
# plt.scatter(gal_data.xcenter, gal_data.ycenter, s = 1, c = gal_data.L_Bol/gal_data.L_Edd, norm = matplotlib.colors.DivergingNorm(1), cmap = 'YlOrRd')
# plt.colorbar()
# plt.savefig('v_inf cmap for M51 cmap.png', dpi = 200)
# # --------------------------------------------


# Plot things after cgs update
## ---------------------------------------------------------------------------


# # Plot mass of gas
# # ----------------------------------------------
# plt.scatter(gal_data.ID, gal_data.Mass_g/g_Msun)
# plt.yscale('log')
# plt.ylabel(r'$M_g \, (M_\odot)$')
# plt.savefig('M51 Mass_g scatter.png', dpi = 200)
# # ----------------------------------------------

# # Plot mass of new stars
# # ----------------------------------------------
# plt.scatter(gal_data.ID, gal_data.Mass_new/g_Msun)
# plt.yscale('log')
# plt.ylabel(r'$M_{\rm new} \, (M_\odot)$')
# plt.savefig('M51 Mass_new scatter.png', dpi = 200)
# # ----------------------------------------------

# # Plot mass of old stars
# # ----------------------------------------------
# plt.scatter(gal_data.ID, gal_data.Mass_old/g_Msun)
# plt.yscale('log')
# plt.ylabel(r'$M_{\rm old} \, (M_\odot)$')
# plt.savefig('M51 Mass_old scatter.png', dpi = 200)
# # ----------------------------------------------

# Plot bolometric luminosity
# # ------------------------------------------------
# plt.scatter(gal_data.ID, gal_data.L_Bol/L_sun_conversion)
# plt.yscale('log')
# plt.ylabel(r'$L_{\rm Bol} \, (L_\odot)$')
# plt.savefig('M51 L_bol scatter.png', dpi = 200)
# ------------------------------------------------

# # Plot Eddington luminosity
# # ------------------------------------------------
# plt.scatter(gal_data.ID, gal_data.L_Edd/L_sun_conversion)
# plt.yscale('log')
# plt.ylabel(r'$L_{\rm Edd} \, (L_\odot)$')
# plt.savefig('M51 L_Edd scatter.png', dpi = 200)
# ------------------------------------------------

# # Histogram of L_Edd by gas column height
# # ------------------------------------------------

# def ChangeOldColumn(df, old, new):

#     df.L_Edd = df.L_Edd/(df.Mass_g + df.Mass_old + df.Mass_new)

#     df.Mass_old = df.Mass_old*old/(new*cm_pc)

#     df.L_Edd = df.L_Edd*(df.Mass_g + df.Mass_old + df.Mass_new)
    
#     return df

# df1 = pd.read_csv('5pc run.csv')
# df2 = pd.read_csv('10pc run.csv')
# df3 = pd.read_csv('15pc run.csv')

# df4 = ChangeOldColumn(df1, h_old, 500)
# df5 = ChangeOldColumn(df2, h_old, 500)
# df6 = ChangeOldColumn(df3, h_old, 500)

# df7 = ChangeOldColumn(df1, h_old, 1000)
# df8 = ChangeOldColumn(df2, h_old, 1000)
# df9 = ChangeOldColumn(df3, h_old, 1000)

# fig, [ax1, ax2, ax3] = plt.subplots(1,3, sharey=True)
# ax1.hist(df1.L_Bol/df1.L_Edd, bins=np.logspace(np.log10(np.min(df1.L_Bol/df1.L_Edd)),np.log10(np.max(df1.L_Bol/df1.L_Edd)), 200))
# ax2.hist(df2.L_Bol/df2.L_Edd, bins=np.logspace(np.log10(np.min(df2.L_Bol/df2.L_Edd)),np.log10(np.max(df2.L_Bol/df2.L_Edd)), 200))
# ax3.hist(df3.L_Bol/df3.L_Edd, bins=np.logspace(np.log10(np.min(df3.L_Bol/df3.L_Edd)),np.log10(np.max(df3.L_Bol/df3.L_Edd)), 200))
# ax1.set_xscale('log')
# ax2.set_xscale('log')
# ax3.set_xscale('log')
# ax1.set_title(r'$r_g$ = 5 pc')
# ax2.set_title(r'$r_g$ = 10 pc')
# ax3.set_title(r'$r_g$ = 15 pc')
# fig.set_figheight(5)
# fig.set_figwidth(10)
# fig.text(0.5, 0.04, r'$L_{\rm bol}/L_{\rm Edd}$', ha='center')
# plt.savefig('L_bol over L_Edd by column height h_old 500.png', dpi = 200)
# # ------------------------------------------------

## Plot L_bol/L_Edd as a function of everything

# fig, axs = plt.subplots(3,3, sharey = True)
# axs[0,0].scatter(df1.Mass_g/g_Msun, df1.L_Bol/df1.L_Edd)
# axs[0,1].scatter(df1.Mass_old/g_Msun, df1.L_Bol/df1.L_Edd)
# axs[0,2].scatter(df1.Mass_new/g_Msun, df1.L_Bol/df1.L_Edd)
# axs[1,0].scatter(df1.AHa, df1.L_Bol/df1.L_Edd)
# axs[1,1].scatter(df1.Sigma_star, df1.L_Bol/df1.L_Edd)
# axs[1,2].scatter(df1.Momentum, df1.L_Bol/df1.L_Edd)
# axs[2,0].scatter(df1.tau, df1.L_Bol/df1.L_Edd)
# axs[2,1].scatter(df1['3.6um_aperture_average'], df1.L_Bol/df1.L_Edd)
# axs[2,2].scatter(df1.L_Edd, df1.L_Bol/df1.L_Edd)
# axs[0,0].set_xscale('log')
# axs[0,1].set_xscale('log')
# axs[0,2].set_xscale('log')
# axs[1,0].set_xscale('log')
# axs[1,1].set_xscale('log')
# axs[1,2].set_xscale('log')
# axs[2,0].set_xscale('log')
# axs[2,1].set_xscale('log')
# axs[2,2].set_xscale('log')
# fig.set_figheight(10)
# fig.set_figwidth(10)


# # Histogram of L_Edd by gas column height
# # ------------------------------------------------
# df1 = pd.read_csv('5pc run.csv')

# df3 = pd.read_csv('15pc run.csv')

# df2 = pd.read_csv('10pc run.csv')

# fig, [ax1, ax2, ax3] = plt.subplots(1,3)
# ax1.scatter(df1.ID, df1.Mass_old/g_Msun)
# ax2.scatter(df2.ID, df2.Mass_old/g_Msun)
# ax3.scatter(df3.ID, df3.Mass_old/g_Msun)
# ax1.set_yscale('log')
# ax2.set_yscale('log')
# ax3.set_yscale('log')
# ax1.set_title(r'$r_g$ = 5 pc')
# ax2.set_title(r'$r_g$ = 10 pc')
# ax3.set_title(r'$r_g$ = 15 pc')
# fig.set_figheight(5)
# fig.set_figwidth(10)
# plt.savefig('M Old by column height h_old 500.png', dpi = 200)
# # ------------------------------------------------

## Plot Eddington ratio cmaps
## -------------------------------------------------
##  Need to fix ax3 to be less squished.

# Edd_ratio = df1.L_Bol/df1.L_Edd

# plt.style.use('dark_background')
# fig, [ax1, ax2, ax3] = plt.subplots(1,3)
# im1 = ax1.scatter(df1.xcenter, df1.ycenter, s = 1, c = df1.L_Bol/df1.L_Edd, norm = matplotlib.colors.DivergingNorm(1), vmin = 0, vmax = Edd_ratio.max(), cmap = 'YlOrRd')
# im2 = ax2.scatter(df2.xcenter, df2.ycenter, s = 1, c = df2.L_Bol/df2.L_Edd, norm = matplotlib.colors.DivergingNorm(1), vmin = 0, vmax = Edd_ratio.max(), cmap = 'YlOrRd')
# im3 = ax3.scatter(df3.xcenter, df3.ycenter, s = 1, c = df3.L_Bol/df3.L_Edd, norm = matplotlib.colors.DivergingNorm(1), vmin = 0, vmax = Edd_ratio.max(), cmap = 'YlOrRd')
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax2.set_xticks([])
# ax2.set_yticks([])
# ax3.set_xticks([])
# ax3.set_yticks([])
# fig.colorbar(im1)
# fig.set_figheight(5)
# fig.set_figwidth(15)
# plt.tight_layout()
# plt.savefig('Eddington ratio cmap for M51 cmap.png', dpi = 200)
## -------------------------------------------------

# # Eddington ratio by old mass
# # ------------------------------------------------
# # df1 = pd.read_csv('5pc run.csv')

# # df3 = pd.read_csv('15pc run.csv')

# # df2 = pd.read_csv('10pc run.csv')

# fig, [ax1, ax2, ax3] = plt.subplots(1,3)
# ax1.scatter(df1.Mass_old, df1.L_Bol/df1.L_Edd)
# ax2.scatter(df2.Mass_old, df2.L_Bol/df2.L_Edd)
# ax3.scatter(df3.Mass_old, df3.L_Bol/df3.L_Edd)
# fig.suptitle(r'Eddington ratio vs $M_{\rm old, \star}$', y = 0.94)
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
# # ax3.set_yscale('log')
# ax1.set_xscale('log')
# ax2.set_xscale('log')
# ax3.set_xscale('log')
# # ax1.set_title(r'$r_g$ = 5 pc')
# # ax2.set_title(r'$r_g$ = 10 pc')
# # ax3.set_title(r'$r_g$ = 15 pc')
# fig.set_figheight(5)
# fig.set_figwidth(10)
# plt.savefig('Eddington ratio by M Old per column height h_old 500.png', dpi = 200)
# # ------------------------------------------------

# # Eddington ratio by new mass
# # ------------------------------------------------
# # df1 = pd.read_csv('5pc run.csv')

# # df3 = pd.read_csv('15pc run.csv')

# # df2 = pd.read_csv('10pc run.csv')

# fig, [ax1, ax2, ax3] = plt.subplots(1,3)
# ax1.scatter(df1.Mass_new, df1.L_Bol/df1.L_Edd)
# ax2.scatter(df2.Mass_new, df2.L_Bol/df2.L_Edd)
# ax3.scatter(df3.Mass_new, df3.L_Bol/df3.L_Edd)
# fig.suptitle(r'Eddington ratio vs $M_{\rm new, \star}$', y = 0.94)
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
# # ax3.set_yscale('log')
# ax1.set_xscale('log')
# ax2.set_xscale('log')
# ax3.set_xscale('log')
# # ax1.set_title(r'$r_g$ = 5 pc')
# # ax2.set_title(r'$r_g$ = 10 pc')
# # ax3.set_title(r'$r_g$ = 15 pc')
# fig.set_figheight(5)
# fig.set_figwidth(10)
# plt.savefig('Eddington ratio by M New per column height h_old 500.png', dpi = 200)
# # ------------------------------------------------


# # Eddington ratio by gas mass
# # ------------------------------------------------
# # df1 = pd.read_csv('5pc run.csv')

# # df3 = pd.read_csv('15pc run.csv')

# # df2 = pd.read_csv('10pc run.csv')

# fig, [ax1, ax2, ax3] = plt.subplots(1,3)
# ax1.scatter(df1.Mass_g, df1.L_Bol/df1.L_Edd)
# ax2.scatter(df2.Mass_g, df2.L_Bol/df2.L_Edd)
# ax3.scatter(df3.Mass_g, df3.L_Bol/df3.L_Edd)
# fig.suptitle(r'Eddington ratio vs $M_{\rm gas}$', y = 0.94)
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
# # ax3.set_yscale('log')
# ax1.set_xscale('log')
# ax2.set_xscale('log')
# ax3.set_xscale('log')
# # ax1.set_title(r'$r_g$ = 5 pc')
# # ax2.set_title(r'$r_g$ = 10 pc')
# # ax3.set_title(r'$r_g$ = 15 pc')
# fig.set_figheight(5)
# fig.set_figwidth(10)
# plt.savefig('Eddington ratio by M Gas per column height h_old 500.png', dpi = 200)
# # ------------------------------------------------

# # # Eddington ratio by AHa
# # # ------------------------------------------------
# # df1 = pd.read_csv('5pc run.csv')

# # df3 = pd.read_csv('15pc run.csv')

# # df2 = pd.read_csv('10pc run.csv')

# fig, [ax1, ax2, ax3] = plt.subplots(1,3)
# ax1.scatter(df1.AHa, df1.L_Bol/df1.L_Edd)
# ax2.scatter(df2.AHa, df2.L_Bol/df2.L_Edd)
# ax3.scatter(df3.AHa, df3.L_Bol/df3.L_Edd)
# fig.suptitle(r'Eddington ratio vs $A_{\rm H\alpha}$', y = 0.94)
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
# # ax3.set_yscale('log')
# # ax1.set_xscale('log')
# # ax2.set_xscale('log')
# # ax3.set_xscale('log')
# # ax1.set_title(r'$r_g$ = 5 pc')
# # ax2.set_title(r'$r_g$ = 10 pc')
# # ax3.set_title(r'$r_g$ = 15 pc')
# fig.set_figheight(5)
# fig.set_figwidth(10)
# plt.savefig('Eddington ratio by AHa per column height h_old 500.png', dpi = 200)
# # # ------------------------------------------------

# # # Eddington ratio by AHa
# # # ------------------------------------------------
# # df1 = pd.read_csv('5pc run.csv')

# # df3 = pd.read_csv('15pc run.csv')

# # df2 = pd.read_csv('10pc run.csv')

# fig, [ax1, ax2, ax3] = plt.subplots(1,3)
# ax1.scatter(df1.AHa, df1.L_Bol/df1.L_Edd)
# ax2.scatter(df2.AHa, df2.L_Bol/df2.L_Edd)
# ax3.scatter(df3.AHa, df3.L_Bol/df3.L_Edd)
# fig.suptitle(r'Eddington ratio vs $A_{\rm H\alpha}$', y = 0.94)
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
# # ax3.set_yscale('log')
# # ax1.set_xscale('log')
# # ax2.set_xscale('log')
# # ax3.set_xscale('log')
# # ax1.set_title(r'$r_g$ = 5 pc')
# # ax2.set_title(r'$r_g$ = 10 pc')
# # ax3.set_title(r'$r_g$ = 15 pc')
# fig.set_figheight(5)
# fig.set_figwidth(10)
# plt.savefig('Eddington ratio by AHa per column height h_old 500.png', dpi = 200)
# # # ------------------------------------------------

# # # Eddington ratio by AHa
# # # ------------------------------------------------
# # df1 = pd.read_csv('5pc run.csv')

# # df3 = pd.read_csv('15pc run.csv')

# # df2 = pd.read_csv('10pc run.csv')

# fig, [ax1, ax2, ax3] = plt.subplots(1,3)
# ax1.scatter(df1['3.6um_aperture_average'], df1.L_Bol/df1.L_Edd)
# ax2.scatter(df2['3.6um_aperture_average'], df2.L_Bol/df2.L_Edd)
# ax3.scatter(df3['3.6um_aperture_average'], df3.L_Bol/df3.L_Edd)
# fig.suptitle(r'Eddington ratio vs $I_{3.6 \mu \rm m}$', y = 0.94)
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
# # ax3.set_yscale('log')
# ax1.set_xscale('log')
# ax2.set_xscale('log')
# ax3.set_xscale('log')
# # ax1.set_title(r'$r_g$ = 5 pc')
# # ax2.set_title(r'$r_g$ = 10 pc')
# # ax3.set_title(r'$r_g$ = 15 pc')
# fig.set_figheight(5)
# fig.set_figwidth(10)
# plt.savefig('Eddington ratio by I36 per column height h_old 500.png', dpi = 200)
# # # ------------------------------------------------

# # # Eddington ratio by L_Bol
# # # ------------------------------------------------
# # df1 = pd.read_csv('5pc run.csv')

# # df3 = pd.read_csv('15pc run.csv')

# # df2 = pd.read_csv('10pc run.csv')

# fig, [ax1, ax2, ax3] = plt.subplots(1,3)
# ax1.scatter(df1.L_Bol, df1.L_Bol/df1.L_Edd)
# ax2.scatter(df2.L_Bol, df2.L_Bol/df2.L_Edd)
# ax3.scatter(df3.L_Bol, df3.L_Bol/df3.L_Edd)
# fig.suptitle(r'Eddington ratio vs $L_{\rm bol}$', y = 0.94)
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
# # ax3.set_yscale('log')
# ax1.set_xscale('log')
# ax2.set_xscale('log')
# ax3.set_xscale('log')
# # ax1.set_title(r'$r_g$ = 5 pc')
# # ax2.set_title(r'$r_g$ = 10 pc')
# # ax3.set_title(r'$r_g$ = 15 pc')
# fig.set_figheight(5)
# fig.set_figwidth(10)
# plt.savefig('Eddington ratio by L_Bol per column height h_old 500.png', dpi = 200)
# # # ------------------------------------------------

# # # Eddington ratio by AHa
# # # ------------------------------------------------
# # df1 = pd.read_csv('5pc run.csv')

# # df3 = pd.read_csv('15pc run.csv')

# # df2 = pd.read_csv('10pc run.csv')

# fig, [ax1, ax2, ax3] = plt.subplots(1,3)
# ax1.scatter(df1.tau, df1.L_Bol/df1.L_Edd)
# ax2.scatter(df2.tau, df2.L_Bol/df2.L_Edd)
# ax3.scatter(df3.tau, df3.L_Bol/df3.L_Edd)
# fig.suptitle(r'Eddington ratio vs $\langle \tau_{\rm RP} \rangle$', y = 0.94)
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
# # ax3.set_yscale('log')
# # ax1.set_xscale('log')
# # ax2.set_xscale('log')
# # ax3.set_xscale('log')
# # ax1.set_title(r'$r_g$ = 5 pc')
# # ax2.set_title(r'$r_g$ = 10 pc')
# # ax3.set_title(r'$r_g$ = 15 pc')
# fig.set_figheight(5)
# fig.set_figwidth(10)
# plt.savefig('Eddington ratio by tau per column height h_old 500.png', dpi = 200)
# # # ------------------------------------------------

# # # Eddington ratio by Sigma gas
# # # ------------------------------------------------
# # df1 = pd.read_csv('5pc run.csv')

# # df3 = pd.read_csv('15pc run.csv')

# # df2 = pd.read_csv('10pc run.csv')

# fig, [ax1, ax2, ax3] = plt.subplots(1,3)
# ax1.scatter(df1.Sigma_g, df1.L_Bol/df1.L_Edd)
# ax2.scatter(df2.Sigma_g, df2.L_Bol/df2.L_Edd)
# ax3.scatter(df3.Sigma_g, df3.L_Bol/df3.L_Edd)
# fig.suptitle(r'Eddington ratio vs $\Sigma_{\rm g}$', y = 0.94)
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
# # ax3.set_yscale('log')
# # ax1.set_xscale('log')
# # ax2.set_xscale('log')
# # ax3.set_xscale('log')
# ax1.set_xlim(-0.001,df1.Sigma_g.max())
# ax2.set_xlim(-0.001,df1.Sigma_g.max())
# ax3.set_xlim(-0.001,df1.Sigma_g.max())
# # ax1.set_title(r'$r_g$ = 5 pc')
# # ax2.set_title(r'$r_g$ = 10 pc')
# # ax3.set_title(r'$r_g$ = 15 pc')
# fig.set_figheight(5)
# fig.set_figwidth(10)
# plt.savefig('Eddington ratio by Sigma g per column height h_old 500.png', dpi = 200)
# # # ------------------------------------------------

# # # Eddington ratio by Sigma old
# # # ------------------------------------------------
# # df1 = pd.read_csv('5pc run.csv')

# # df3 = pd.read_csv('15pc run.csv')

# # df2 = pd.read_csv('10pc run.csv')

# fig, [ax1, ax2, ax3] = plt.subplots(1,3)
# ax1.scatter(df1.Sigma_star, df1.L_Bol/df1.L_Edd)
# ax2.scatter(df2.Sigma_star, df2.L_Bol/df2.L_Edd)
# ax3.scatter(df3.Sigma_star, df3.L_Bol/df3.L_Edd)
# fig.suptitle(r'Eddington ratio vs $\Sigma_{\rm old, \star}$', y = 0.94)
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
# # ax3.set_yscale('log')
# # ax1.set_xscale('log')
# # ax2.set_xscale('log')
# # ax3.set_xscale('log')
# # ax1.set_title(r'$r_g$ = 5 pc')
# # ax2.set_title(r'$r_g$ = 10 pc')
# # ax3.set_title(r'$r_g$ = 15 pc')
# fig.set_figheight(5)
# fig.set_figwidth(10)
# plt.savefig('Eddington ratio by Sigma old per column height h_old 500.png', dpi = 200)
# # # ------------------------------------------------

## ---------------------------------------------------------------------------

# ## Plot Eddington ratio by M_g all in one plot.
# ## ----------------------------------------------------------
# plt.scatter(df1.Mass_g,df1.L_Bol/df1.L_Edd, alpha = 0.5, s = 10, label = '5 pc')
# plt.scatter(df2.Mass_g,df2.L_Bol/df2.L_Edd, alpha = 0.5, s = 10, label = '10 pc')
# plt.scatter(df3.Mass_g,df3.L_Bol/df3.L_Edd, alpha = 0.5, s = 10, label = '15 pc')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.xlabel(r'$M_g$ (g)')
# plt.ylabel('Eddington ratio')
# plt.savefig('Eddington ratio by M_g h_old 500.png', dpi = 200)
# plt.close()
# ## ----------------------------------------------------------

# ## Plot Eddington ratio by M_old all in one plot.
# ## ----------------------------------------------------------
# plt.scatter(df1.Mass_old,df1.L_Bol/df1.L_Edd, alpha = 0.5, s = 10, label = '5 pc')
# plt.scatter(df2.Mass_old,df2.L_Bol/df2.L_Edd, alpha = 0.5, s = 10, label = '10 pc')
# plt.scatter(df3.Mass_old,df3.L_Bol/df3.L_Edd, alpha = 0.5, s = 10, label = '15 pc')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.xlabel(r'$M_{\rm old}$ (g)')
# plt.ylabel('Eddington ratio')
# plt.savefig('Eddington ratio by M_old h_old 500.png', dpi = 200)
# plt.close()
# ## ----------------------------------------------------------

# ## Plot Eddington ratio by M_new all in one plot.
# ## ----------------------------------------------------------
# plt.scatter(df1.Mass_new,df1.L_Bol/df1.L_Edd, alpha = 0.5, s = 10, label = '5 pc')
# plt.scatter(df2.Mass_new,df2.L_Bol/df2.L_Edd, alpha = 0.5, s = 10, label = '10 pc')
# plt.scatter(df3.Mass_new,df3.L_Bol/df3.L_Edd, alpha = 0.5, s = 10, label = '15 pc')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.xlabel(r'$M_{\rm new}$ (g)')
# plt.ylabel('Eddington ratio')
# plt.savefig('Eddington ratio by M_new h_old 500.png', dpi = 200)
# plt.close()
# ## ----------------------------------------------------------

# ## Plot Eddington ratio by Sigma_old all in one plot.
# ## ----------------------------------------------------------
# plt.scatter(df1.Sigma_star,df1.L_Bol/df1.L_Edd, alpha = 0.5, s = 10, label = '5 pc')
# plt.scatter(df2.Sigma_star,df2.L_Bol/df2.L_Edd, alpha = 0.5, s = 10, label = '10 pc')
# plt.scatter(df3.Sigma_star,df3.L_Bol/df3.L_Edd, alpha = 0.5, s = 10, label = '15 pc')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.xlabel(r'$\Sigma_{\rm old}$ (g/$\rm cm^2$)')
# plt.ylabel('Eddington ratio')
# plt.savefig('Eddington ratio by Sigma_old h_old 500.png', dpi = 200)
# plt.close()
# ## ----------------------------------------------------------

# ## Plot Eddington ratio by Sigma_g all in one plot.
# ## ----------------------------------------------------------
# plt.scatter(df1.Sigma_g,df1.L_Bol/df1.L_Edd, alpha = 0.5, s = 10, label = '5 pc')
# plt.scatter(df2.Sigma_g,df2.L_Bol/df2.L_Edd, alpha = 0.5, s = 10, label = '10 pc')
# plt.scatter(df3.Sigma_g,df3.L_Bol/df3.L_Edd, alpha = 0.5, s = 10, label = '15 pc')
# # plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.xlim(-0.0002,0.03)
# plt.xlabel(r'$\Sigma_{\rm g}$ (g/$\rm cm^2$)')
# plt.ylabel('Eddington ratio')
# plt.savefig('Eddington ratio by Sigma_g h_old 500.png', dpi = 200)
# plt.close()
# ## ----------------------------------------------------------

# ## Plot Eddington ratio by AHa all in one plot.
# ## ----------------------------------------------------------
# plt.scatter(df1.AHa,df1.L_Bol/df1.L_Edd, alpha = 0.5, s = 10, label = '5 pc')
# plt.scatter(df2.AHa,df2.L_Bol/df2.L_Edd, alpha = 0.5, s = 10, label = '10 pc')
# plt.scatter(df3.AHa,df3.L_Bol/df3.L_Edd, alpha = 0.5, s = 10, label = '15 pc')
# # plt.xscale('log')
# plt.yscale('log')
# plt.xlim(-0.0002)
# plt.legend()
# plt.xlabel(r'$A_{\rm H\alpha}$ (g)')
# plt.ylabel('Eddington ratio')
# plt.savefig('Eddington ratio by AHa h_old 500.png', dpi = 200)
# plt.close()
# ## ----------------------------------------------------------

# ## Plot Eddington ratio by L_Bol all in one plot.
# ## ----------------------------------------------------------
# plt.scatter(df1.L_Bol,df1.L_Bol/df1.L_Edd, alpha = 0.5, s = 10, label = '5 pc')
# plt.scatter(df2.L_Bol,df2.L_Bol/df2.L_Edd, alpha = 0.5, s = 10, label = '10 pc')
# plt.scatter(df3.L_Bol,df3.L_Bol/df3.L_Edd, alpha = 0.5, s = 10, label = '15 pc')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.xlabel(r'$L_{\rm bol}$ (ergs/s)')
# plt.ylabel('Eddington ratio')
# plt.savefig('Eddington ratio by L_bol h_old 500.png', dpi = 200)
# plt.close()
# ## ----------------------------------------------------------

# ## Plot Eddington ratio by I_3.6um all in one plot.
# ## ----------------------------------------------------------
# plt.scatter(df1['3.6um_aperture_average'],df1.L_Bol/df1.L_Edd, alpha = 0.5, s = 10, label = '5 pc')
# plt.scatter(df2['3.6um_aperture_average'],df2.L_Bol/df2.L_Edd, alpha = 0.5, s = 10, label = '10 pc')
# plt.scatter(df3['3.6um_aperture_average'],df3.L_Bol/df3.L_Edd, alpha = 0.5, s = 10, label = '15 pc')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.2)
# plt.legend()
# plt.xlabel(r'$I_{3.6 \mu \rm m}$')
# plt.ylabel('Eddington ratio')
# plt.savefig('Eddington ratio by I36um h_old 500.png', dpi = 200)
# plt.close()
# ## ----------------------------------------------------------

# ## Plot Eddington ratio by tau all in one plot.
# ## ----------------------------------------------------------
# plt.scatter(df1.tau,df1.L_Bol/df1.L_Edd, alpha = 0.5, s = 10, label = '5 pc')
# plt.scatter(df2.tau,df2.L_Bol/df2.L_Edd, alpha = 0.5, s = 10, label = '10 pc')
# plt.scatter(df3.tau,df3.L_Bol/df3.L_Edd, alpha = 0.5, s = 10, label = '15 pc')
# # plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.xlim(-0.0002)
# plt.xlabel(r'$\langle \tau_{\rm RP} \rangle$')
# plt.ylabel('Eddington ratio')
# plt.savefig('Eddington ratio by tau h_old 500.png', dpi = 200)
# plt.close()
# ## ----------------------------------------------------------

# ## Plot V_inf by M_g all in one plot.
# ## ----------------------------------------------------------
# plt.scatter(df1.Mass_g[df1.v_inf > 0], df1.v_inf[df1.v_inf > 0], alpha = 0.5, s = 15 ,label = '5 pc', marker = "o")
# # plt.scatter(df2.Mass_g[df1.v_inf > 0], df2.v_inf[df1.v_inf > 0], alpha = 0.5, s = 10, label = '10 pc')
# plt.scatter(df3.Mass_g[df1.v_inf > 0], df3.v_inf[df1.v_inf > 0], alpha = 0.5, label = '15 pc', marker = "x")
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# # plt.ylim(0)
# plt.xlabel(r'$M_g$ (g)')
# plt.ylabel(r'$v_{\rm inf}$ (cm/s)')
# plt.savefig('V_inf by Mass_g h_old 500.png', dpi = 200)
# plt.close()
# ## ----------------------------------------------------------

## ---------------------------------------------------------------------------


# heatmap = sns.heatmap(subset_df.corr(), mask=mask, cmap="mako")
# fig = heatmap.get_figure()
# fig.savefig('M51 h_old 100 corr heatmap.png', dpi = 200)

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


##  Plot masses and other galactic properties
## ----------------------------------------------------------------

# M51 = pd.read_csv('M51 MC Complete.csv')

# plt.scatter(M51.ID,M51.Mass_new)
# plt.yscale('log')

# plt.grid(which = 'both', linestyle = ':')
# plt.tick_params(labelright=True, right = True, grid_color = 'black', grid_alpha = 0.9)

# plt.ylabel(r'$M_{\rm new} (M_\odot)$')
# plt.xlabel('ID')

# plt.savefig('M_new by ID M51.png', dpi = 200)


# plt.scatter(M51.ID,M51.Mass_old)
# plt.yscale('log')

# plt.grid(which = 'both', linestyle = ':')
# plt.tick_params(labelright=True, right = True, grid_color = 'black', grid_alpha = 0.9)

# plt.ylabel(r'$M_{\rm old} (M_\odot)$')
# plt.xlabel('ID')

# plt.savefig('M_old by ID M51.png', dpi = 200)

# plt.scatter(M51.ID,M51.Mass_g)
# plt.yscale('log')

# plt.grid(which = 'both', linestyle = ':')
# plt.tick_params(labelright=True, right = True, grid_color = 'black', grid_alpha = 0.9)

# plt.ylabel(r'$M_{\rm g} (M_\odot)$')
# plt.xlabel('ID')

# plt.savefig('M_gas by ID M51.png', dpi = 200)


# plt.scatter(M51.ID,M51.Momentum)
# # plt.yscale('log')

# plt.grid(which = 'both', linestyle = ':')
# plt.tick_params(labelright=True, right = True, grid_color = 'black', grid_alpha = 0.9)

# plt.ylabel('MC result')
# plt.xlabel('ID')

# plt.savefig('Momentum by ID M51.png', dpi = 200)

# plt.scatter(M51.ID,M51.L_Edd/12990)
# plt.yscale('log')

# plt.grid(which = 'both', linestyle = ':')
# plt.tick_params(labelright=True, right = True, grid_color = 'black', grid_alpha = 0.9)

# plt.ylabel(r'L $(L_\odot)$')
# plt.xlabel('ID')

# plt.savefig('L_Edd by ID M51.png', dpi = 200)

## ----------------------------------------------------------------

## Plot g by wl
## ----------------------------------------------------------------
# plt.plot(Grain_data.WL, Grain_data.g)
# plt.xscale('log')
# plt.xlabel('Wavelength (microns)')
# plt.ylabel('g')
# plt.savefig('g by wl.png', dpi = 200)
## ----------------------------------------------------------------


# ## Test approximation
# ## ----------------------------------------------------------------

# K_RP = 500

# tau_kindof = gal_data.Sigma_g * K_RP

# L_Edd = 4*np.pi*G*c*gal_data.Mass_g * (gal_data.Mass_g + gal_data.Mass_old + gal_data.Mass_new) / (h_g**2 * (1-np.exp(-tau_kindof)))

# plt.scatter(gal_data.ID, (L_Edd - gal_data.L_Edd)/L_Edd)


# plt.scatter(gal_data.ID,gal_data.L_Bol/gal_data.L_Edd)
# plt.xlabel('ID')
# plt.ylabel(r'$L_{\rm bol}/L_{\rm Edd}$')
# # plt.savefig('Eddington rations with averaged kappa.png', dpi = 200)

# ## ----------------------------------------------------------------

##  Big plot time
## ---------------------------------------------------------------------------

# cols = ['AHa', 'L_Bol', 'L_Edd', 'Sigma_star', 'Sigma_g', 'tau', 'Mass_new', 'Mass_g', 'Mass_old']

# fig, ax = plt.subplots(3,3, sharey = True)

# for i in range(3):
#     for j in range(3):
#         ax[i,j].scatter(df1[cols[3*i+j]], df1.L_Bol/df1.L_Edd)
#         ax[i,j].set_yscale('log')
#         ax[i,j].set_xscale('log')
##----------------------------------------------------------------------------

ratio = gal_data.L_Bol[gal_data.L_Bol > gal_data.L_Edd].sum()/gal_data.L_Bol.sum()
print(str(h_g/cm_pc) + ' pc, Super Eddington luminosity ratio: ' + str(ratio))


## evolve region velocity
## ---------------------------------------------------------------------------

def dvdr(r,v,G,M_g,M_new,M_old_i,momentum_i,L_Bol,c,r_i, tau_i):
    
    tau_new = (r_i/r)**2*tau_i
    
    momentum_new = np.interp(tau_new, MC_data.tau.values, MC_data.momentum.values)
    
    return (-G*(M_g + M_new + M_old_i * (r / r_i)**3)/r**2 + momentum_new*L_Bol/(c*M_g))/v


# gal_data = pd.read_csv('tau test.csv')

region_ID = 1

mask = gal_data.ID == region_ID

r_span = [h_g,50*h_g]
v_0 = [100]
pnts = np.linspace(r_span[0],r_span[1],100000)

solved = inte.solve_ivp(dvdr, r_span, v_0, method='Radau', args = [G,gal_data.Mass_g[mask].to_numpy()[0],gal_data.Mass_new[mask].to_numpy()[0],gal_data.Mass_old[mask].to_numpy()[0],gal_data.Momentum[mask].to_numpy()[0],gal_data.L_Bol[mask].to_numpy()[0],c,h_g,gal_data.tau[mask].to_numpy()[0]])


Gamma_UV = gal_data.L_Bol/(4*np.pi*G*c*gal_data.Mass_tot/L_Edd.kappa_av_RP_Gra[L_Edd.time == time_slice].to_numpy())
Gamma_SS =  gal_data.L_Bol/(G*c*gal_data.Mass_g * gal_data.Mass_tot/h_g**2)

gal_data['v_inf_3'] = np.sqrt(v_0[0]**2 + 2*G*
                              (gal_data.Mass_tot)/
                              h_g * (2*Gamma_SS*gal_data.R_UV/h_g*(1-h_g/(2*gal_data.R_UV)) - 1))

plt.scatter(solved.t/cm_pc,solved.y[0]/10**5, label = 'ODE solution', color = 'blue', marker = 'x')
plt.plot([solved.t[0]/cm_pc,solved.t[-1]/cm_pc], (gal_data.v_inf_1[mask].to_list()[0]/10**5,gal_data.v_inf_1[mask].to_list()[0]/10**5), label = r'$\sqrt{v_{\rm esc}}\sqrt{L_{\rm bol}/L_{\rm Edd}}$', color ='orange')
plt.plot([solved.t[0]/cm_pc,solved.t[-1]/cm_pc], (gal_data.v_inf_2[mask].to_list()[0]/10**5,gal_data.v_inf_2[mask].to_list()[0]/10**5), label = r'$\sqrt{(4R_{\rm UV}L)/(M_{\rm g}c)}$', color ='red')
plt.plot([solved.t[0]/cm_pc,solved.t[-1]/cm_pc], (gal_data.v_inf_3[mask].to_list()[0]/10**5,gal_data.v_inf_3[mask].to_list()[0]/10**5), label = 'Analytic solution', color ='green')
# plt.yscale('log')
# plt.xscale('log')
plt.xlabel(r'$h_{\rm g}$ (pc)')
plt.ylabel('v (km/s)')
plt.legend()

title = 'velocity data for region ' + str(region_ID)
plt.title(title)

# plt.savefig('region ' + str(region_ID) + ' ' + str(int(h_g/cm_pc)) + ' pc velocity non log.png', dpi = 200)

bins = []

for i in range(0,len(solved.t)-1):
    bins += [solved.t[i+1]-solved.t[i]]


times = []
for i, distance in enumerate(bins):
    times += [distance/((solved.y[0][i]+solved.y[0][i+1])/2)]
    
total_time = sum(times)/60/60/24/365