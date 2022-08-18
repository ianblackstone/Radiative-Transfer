# Imports
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from hoki import load
from scipy import integrate as inte
from scipy.interpolate import interpn
# from scipy.optimize import fsolve
# from mpl_toolkits.axes_grid1 import host_subplot
# import mpl_toolkits.axisartist as AA
import sys

from read_BPASS import GetGrainData, GetKappas, GetLEddDataFrame
from constants import cm_pc, g_Msun, L_sun_conversion, h, c, G, sigma_SB, p_mass


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

# Declare the number of bins, photons, and atmospheres.
num_photons = 100000
num_bins = 2000
num_atm = 100
photon_reduction = 10000

# "mix", "Sil", "Gra", or "SiC"
grain_type = 'mix'
grain_min = 0.001
grain_max = 1

# Set a tau cutoff. Below this the Monte Carlo results will not be used.
tau_cutoff = 0

# Define the mixture of [Sil, Gra, SiC].  These should add to 1.
grain_mix = [0.5,0.45,0.05]

min_tau = 10**-3
max_tau = 100

## Wavelength bounds. Do no change right now!!!
#####################
wl_min = 0.001
wl_max = 10
#####################

# Select the BPASS files to use.  Uncomment the file to use
BPASS_file = 'spectra-bin-imf135_300.z020.dat'
# BPASS_file = 'spectra-bin-imf135_100.z020.dat'
# BPASS_file = 'spectra-bin-imf100_300.z020.dat'
# BPASS_file = 'spectra-bin-imf100_300.z010.dat'
# BPASS_file = 'spectra-bin-imf100_300.z001.dat'
# BPASS_file = 'spectra-bin-imf_chab300.z020.dat'
ionizing_file = BPASS_file.replace('spectra','ionizing')
yields_file = BPASS_file.replace('spectra','yields')

MC_data = pd.read_csv('detailed MC {} {}csv'.format(grain_type, BPASS_file.replace('.z',' z').replace('dat','')))

tau_list = np.logspace(-3, 2, num_atm, base=10)
# tau_list = MC_data.tau.to_numpy()
tau_list_calc = tau_list[(tau_list >= min_tau) & (tau_list <= max_tau)]
num_atm = len(tau_list_calc)

# Flag for the boundary.  it can be 'absorb', 'reemit', or 'reflect'.
boundary = 'reflect'

# Flag for the scattering type.  'hg', 'iso', 'draine'
scatter = 'hg'

# Boolean flags for randomizing mu at the start and turning off scattering.
rand_mu = False
abs_only = False

# monochromatic light, set to 0 to draw from an SED.
monochromatic = 0

# Height of the gas column in parsecs
h_g = 5

# Sigma g scaling factor
h_g_scale = 1/h_g

region_ID = 162

# dust to gas ratio
f_dg = 1/100

# Starting velocity in cm/s and time in years.
v_0 = 100
t_0 = np.power(10,time_slice)

# Column height of Hi and CO gas in parsecs
h_Hi = 100

## 'M51' or `NGC6946'
galaxy = 'M51'

# Define the column scale for the old stellar mass (r_star / h_star)
h_scale = 7.3

# Determine wether to age the stellar population as the galaxy simulation runs.
stellar_aging = True

# Determine whether to include the whole galaxy in the velocity calculations
includeWholeGalaxy = False


## There is a maximum and minimum grain size dictated by the grain files. We
## need to enforce that to avoid file bloat.
grain_min_def = 0.001
grain_max_def = 10

if grain_min < grain_min_def:
    print(f"minimum grain size is too small, setting to {grain_min_def}")
    grain_min = grain_min_def
    
if grain_max > grain_max_def:
    print(f"maximum grain size is too large, setting to {grain_max_def}")
    grain_max = grain_max_def

## Begin code to run the MC.
## -------------------------------------------------------------------------

# def GetBPASS(region_data):
#     ##########################################################################
#     # Create the filename for BPASS files.  This will need to be rewritten
#     # to take in data and find the metalicity and imf based on observation.
#     # Like most functions this will need moved to another file when it is
#     # finished.
#     #
#     #
#     ##########################################################################
    
#     possible_metal = [1, 2, 3, 4, 6, 8, 10, 14, 20, 30, 40]
#     possible_imf = np.array([ [100,100], [100,300], [135,300], [170,100], [170,300] ])
    
#     metal_guess = 22
#     metal_diff = lambda list_value : abs(list_value - metal_guess)
    
    
    
#     imf_i = 2
    
#     binary = 'bin'
    
    
#     imf = str(possible_imf[imf_i][0]) + '_' + str(possible_imf[imf_i][1])
    
#     if len(str(min(possible_metal, key=metal_diff))) > 1:
#         metalicity = '0' + str(min(possible_metal, key=metal_diff))
#     else:
#         metalicity = '00' + str(min(possible_metal, key=metal_diff))
    
#     BPASS_file = 'spectra-' + binary + '-imf' + imf + '.z' + metalicity + '.dat'
#     ionizing_file = 'ionizing-' + binary + '-imf' + imf + '.z' + metalicity + '.dat'
    
#     # Load the BPASS data
#     BPASS_data = load.model_output(BPASS_file)
#     # Make a copy of the BPASS data to downsample.
#     BPASS_data_r = BPASS_data.copy()
#     # Load the ionizing photon data.
#     ion_data = load.model_output(ionizing_file)
    
#     return BPASS_data, BPASS_data_r,ion_data

def CheckFiducialModel():
    
    grain_type_fid = "mix"
    time_slice_fid = 6.0
    BPASS_file_fid = "spectra-bin-imf135_300.z020.dat"
    wl_max_fid = 10
    wl_min_fid = 0.001
    grain_max_fid = 1
    grain_min_fid = 0.001
    
    if grain_type != grain_type_fid:
        print("Grain mix not fiducial model")
    if time_slice != time_slice_fid:
        print("Time not 1 Myr")
    if BPASS_file != BPASS_file_fid:
        print("Non fiducial BPASS model")
    if wl_max != wl_max_fid or wl_min != wl_min_fid:
        print("Wavelength bounds not fiducial")
    if grain_max != grain_max_fid or grain_min != grain_min_fid:
        print("Grain size bounds not fiducial")
    
    return True

CheckFiducialModel()

def GetGrainKappas(BPASS_data, time_slice, BPASS_data_r, wl_ref = 0.656, grain_mix = [0.5,0.45,0.05], grain_min = 0.001, grain_max = 1, grain_type = 'mix'):
    if grain_type == "mix":
        Sil_data, Sil_kappa = GetGrainData('Sil', grain_min, grain_max, BPASS_data_r, time_slice, wl_ref)
        Gra_data, Gra_kappa = GetGrainData('Gra', grain_min, grain_max, BPASS_data_r, time_slice, wl_ref)
        SiC_data, SiC_kappa = GetGrainData('SiC', grain_min, grain_max, BPASS_data_r, time_slice, wl_ref)
        
        Grain_data = Sil_data*grain_mix[0] + Gra_data*grain_mix[1] + SiC_data*grain_mix[2]
        kappa_data = Sil_kappa*grain_mix[0] + Gra_kappa*grain_mix[1] + SiC_kappa*grain_mix[2]

    else:
        Grain_data, kappa_data = GetGrainData(grain_type, grain_min, grain_max, BPASS_data_r, time_slice, wl_ref)
        
    return Grain_data, kappa_data

# Initial values, defined this way to make the photon state more explicit.
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

# Define the initial state of each photon. This is the dumbest possible way to do this.
#                            0        1      2       3          4         5        6       7        8
initial_state = np.array([x_init, y_init, z_init, phi_init, mu_init, lambda_init, g_av, albedo, tau_scale])

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

# def GetDraineAverage(folder, wavelengths):  Very old function, best left as a reference.
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

# Sigma_abs_av, Sigma_scatt_av, g_av, Q_ext = GetDraineAverage(folder, BPASS_data_r.WL.to_numpy())

# albedo = Sigma_scatt_av / (Sigma_scatt_av + Sigma_abs_av)

# DF = pd.DataFrame({'WL':BPASS_data_r.WL, 'Sigma-Abs':Sigma_abs_av, 'Sigma-Scatt':Sigma_scatt_av, 'g':g_av, 'Albedo':albedo, 'Q_ext': Q_ext})

# DF.to_csv('Gra Average Values.csv', index = False)

# # Plot the average sigma and g values.
# ###########################################################################
# fig, axs = plt.subplots(3,1, sharex = True)

# axs[0].plot(BPASS_data_r.WL.to_numpy(), g_av)
# axs[0].set_ylabel('<cos>')
# axs[2].set_xlabel('Wavelength (microns)')
# axs[0].set_xscale('log')
# # axs[0].set_title('g weighted by MRN distribution')

# axs[1].plot(BPASS_data_r.WL.to_numpy(), Sigma_abs_av)
# axs[1].set_ylabel(r'<$\sigma_{abs}$>')
# axs[1].set_yscale('log')
# # axs[1].xlabel('Wavelength (microns)')
# # axs[1].xscale('log')
# # axs[1].set_title(r'\sigma_{abs} weighted by MRN distribution')

# axs[2].plot(BPASS_data_r.WL.to_numpy(), Sigma_scatt_av)
# axs[2].set_ylabel(r'<$\sigma_{scatt}$>')
# axs[2].set_yscale('log')
# # axs[2].xlabel('Wavelength (microns)')
# # axs[2].xscale('log')
# # axs[2].set_title(r'\sigma_{scatt} weighted by MRN distribution')

# fig.tight_layout()
# plt.savefig('MRN averaged values.png', dpi = 200)
# ###########################################################################

# plt.plot(BPASS_data_r.WL.to_numpy(), Sigma_abs_av)
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

def CreatePhotons(num_photons,initial_state,rand_mu, PDF, Range, Grain_data):
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

def CheckBoundary(photon_state,boundary,boundary_momentum, PDF, Range, Grain_data):
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
            photon_state[boundary_photons] = CreatePhotons(len(boundary_photons), initial_state, rand_mu, PDF, Range, Grain_data)
        
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

def RunPhotons(tau_atm, num_photons, boundary, scatter, PDF, Range, Grain_data):
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
    photon_state = CreatePhotons(num_photons,initial_state,rand_mu, PDF, Range, Grain_data)
    
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
        photon_state, boundary_momentum = CheckBoundary(photon_state,boundary,boundary_momentum, PDF, Range, Grain_data)

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

def get_tau_IR(r, L_Bol, Sigma_g):
    #######################################################################################
    # This function returns the IR optical thickness of a region.
    #
    # r -- a number giving the physical size of the gas shell, in cm.
    # L_Bol -- A number giving the bolometric luminosity of the region in cgs.
    # Sigma_g -- A number giving the area mass density of the gas shell in cgs.
    #######################################################################################
    
    T_Eff = (L_Bol/(4*np.pi* r**2*sigma_SB))**0.25
    Kappa_R = 2.4*10**-4 * T_Eff**2
    tau_R_Eff = Kappa_R * Sigma_g
    
    try:
        tau_R_Eff = tau_R_Eff[0]
    except:
        pass
    
    if tau_R_Eff > 1:
        Temp = (3/4*(tau_R_Eff + 2/3)*T_Eff**4)**0.25
        tau_IR = 2.4*10**-4 * Temp**2 * Sigma_g
    else:
        tau_IR = tau_R_Eff
    
    # tau_IR = 0
    
    return tau_IR

vec_tau_IR = np.vectorize(get_tau_IR)

# def dvdr(r, v, M_g, M_new, M_old_i, L_Bol, tau_i, Sigma_g):
#     #######################################################################################
#     # This function returns the ODE for velocity: dv/dr
#     #
#     # r -- A number giving the physical size of the region in cm.
#     # v -- A number giving the velocity in cm/s.
#     # M_g, M_new -- Numbers giving the mass of the gas shell and new stellar mass, in g.
#     # M_old_i -- A number giving the initial enclosed mass of old stars.
#     # L_Bol -- A number giving the bolometric luminosity of the region in cgs.
#     # tau_i -- A number giving the initial hydrogen-alpha optical thickness.
#     # Sigma_g -- A number giving the area mass density of the gas shell in cgs.
#     #######################################################################################
    
#     global time_slice
#     global delta_t
    
#     tau_new = (h_g/r)**2*tau_i
    
#     momentum_new = interpn((MC_data.tau.values, MC_data.drop('tau', axis = 1).columns.values.astype(float))
#                            ,MC_data.drop('tau', axis = 1).to_numpy(), (tau_new, time_slice)) + get_tau_IR(r, L_Bol,  (h_g/r)**2 * Sigma_g)
    
#     M_old = M_old_i * (r / h_g)**3
    
#     # delta_t += [delta_r/v/31556952]
    
#     # time_slice = np.log10(10**time_slice + delta_t[-1])[0]
#     # if time_slice > 10.0:
#     #     time_slice = 10.0
    
#     return (-G*(M_g + M_new + M_old)/r**2 + momentum_new*L_Bol/(c*M_g))/v


def GetMC(MC_data, tau, new_time, L_Edd_DF, Grain_data, tau_cutoff = 0):

    _, _, ratio_rp_lambda_rp = GetRatios(L_Edd_DF, Grain_data, new_time, grain_type, wl_ref)

    MC_data_cutoff = MC_data.copy()

    if tau_cutoff > 0:
        mask = MC_data.tau*ratio_rp_lambda_rp < tau_cutoff
    
        for col in MC_data_cutoff.columns:
            MC_data_cutoff[col][mask] = 1- np.exp(-MC_data.tau[mask]*ratio_rp_lambda_rp)
    
        MC_data_cutoff.tau = MC_data.tau

    if np.any(new_time == MC_data_cutoff.drop('tau', axis=1).columns.values.astype(float)):
        MC_result = np.interp(tau, MC_data_cutoff.tau, MC_data_cutoff[str(new_time)])
    else:
        MC_result = interpn((MC_data_cutoff.tau.values, MC_data_cutoff.drop('tau', axis = 1).columns.values.astype(float))
                            ,MC_data_cutoff.drop('tau', axis = 1).to_numpy(), (tau, new_time))

    return MC_result

# def OldODEs(r, X, grain_type, grain_min, grain_max, BPASS_data, time_slice, tau_cutoff, regionData, grain_mix):
#     ##########################################################################
#     # Returns the ODE for dv/dr and dt/dr. I think this is the old one?????
#     #
#     # r -- A number giving the physical size of the region in cm.
#     # X -- A vector containing [v, t] in cm/s and years.
#     # tau_data -- A collection of variables needed to calculate a new tau
#     # # folder -- 
#     ##########################################################################
    
#     v, t = X
    
#     L_Bol_new = regionData.L_Bol
    
#     if stellar_aging:
#         if grain_type == 'mix':
#             _, _, Sil_kappa, _, _, _  = GetKappas('Draine data Sil/', grain_min, grain_max, BPASS_data.WL, BPASS_data, time_slice)
#             _, _, Gra_kappa, _, _, _ = GetKappas('Draine data Gra/', grain_min, grain_max, BPASS_data.WL, BPASS_data, time_slice)
#             _, _, SiC_kappa, _, _, _ = GetKappas('Draine data SiC/', grain_min, grain_max, BPASS_data.WL, BPASS_data, time_slice)

#             kappa = Sil_kappa*grain_mix[0] + Gra_kappa*grain_mix[1] + SiC_kappa*grain_mix[2]
#         else:
#             kappa = GetKappas('Draine data {}/'.format(grain_type), grain_min, grain_max, BPASS_data.WL, BPASS_data, time_slice)
        
#         tau_new = regionData.Sigma_g*(h_g/r)**2 * kappa[BPASS_data.WL == wl_ref*1000]*f_dg
#         new_time = np.log10(t)
#         L_ratio = np.interp(new_time,BPASS_data.drop('WL', axis = 1).sum(axis = 0).index.astype(float), BPASS_data.drop('WL', axis = 1).sum(axis = 0).values)/BPASS_data[str(time_slice)].sum()
#         L_Bol_new *= L_ratio
#         momentum_new = GetMC(tau_new, new_time, tau_cutoff) + get_tau_IR(r, L_Bol_new,  (h_g/r)**2 * regionData.Sigma_g)
#         print('L_Bol: {}, time: {}, momentum: {}'.format(L_Bol_new/L_sun_conversion, new_time, momentum_new))
#     else:
#         tau_new = np.array(regionData.tau_i*(h_g/r)**2)
#         momentum_new = GetMC(tau_new, time_slice, tau_cutoff) + get_tau_IR(r, L_Bol_new,  (h_g/r)**2 * regionData.Sigma_g)
    
#     dvdr = (-G*(regionData.M_g + regionData.M_new + regionData.M_old*(r/h_g)**3)/r**2 + momentum_new*L_Bol_new/(c*regionData.M_g))/v
#     print('dvdr: {}'.format(dvdr))
#     dtdr = 1/(v*31556952)

#     return dvdr, dtdr

def ODEs(r, X, grain_type, grain_min, grain_max, BPASS_data, time_slice, regionData, tau_cutoff, Grain_data, stellar_aging, gal_data):
    ##########################################################################
    # Returns the ODE for dv/dr and dt/dr.
    #
    # r -- A number giving the physical size of the region in cm.
    # X -- A vector containing [v, t] in cm/s and years.
    ##########################################################################
    
    v, t = X
    
    new_time = np.log10(t)
    
    L_ratio = np.interp(new_time,BPASS_data.drop('WL', axis = 1).sum(axis = 0).index.astype(float), BPASS_data.drop('WL', axis = 1).sum(axis = 0).values)/BPASS_data[str(time_slice)].sum()
    
    if includeWholeGalaxy:
        L_Bol_new = regionData.L_Bol * (1 - np.exp(-regionData.tau_i*ratio_rp_lambda_rp) + np.sum(gal_data.AHa_ratio_f * np.sin(np.arctan(r/gal_data.dist_to_ref/cm_pc))))
    else:
        L_Bol_new = regionData.L_Bol
    
    tau_new = (regionData.h_g_i/r)**2 * regionData.tau_i
    
    if stellar_aging:
        L_Bol_new *= L_ratio
        momentum_new = GetMC(MC_data, tau_new, new_time, L_Edd_DF, Grain_data, tau_cutoff) + get_tau_IR(r, L_Bol_new,  (regionData.h_g_i/r)**2 * regionData.Sigma_g)
    else:
        momentum_new = GetMC(MC_data, tau_new, time_slice, L_Edd_DF, Grain_data, tau_cutoff) + get_tau_IR(r, L_Bol_new,  (regionData.h_g_i/r)**2 * regionData.Sigma_g)
    
    dvdr = (-G*(regionData.M_g + regionData.M_CO_Hi * (min(r,h_Hi*cm_pc)/regionData.h_g_i)**3 + regionData.M_new + regionData.M_old*(min(r, regionData.H_old)/regionData.h_g_i)**3)/r**2 + momentum_new*L_Bol_new/(c*regionData.M_g))/v

    dtdr = 1/(v*31556952)

    # # Watch as I devolve into madness and just start printing every variable!
    # print('dvdr: {}'.format(dvdr))
    # print('dtdr: {}'.format(dtdr))
    # print('L_Bol: {}'.format(L_Bol_new))
    # print('momentum: {}'.format(momentum_new))

    return dvdr, dtdr

def FluxODEs(r, X, regionData, stellar_aging):
    ##########################################################################
    # Returns the ODE for dv/dr and dt/dr.
    #
    # r -- A number giving the physical size of the region in cm.
    # X -- A vector containing [v, t] in cm/s and years.
    ##########################################################################
    
    v, t = X

    F_g = 2*np.pi*G * ( regionData.Sigma_g + regionData.Sigma_new + regionData.Sigma_old * min(r/regionData.H_old,1) + regionData.Sigma_CO_Hi * min(r/h_Hi*cm_pc,1))
    
    Flux = regionData.F_Bol
    
    if stellar_aging:
        new_time = np.log10(t)
        L_ratio = np.interp(new_time,BPASS_data.drop('WL', axis = 1).sum(axis = 0).index.astype(float), BPASS_data.drop('WL', axis = 1).sum(axis = 0).values)/BPASS_data[str(time_slice)].sum()
    
        Momentum_new = GetMC(MC_data, np.array(regionData.tau_i), new_time, L_Edd_DF, Grain_data, tau_cutoff) + get_tau_IR(r, regionData.L_Bol*L_ratio,  regionData.Sigma_g)

        Flux *= Momentum_new*L_ratio
    else:
        Flux *= (regionData.Momentum + get_tau_IR(r, regionData.L_Bol,  regionData.Sigma_g))
    
    if r > regionData.Radius*cm_pc:
        Flux += regionData.C * regionData.F_gal * ((regionData.Radius*cm_pc-regionData.H_old)/r)**2
    elif r > regionData.H_old:
        Flux += regionData.C * regionData.F_gal * ((r-regionData.H_old)/regionData.Radius*cm_pc)**2

    dvdr = (Flux/(c*regionData.Sigma_g) - F_g)/v
    
    dtdr = 1/(v*31556952)
    
    return dvdr, dtdr

def runMC(BPASS_file, time_slice, num_photons, Range, Grain_data, momentum_method_1, photon_reduction=0, boundary = 'reflect', scatter = 'hg', save_MC = False):
    ##############################################################################
    # Runs the monte carlo simulation for a list of atmospheric depths.
    #
    # tau_list_calc -- numpy array containing the list of tau_lambda to be used.
    # num_photons -- an integer giving the number of photons to run at each atmosphere.
    # photon_reduction -- an integer giving the factor of reduction for photons at
    #                     thicker atmospheres.  Setting to 0 disables this.  For example
    #                     100 means that the thickest atmosphere hass 100 times fewer
    #                     photons than the thinnest.
    # boundary -- string giving the type of boundary to use.  Defaults to reflecting.
    #             It can be 'absorb', 'reemit', or 'reflect'.
    # scatter -- string giving the scattering type.  Defaults to hg.
    #             It can be 'hg', 'iso', 'draine'
    #             'draine' not currently functioning, missing alpha data for grains.
    # save_MC -- boolean giving whether to add the data to the MC data file.
    #            Defaults to false, this does not re-order the data and the MC data
    #            must be sorted by age (ascending) for interpolation to function.
    ##############################################################################
    
    # Load the BPASS data
    BPASS_data = load.model_output(BPASS_file)
    # Make a copy of the BPASS data to downsample.
    BPASS_data_r = BPASS_data.copy()
    
    # Convert the BPASS data to microns.
    BPASS_data_r.WL *= 10**-4
    
    BPASS_data_r = BPASS_data_r[ (BPASS_data_r.WL >= wl_min) & (BPASS_data_r.WL <= wl_max) ]
    
    BPASS_data_down = BPASS_data_r

    photon = BPASS_data_r[str(time_slice)] * BPASS_data_down.WL**2 / (h*c)
    
    # -----------------------------------------------------
        
    ## Find CDF
    # -----------------------------------------------------
    
    norm = np.trapz(photon, x = BPASS_data_down.WL)
    CDF = np.zeros_like(BPASS_data_down.WL)
    
    for i, _ in enumerate(BPASS_data_down.WL):
        phot = photon[0:i]
        WL = BPASS_data_r.WL[0:i]
        CDF[i] = np.trapz(phot, x = WL) / norm
    
    # plt.plot(BPASS_data_r.WL, CDF)
    # plt.ylabel('Continuous Distribution')
    # plt.xlabel('Wavelength (micron)')
    # plt.title('CDF by wavelength')
    # plt.savefig('CDF.png', dpi = 200)
    
    # -----------------------------------------------------
    
    
    ## Generate PDF from CDF.
    # -----------------------------------------------------
    
    PDF = np.gradient(CDF)
    
    if photon_reduction != 0:
        num_photons_tau = np.linspace(num_photons,num_photons/photon_reduction,len(tau_list_calc)).astype(int)
    
    for atm_i, tau_atm in enumerate(tau_list_calc):
        # Run the photons and return the mu at escape
        if photon_reduction != 0:
            num_photons = num_photons_tau[atm_i]
        escaped_mu, boundary_momentum, escaped_momentum, initial_momentum, _ = RunPhotons(tau_atm, num_photons, boundary, scatter, PDF, Range, Grain_data)
        print(tau_atm)
        momentum_method_1[atm_i] = (initial_momentum - escaped_momentum + boundary_momentum)/initial_momentum
    
    momentum_transfer = np.zeros(len(tau_list))
    momentum_transfer[tau_list < min_tau] = tau_list[tau_list < min_tau]*np.mean(momentum_method_1[0:10]/(1-np.exp(-tau_list_calc[0:10])))
    momentum_transfer[(tau_list >= min_tau) & (tau_list <= max_tau)] = momentum_method_1
    momentum_transfer[tau_list > max_tau] = np.mean(momentum_method_1[-5:])
    
    if save_MC:
        MC_data[str(time_slice)] = momentum_transfer
        MC_data.to_csv('detailed MC {} {}csv'.format(grain_type, BPASS_file.replace('.z',' z').replace('dat','')), index = False)
    
    return momentum_transfer

def ContinuousStarFormationSED(BPASS_data, time_list_exp, time_list, delta_t, age, rate):
    
    SED = np.zeros(100000)
    
    for j, time_slice in enumerate(time_list_exp):
        if time_slice <= age:
            SED += BPASS_data[time_list[j]]*delta_t[j]*rate
            # print("Age: {},  Time Slice: {}, Delta t: {}".format(age,time_slice,delta_t[j]))
        else:
            SED = SED/(10**6)
            return SED
    SED = SED/(10**6)
    
    return SED

def GetRatios(L_Edd_DF, Grain_data, time_slice, grain_type = 'mix', wl_ref = 0.656):
    ##########################################################################
    # Returns the ratio or different kappa values, interpolating in time if
    # necessary
    #
    # L_Edd_DF -- Pandas dataframe containing the <kappa_RP> data.
    # Grain_data -- Pandas dataframe containing the non flux averaged kappas.
    # time -- Float giving the log10 of the time in years.
    # grain_type -- string giving the type of grain(s) to use.
    # wl_ref -- The reference photon wavelength. Hydrogen alpha is the default.
    ##########################################################################
    
    ratio_lambda_f_lambda_rp = Grain_data.Kappa_F[Grain_data.WL.round(decimals=4) == wl_ref].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL.round(decimals = 4) == wl_ref].to_numpy()
    
    if time in L_Edd_DF.time:
        if grain_type == 'Gra':
            ratio_f_lambda_rp = L_Edd_DF.kappa_av_F_Gra[L_Edd_DF.time == time_slice].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
            ratio_rp_lambda_rp = L_Edd_DF.kappa_av_RP_Gra[L_Edd_DF.time == time_slice].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
        
        elif grain_type == 'Sil':        
            ratio_f_lambda_rp = L_Edd_DF.kappa_av_F_Sil[L_Edd_DF.time == time_slice].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
            ratio_rp_lambda_rp = L_Edd_DF.kappa_av_RP_Sil[L_Edd_DF.time == time_slice].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
        
        elif grain_type == 'SiC':
            ratio_f_lambda_rp = L_Edd_DF.kappa_av_F_SiC[L_Edd_DF.time == time_slice].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
            ratio_rp_lambda_rp = L_Edd_DF.kappa_av_RP_SiC[L_Edd_DF.time == time_slice].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
        
        elif grain_type =='mix':
            ratio_f_lambda_rp = (L_Edd_DF.kappa_av_F_Sil[L_Edd_DF.time == time_slice].to_numpy()*grain_mix[0] + 
                                 L_Edd_DF.kappa_av_F_Gra[L_Edd_DF.time == time_slice].to_numpy()*grain_mix[1] +
                                 L_Edd_DF.kappa_av_F_SiC[L_Edd_DF.time == time_slice].to_numpy()*grain_mix[2]
                                 )/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
            ratio_rp_lambda_rp = (L_Edd_DF.kappa_av_RP_Sil[L_Edd_DF.time == time_slice].to_numpy()*grain_mix[0] +
                                  L_Edd_DF.kappa_av_RP_Gra[L_Edd_DF.time == time_slice].to_numpy()*grain_mix[1] +
                                  L_Edd_DF.kappa_av_RP_SiC[L_Edd_DF.time == time_slice].to_numpy()*grain_mix[2]
                                  )/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
        
        else:
            print('Unknown grain type, this will not plot.')
    
    else:
        if grain_type == 'Gra':
            ratio_f_lambda_rp = np.interp(time_slice, L_Edd_DF.time, L_Edd_DF.kappa_av_F_Gra)/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
            ratio_rp_lambda_rp = np.interp(time_slice, L_Edd_DF.time, L_Edd_DF.kappa_av_RP_Gra)/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
        
        elif grain_type == 'Sil':
            
            ratio_f_lambda_rp = np.interp(time_slice, L_Edd_DF.time, L_Edd_DF.kappa_av_F_Sil)/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
            ratio_rp_lambda_rp = np.interp(time_slice, L_Edd_DF.time, L_Edd_DF.kappa_av_RP_Sil)/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
        
        elif grain_type == 'SiC':
            ratio_f_lambda_rp = np.interp(time_slice, L_Edd_DF.time, L_Edd_DF.kappa_av_F_SiC)/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
            ratio_rp_lambda_rp = np.interp(time_slice, L_Edd_DF.time, L_Edd_DF.kappa_av_RP_SiC)/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
        
        elif grain_type =='mix':
            ratio_f_lambda_rp = (np.interp(time_slice, L_Edd_DF.time, L_Edd_DF.kappa_av_F_Sil)*grain_mix[0] + 
                                 np.interp(time_slice, L_Edd_DF.time, L_Edd_DF.kappa_av_F_Gra)*grain_mix[1] +
                                 np.interp(time_slice, L_Edd_DF.time, L_Edd_DF.kappa_av_F_SiC)*grain_mix[2]
                                 )/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
            ratio_rp_lambda_rp = (np.interp(time_slice, L_Edd_DF.time, L_Edd_DF.kappa_av_RP_Sil)*grain_mix[0] +
                                  np.interp(time_slice, L_Edd_DF.time, L_Edd_DF.kappa_av_RP_Gra)*grain_mix[1] +
                                  np.interp(time_slice, L_Edd_DF.time, L_Edd_DF.kappa_av_RP_SiC)*grain_mix[2]
                                  )/Grain_data.Kappa_RP[Grain_data.WL == wl_ref].to_numpy()
        
        else:
            print('Unknown grain type, this will not plot.')

    return ratio_lambda_f_lambda_rp, ratio_f_lambda_rp, ratio_rp_lambda_rp

def GetTauRP(Sigma_g, kappa_RP, time, f_dg, r, h_g):
    
    tau = Sigma_g * (h_g/r)**2 * kappa_RP * f_dg
    
    return tau

def PrepareGalaxyData(galaxy, time_slice, BPASS_file, h_g):

    # Convert to cgs
    h_g = h_g * cm_pc
    
    # Define the files to read
    if galaxy == 'M51':
        gal_file_1 = 'NGC5194_Full_Aperture_table_2arc.dat'
        gal_file_2 = 'M51_3p6um_avApertures_MjySr.txt'
        gal_file_3 = 'NGC5194_gas_addendum.dat'
        # The center of the galaxy, obtained from http://ned.ipac.caltech.edu/
        # [RA, Dec] Numbers here are assuming that our dataset is in the J2000.0 frame.
        gal_center = [202.484167, 47.230556] ## M51
        Distance = 8.34*10**6 ## M51
    elif galaxy == 'NGC6946':
        gal_file_1 = 'NGC6946_Full_Aperture_table_2arc.dat'
        gal_file_2 = 'NGC6946_3p6um_avApertures_MjySr_radial.txt'
        gal_file_3 = 'NGC6946_gas_addendum.dat'
        # The center of the galaxy, obtained from http://ned.ipac.caltech.edu/
        # [RA, Dec] Numbers here are assuming that our dataset is in the J2000.0 frame.
        gal_center = [308.718015, 60.153915] ## NGC6946
        Distance = 7.72*10**6 ## NGC6946
    else:
        print('Unknown galaxy')
    
    # start_time = time.time()
    
    SM_file = BPASS_file.replace('spectra','starmass')
    
    # Load the BPASS data
    BPASS_data = load.model_output(BPASS_file)
    # Make a copy of the BPASS data to downsample.
    BPASS_data_r = BPASS_data.copy()
    # Load the ionizing photon data.
    ion_data = load.model_output(ionizing_file)
    yield_data = load.model_output(yields_file)
    starmass_data = load.model_output(SM_file)
    
    H_Alpha_ratio = (np.power(10,ion_data.halpha[ion_data.log_age == time_slice]))/L_sun_conversion/(BPASS_data[str(time_slice)].sum())
    
    # Convert the BPASS data to microns.
    BPASS_data_r.WL *= 10**-4
    
    BPASS_data_r = BPASS_data_r[ (BPASS_data_r.WL >= wl_min) & (BPASS_data_r.WL <= wl_max) ]
    
    time_list = BPASS_data.columns[BPASS_data.columns != 'WL']
    
    # kappa_df = np.zeros_like(time_list)
        
    # Resolution in arcseconds
    AngularResolution = 1
    
    # Spatial resolution in cm
    Resolution = Distance * np.tan(AngularResolution*np.pi/648000) * cm_pc
    
    time_list_exp = np.power(10,time_list.astype(float))
    
    Grain_data, kappa_data = GetGrainKappas(BPASS_data, time_slice, BPASS_data_r, wl_ref, grain_mix, grain_min, grain_max, grain_type)
    
    # kappa_data = pd.read_csv(kappa_file)
    
    kappa_data.WL = kappa_data.WL.round(4)
    Grain_data.WL = Grain_data.WL.round(4)
    
    Grain_data = pd.merge(Grain_data, kappa_data, left_on = 'WL', right_on = 'WL')
    
    ## Get Continuous Star Formation Rate data.
    ## -------------------------------------------------------------------------
    delta_t = np.zeros_like(time_list_exp)
    
    for i in range(len(delta_t)):
        if i == 0:
            delta_t[i] = 10**6.05
        else:
            delta_t[i] = 10**(6.15 + 0.1*(i-1))-10**(6.05 + 0.1*(i-1))
    
    
    
    rate_list = [1]
    
    for rate in rate_list:
        for i, age in enumerate(time_list_exp):
            
            SED = ContinuousStarFormationSED(BPASS_data, time_list_exp, time_list, delta_t, age, rate)
            
            if age == 10**6:
                continuous_SFR = SED
            else:
                continuous_SFR = pd.concat([continuous_SFR, SED.rename(time_list[i])], axis=1)
        
        continuous_SFR.insert(0, 'WL', BPASS_data.WL)
        
        mass = delta_t.copy()
        mass[1:] *= rate
        mass = np.cumsum(mass)
        
        continuous_SFR.WL *= 10**-4
        continuous_SFR = continuous_SFR[ (continuous_SFR.WL >= wl_min) & (continuous_SFR.WL <= wl_max) ]
        
        # age_list = np.insert(age_list, [0], 10**6)
        
        # plt.plot(time_list_exp, continuous_SFR.sum(axis = 0)[1:]/mass, label = str(rate) + ' solar M/year')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.plot(time_list_exp, continuous_SFR.sum(axis = 0)[1:], label = str(rate) + ' solar M/year')
    ## -------------------------------------------------------------------------
    
    # L_Edd_DF = pd.read_csv('L_Edd dataframe {}.csv'.format(BPASS_file.replace('.z',' z').replace('.dat','')))
    L_Edd_DF = GetLEddDataFrame(BPASS_file, grain_min, grain_max, wl_min, wl_max, f_dg)
    
    ratio_lambda_f_lambda_rp, ratio_f_lambda_rp, ratio_rp_lambda_rp = GetRatios(L_Edd_DF, Grain_data, time_slice, grain_type, wl_ref = wl_ref)
    
    # ## Combine galaxy data
    # ## ---------------------------------------------------------------------------
    
    # Open and read the files
    df1 = pd.read_csv(gal_file_1, delim_whitespace = True)
    df2 = pd.read_csv(gal_file_2, delim_whitespace = True)
    df3 = pd.read_csv(gal_file_3, delim_whitespace = True)
    
    # Merge the files and delete the extra id column.
    gal_data = pd.merge(df1,df2, left_on = 'ID', right_on = 'id')
    gal_data = gal_data.drop(columns='id')
    
    df3 = df3.drop(columns = 'RA')
    df3 = df3.drop(columns = 'Dec')
    
    gal_data = pd.merge(gal_data,df3, on = "ID")
    
    # Replace the missing high res co with low res measurements and make a new column denoting which measurement was used.
    mask = gal_data.highres_co > 0
    gal_data["highres"] = False
    gal_data["highres"][mask] = True
    gal_data["highres_co"][~mask] = gal_data["lowres_co"][~mask]
    
    # Calculate rho CO and rho Hi
    gal_data["rho_Hi"] = p_mass * gal_data.hi /(2*h_Hi*cm_pc)
    gal_data["rho_CO"] = p_mass * gal_data.highres_co /(2*h_Hi*cm_pc)
    
    # calculate the distance to galactic center.
    if galaxy == 'M51':
        gal_data['Dist_to_center'] = Distance * np.sqrt(np.tan(np.radians(gal_data['RA'] - gal_center[0]))**2 + np.tan(np.radians(gal_data['Dec'] - gal_center[1]))**2)
    elif galaxy == 'NGC6946':
        gal_data['Dist_to_center'] = Distance * np.sqrt(np.tan(np.radians(gal_data['RA_x'] - gal_center[0]))**2 + np.tan(np.radians(gal_data['Dec_x'] - gal_center[1]))**2)
    else:
        print('Unknown galaxy')
        
    # Calculate the old stellar column height.
    
    gal_data['H_old'] = gal_data.Dist_to_center/h_scale * cm_pc
    
    # hg_array = np.minimum(np.ones_like(gal_data.ID)*h_g, gal_data['H_old'])
    
    # gal_data['H_old'] = h_old
    
    # Define new columns
    # Find bolometric luminosty: L_Ha / 0.00724 (Kennicutt & Evans)
    gal_data['L_Bol'] = gal_data.LHa/H_Alpha_ratio.values
    
    gal_data['F_Bol'] = gal_data.L_Bol / (np.pi * Resolution**2)
    
    # Find Sigma_old, area density of old stars.
    # in solar masses/pc^2, then converted to cgs units.
    gal_data['Sigma_star'] = 350*gal_data['3.6um_aperture_average'] / cm_pc**2 * g_Msun
    gal_data['Sigma_g'] = gal_data.AHa/(1.086 * Grain_data.Kappa_F[Grain_data.WL.round(decimals=4) == wl_ref].to_numpy() * f_dg)
    
    gal_data["Sigma_CO_Hi"] = (gal_data.rho_Hi + gal_data.rho_CO) * 2*h_Hi*cm_pc
    
    # ## -------------------------------------------------------------------------
    
    
    ## Get Eddington ratios for galaxies
    ## ---------------------------------------------------------------------------
    
    # gal_file = 'M51 MC Complete.csv'
    # gal_file = 'M51.csv'
    
    # Range = BPASS_data_r.WL.to_numpy()
    
    # gal_data = pd.read_csv(gal_file)
    
    gal_data['Momentum'] = 0
    
    # gal_data['tau'] = gal_data.AHa/(1.086 * ratio_lambda_f_lambda_rp.values[0])
    
    gal_data['tau'] = gal_data['Sigma_g'] * Grain_data.Kappa_RP[Grain_data.WL.round(decimals=4) == wl_ref].to_numpy() * f_dg
    
    L_over_M = L_Edd_DF[L_Edd_DF.time == time_slice].L_bol_BPASS/L_Edd_DF[L_Edd_DF.time == time_slice].Mass
    
    # Mass of new stars, in grams.
    gal_data['Mass_new'] = gal_data.L_Bol/(L_over_M.to_numpy()[0] * L_sun_conversion  / g_Msun)
    gal_data['Sigma_new'] = gal_data['Mass_new'] / (np.pi * Resolution**2)
    
    # Mass of old stars in M_sun. No idea where the cosh came from
    gal_data['Mass_old'] = gal_data.Sigma_star * 2/3 * np.pi * np.minimum(h_g,gal_data['H_old'])**3 / (gal_data['H_old'])# * np.cosh(h_g/(2*(gal_data['H_old'])))**-2
    
    # Mass of gas in M_sun.
    gal_data['Mass_g'] = gal_data['Sigma_g']*4*np.pi*h_g**2
    
    gal_data["Mass_CO_Hi"] = 4/3 * np.pi * (gal_data.rho_CO + gal_data.rho_Hi) * np.minimum(h_g,h_Hi*cm_pc)**3
    
    
    
    gal_data['Mass_tot'] = (gal_data.Mass_g + gal_data.Mass_old + gal_data.Mass_new + gal_data.Mass_CO_Hi)
    
    # momentum_method_1 = np.zeros(num_atm)
    # escaped_mu = np.zeros_like(momentum_method_1)
    # scatter_count = np.zeros_like(momentum_method_1)
    
    # for i, row in gal_data.iterrows():
    
    #     tau_max = row.tau
        
    #     escaped_mu, boundary_momentum, escaped_momentum, initial_momentum, scatter_count = RunPhotons(tau_max , num_photons, boundary, scatter, PDF, Range, Grain_data)
        
    #     momentum = (initial_momentum - escaped_momentum + boundary_momentum)/initial_momentum
        
    #     gal_data.loc[gal_data.ID == row.ID, 'Momentum'] = momentum
        
    # MC_data = pd.read_csv('detailed MC.csv')
    momentum = GetMC(MC_data, gal_data.tau.values, time_slice, L_Edd_DF, Grain_data, tau_cutoff = tau_cutoff)
    gal_data['Momentum'] = momentum
    
    # Wind output in MSun m^2 / s^2 / yr. Multiply by 2*10^30 kg to get in Joules/year
    gal_data["E_dot_Wind"] = yield_data.E_wind[yield_data.log_age == time_slice].values / 10**6 * gal_data['Mass_new']/g_Msun
    
    gal_data['L_Edd'] = (G * c * gal_data['Mass_g'] * gal_data.Mass_tot  / (h_g**2 * (gal_data['Momentum'] + vec_tau_IR(h_g, gal_data.L_Bol, gal_data.Sigma_g))))
    
    gal_data['F_Edd'] = 2 * np.pi * G * c * gal_data['Sigma_g'] * (gal_data['Sigma_star'] * np.minimum(h_g /gal_data['H_old'],1) + gal_data['Sigma_g'] + gal_data['Sigma_new'] + gal_data["Sigma_CO_Hi"] * np.minimum(h_g /(100*cm_pc),1))/ (gal_data['Momentum'] + vec_tau_IR(h_g, gal_data.L_Bol, gal_data.Sigma_g))
    
    gal_data['tau_RP'] = gal_data.tau * ratio_rp_lambda_rp
    
    Edd_ratio = gal_data.L_Bol/gal_data.L_Edd-1
    Edd_ratio[Edd_ratio < 0] = 0
    
    gal_data['v_inf'] = np.sqrt(2*G*gal_data.Mass_tot/h_g)*np.sqrt(Edd_ratio)
    
    gal_data['R_UV'] = (L_Edd_DF.kappa_av_RP_Sil[L_Edd_DF.time == time_slice].to_numpy()[0] * f_dg * gal_data.Mass_g/(4*np.pi))**0.5
    # gal_data['v_inf_2'] = ((4 * gal_data.R_UV * gal_data.L_Bol)/(gal_data.Mass_g * c))**0.5    

    return gal_data, continuous_SFR, BPASS_data, Grain_data, kappa_data, L_Edd_DF, yield_data, starmass_data

def GetRegionData(gal_data, ID, dataType):
    
    regionData = gal_data[dataType][gal_data.ID == ID].values[0]
    
    return regionData

# class regionData:
#     def __init__(self, ID, gal_data):
#         self.ID = ID
#         self.gal_data = gal_data
#         self.M_g = GetRegionData(gal_data, self.ID, "Mass_g")
#         self.M_new = GetRegionData(gal_data, self.ID, "Mass_new")
#         self.M_old = GetRegionData(gal_data, self.ID, "Mass_old")
#         self.Sigma_old = GetRegionData(gal_data, self.ID, "Sigma_star")
#         self.Sigma_g = GetRegionData(gal_data, self.ID, "Sigma_g")
#         self.Sigma_new = GetRegionData(gal_data, self.ID, "Sigma_new")
#         self.tau_i = GetRegionData(gal_data, self.ID, "tau")
#         self.L_Bol = GetRegionData(gal_data, self.ID, "L_Bol")
#         self.H_old = GetRegionData(gal_data, self.ID, "H_old")
#         self.Momentum = GetRegionData(gal_data, self.ID, "Momentum")
#         self.F_Bol = GetRegionData(gal_data, self.ID, "F_Bol")
#         self.F_Edd = GetRegionData(gal_data, self.ID, "F_Edd")
#         self.C = 0.5
#         self.F_gal = GetRegionData(gal_data, self.ID, "F_Bol")
#         self.Radius = GetRegionData(gal_data, self.ID, "Dist_to_center")

def GetRegionVelocity(ID, gal_data):
    
    class regionData:
    	pass
     
    regionData.M_g = GetRegionData(gal_data, ID, "Mass_g")
    regionData.M_new = GetRegionData(gal_data, ID, "Mass_new")
    regionData.M_old = GetRegionData(gal_data, ID, "Mass_old")
    regionData.Sigma_old = GetRegionData(gal_data, ID, "Sigma_star")
    regionData.Sigma_g = GetRegionData(gal_data, ID, "Sigma_g")
    regionData.Sigma_new = GetRegionData(gal_data, ID, "Sigma_new")
    regionData.tau_i = GetRegionData(gal_data, ID, "tau")
    regionData.L_Bol = GetRegionData(gal_data, ID, "L_Bol")
    regionData.H_old = GetRegionData(gal_data, ID, "H_old")
    regionData.Momentum = GetRegionData(gal_data, ID, "Momentum")
    regionData.F_Bol = GetRegionData(gal_data, ID, "F_Bol")
    regionData.F_Edd = GetRegionData(gal_data, ID, "F_Edd")
    regionData.C = 0.5
    regionData.F_gal = GetRegionData(gal_data, ID, "F_Bol")
    regionData.Radius = GetRegionData(gal_data, ID, "Dist_to_center")
    regionData.h_g_i = h_g*cm_pc
    regionData.rho_CO = GetRegionData(gal_data, ID, "rho_CO")
    regionData.rho_Hi = GetRegionData(gal_data, ID, "rho_Hi")
    regionData.M_CO_Hi = GetRegionData(gal_data, ID, "Mass_CO_Hi")
    regionData.Sigma_CO_Hi = GetRegionData(gal_data, ID, "Sigma_CO_Hi")
    
    r = np.linspace(regionData.h_g_i, 100*regionData.H_old,num_bins)
    
    r_span = [r[0],r[-1]]

    # Generate the solutions to the ODE with and without stellar aging.
    stellar_aging = False
    spherical_IVP_solution_no_aging = inte.solve_ivp(ODEs, r_span, [v_0, t_0], args = [grain_type, grain_min, grain_max, BPASS_data, time_slice, regionData, tau_cutoff, Grain_data, stellar_aging, gal_data], method='Radau', max_step = cm_pc)
    planar_IVP_solution_no_aging = inte.solve_ivp(FluxODEs, r_span, [v_0, t_0], args = [regionData, stellar_aging], method='Radau', max_step = cm_pc)
    
    stellar_aging = True
    spherical_IVP_solution_aging = inte.solve_ivp(ODEs, r_span, [v_0, t_0], args = [grain_type, grain_min, grain_max, BPASS_data, time_slice, regionData, tau_cutoff, Grain_data, stellar_aging, gal_data], method='Radau', max_step = cm_pc)
    planar_IVP_solution_aging = inte.solve_ivp(FluxODEs, r_span, [v_0, t_0], args = [regionData, stellar_aging], method='Radau', max_step = cm_pc)
    
    return spherical_IVP_solution_no_aging, planar_IVP_solution_no_aging, spherical_IVP_solution_aging, planar_IVP_solution_aging

gal_data, continuous_SFR, BPASS_data, Grain_data, kappa_data, L_Edd_DF, yield_data, starmass_data = PrepareGalaxyData(galaxy, time_slice, BPASS_file, h_g)

# spherical_IVP_solution_no_aging, planar_IVP_solution_no_aging, spherical_IVP_solution_aging, planar_IVP_solution_aging = GetRegionVelocity(region_ID, gal_data)

# ## Plot velocities over time for spherical and planar
# # 2 Panels planar/spherical
# ##############################################################################
# fig, ax = plt.subplots(1,2, dpi = 200, sharey=True, figsize=(8, 6))

# ax[0].plot(spherical_IVP_solution_no_aging.t/cm_pc, spherical_IVP_solution_no_aging.y[0]/10**5, label="No Stellar Aging")
# ax[0].plot(spherical_IVP_solution_aging.t/cm_pc, spherical_IVP_solution_aging.y[0]/10**5, label="Stellar Aging")
# ax[1].plot(planar_IVP_solution_no_aging.t/cm_pc, planar_IVP_solution_no_aging.y[0]/10**5)
# ax[1].plot(planar_IVP_solution_aging.t/cm_pc, planar_IVP_solution_aging.y[0]/10**5)

# ax[0].set_xscale('log')
# ax[1].set_xscale('log')

# ax[0].set_yscale('log')
# ax[1].set_yscale('log')

# ax[0].set_ylim(1)
# ax[1].set_ylim(1)

# ax[0].legend(loc=4)

# ax[0].set_ylabel("Velocity (km/s)")

# ax[0].set_xlabel("Radius (pc)")
# ax[1].set_xlabel("Radius (pc)")
##############################################################################

## 1 Panel
##############################################################################
# fig, ax1 = plt.subplots(dpi = 200, figsize=(8, 6))

# ax2 = ax1.twinx()

# ax1.plot(spherical_IVP_solution_no_aging.t/cm_pc, spherical_IVP_solution_no_aging.y[0]/10**5, 'b', label="No Stellar Aging")
# ax1.plot(spherical_IVP_solution_aging.t/cm_pc, spherical_IVP_solution_aging.y[0]/10**5, 'k', label="Stellar Aging")
# ax2.plot(spherical_IVP_solution_aging.t/cm_pc, spherical_IVP_solution_aging.y[1], 'k:', label="Time (Spherical)")
# ax2.plot(spherical_IVP_solution_no_aging.t/cm_pc, spherical_IVP_solution_no_aging.y[1], 'b:')

# ax1.plot(planar_IVP_solution_no_aging.t/cm_pc, planar_IVP_solution_no_aging.y[0]/10**5, 'b--')
# ax1.plot(planar_IVP_solution_aging.t/cm_pc, planar_IVP_solution_aging.y[0]/10**5, 'k--')
# ax2.plot(planar_IVP_solution_aging.t/cm_pc, planar_IVP_solution_aging.y[1], 'k.-')
# ax2.plot(planar_IVP_solution_no_aging.t/cm_pc, planar_IVP_solution_no_aging.y[1], 'b.-', label="Time (Planar)")

# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax2.set_xscale('log')
# ax2.set_yscale('log')

# ax1.set_ylim(0.1)
# ax2.set_ylim(10**6)

# ax1.legend(loc = 8)
# ax2.legend(loc = 2)

# ax1.set_ylabel("Velocity (km/s)")
# ax1.set_xlabel("Radius (pc)")
# ax2.set_ylabel("Time (Years)")
##############################################################################

## 2 Panels radius/time
##############################################################################

# regionList = [162]

# regionList = gal_data[gal_data.L_Bol > gal_data.L_Edd].ID

# gal_data["velocity_no_aging"] = np.zeros_like(gal_data.ID)
# gal_data["distance_no_aging"] = np.zeros_like(gal_data.ID)
# gal_data["time_no_aging"] = np.zeros_like(gal_data.ID)

# gal_data["velocity_aging"] = np.zeros_like(gal_data.ID)
# gal_data["distance_aging"] = np.zeros_like(gal_data.ID)
# gal_data["time_aging"] = np.zeros_like(gal_data.ID)

   
# print(f"starting region_ID {region_ID}")

# spherical_IVP_solution_no_aging, planar_IVP_solution_no_aging, spherical_IVP_solution_aging, planar_IVP_solution_aging = GetRegionVelocity(region_ID, gal_data)

# fig, ax = plt.subplots(1,2, dpi = 200, figsize=(10, 4))
# ax2 = ax[0].twinx()
# ax3 = ax[1].twinx()

# momentum = np.zeros_like(spherical_IVP_solution_no_aging.t)
# tau = GetRegionData(gal_data, region_ID, "tau_RP") * (h_g / (spherical_IVP_solution_no_aging.t/cm_pc))**2

# # AHa = GetRegionData(gal_data, region_ID, "AHa")

# # Gamma = GetRegionData(gal_data, region_ID, "L_Bol") / GetRegionData(gal_data, region_ID, "L_Edd")

# # v_inf = GetRegionData(gal_data, region_ID, "v_inf") / 10**5
# # v_esc = v_inf / np.sqrt(Gamma)

# for i, r in enumerate(spherical_IVP_solution_no_aging.t):
#     momentum[i] = GetMC(MC_data, tau[i], np.log10(spherical_IVP_solution_no_aging.y[1])[i], L_Edd_DF, Grain_data, 0)

# # no_aging = ax[0].plot(spherical_IVP_solution_no_aging.t/cm_pc, spherical_IVP_solution_no_aging.y[0]/10**5, 'b', label="No Stellar Aging")

# # ax[0].plot(spherical_IVP_solution_no_aging.t/cm_pc, tau, 'g', label=r"$\langle\tau_{\rm RP}\rangle$")
# mo = ax2.plot(spherical_IVP_solution_no_aging.t/cm_pc, momentum, 'r', label="Momentum fraction")
# # ax[1].plot(spherical_IVP_solution_no_aging.y[1], tau, 'g', label=r"$\langle\tau_{\rm RP}\rangle$")
# ax3.plot(spherical_IVP_solution_no_aging.y[1], momentum, 'r', label="Momentum fraction")
# # ax[0].plot([spherical_IVP_solution_no_aging.t[0]/cm_pc,spherical_IVP_solution_no_aging.t[-1]/cm_pc], [v_inf, v_inf], 'purple', label = r"$V_\infty$")
# # ax[0].plot([spherical_IVP_solution_no_aging.t[0]/cm_pc,spherical_IVP_solution_no_aging.t[-1]/cm_pc], [v_esc, v_esc], 'orange', label = r"$V_{\rm esc}$")

# # ax[0].plot(planar_IVP_solution_no_aging.t/cm_pc, planar_IVP_solution_no_aging.y[0]/10**5, 'b--')
# aging = ax[0].plot(spherical_IVP_solution_aging.t/cm_pc, spherical_IVP_solution_aging.y[0]/10**5, 'k', label="Spherical")
# ax[0].plot(planar_IVP_solution_aging.t/cm_pc, planar_IVP_solution_aging.y[0]/10**5, 'k--')
# # ax[1].plot(spherical_IVP_solution_no_aging.y[1], spherical_IVP_solution_no_aging.y[0]/10**5, 'b')
# # ax[1].plot(planar_IVP_solution_no_aging.y[1], planar_IVP_solution_no_aging.y[0]/10**5, 'b--')
# ax[1].plot(spherical_IVP_solution_aging.y[1], spherical_IVP_solution_aging.y[0]/10**5, 'k')
# planar = ax[1].plot(planar_IVP_solution_aging.y[1], planar_IVP_solution_aging.y[0]/10**5, 'k--', label="Planar")

# ax[0].set_xscale('log')
# ax[1].set_xscale('log')

# ax[0].set_yscale('log')
# ax[1].set_yscale('log')
# ax2.set_yscale('log')
# ax3.set_yscale('log')

# ax2.axes.yaxis.set_ticklabels([])
# ax[1].axes.yaxis.set_ticklabels([])

# ax[0].set_ylim(1)
# ax[1].set_ylim(1)
# ax2.set_ylim(0.01,1.5)
# ax3.set_ylim(0.01,1.5)

# lines = aging + planar + mo 
# labels = [line.get_label() for line in lines]

# # gal_data.velocity_no_aging[gal_data.ID == region_ID] = np.max(spherical_IVP_solution_no_aging.y[0]/10**5)
# # gal_data.distance_no_aging[gal_data.ID == region_ID] = np.max(spherical_IVP_solution_no_aging.t/cm_pc)
# # gal_data.time_no_aging[gal_data.ID == region_ID] = np.max(spherical_IVP_solution_no_aging.y[1])

# # gal_data.velocity_aging[gal_data.ID == region_ID] = np.max(spherical_IVP_solution_aging.y[0]/10**5)
# # gal_data.distance_aging[gal_data.ID == region_ID] = np.max(spherical_IVP_solution_aging.t/cm_pc)
# # gal_data.time_aging[gal_data.ID == region_ID] = np.max(spherical_IVP_solution_aging.y[1])

# ax[0].legend(lines, labels)
# # ax[1].legend()
# # ax2.legend()

# AHa_label = r'$A_{\rm H\alpha}$'

# ax[0].set_ylabel("Velocity (km/s)")
# ax3.set_ylabel("Momentum Transfer Function")

# # fig.suptitle(f'Galaxy: NGC 5194    Region_ID: {region_ID}    {AHa_label}: {AHa:.2f}    $\Gamma$: {Gamma:.2f}', fontsize=13)

# # ax[1].text(2.5*10**6, 25, f"NGC 5194 Region {region_ID}")

# ax[0].set_xlabel("Radius (pc)")
# ax[1].set_xlabel("Time (yr)")
# plt.tight_layout()
# plt.show()

# plt.savefig(f'velocity plots/{galaxy} region_ID {region_ID} velocity R0 {h_g}.png')
# plt.close()
##############################################################################

## 2 Panels radius/time - Include galaxy data
##############################################################################

# includeWholeGalaxy = True

# _, _, ratio_rp_lambda_rp = GetRatios(L_Edd_DF, Grain_data, time_slice, grain_type, wl_ref = wl_ref)

# Distance = 8.34*10**6 ## M51

# gal_data["AHa_ratio_f"] = gal_data.AHa *np.exp(-gal_data.tau * ratio_rp_lambda_rp)/ gal_data.AHa[gal_data.ID == region_ID].values[0]
# gal_data["dist_to_ref"] = Distance * np.sqrt(np.tan(np.radians(gal_data['RA'] - float(gal_data[gal_data.ID == region_ID].RA)))**2 + np.tan(np.radians(gal_data['Dec'] - float(gal_data[gal_data.ID == region_ID].Dec)))**2)

# spherical_IVP_solution_whole_galaxy_no_aging, planar_IVP_solution_whole_galaxy_no_aging, spherical_IVP_solution_whole_galaxy_aging, planar_IVP_solution_whole_galaxy_aging = GetRegionVelocity(region_ID, gal_data)
# includeWholeGalaxy = False
# spherical_IVP_solution_no_aging, planar_IVP_solution_no_aging, spherical_IVP_solution_aging, planar_IVP_solution_aging = GetRegionVelocity(region_ID, gal_data)

# fig, ax = plt.subplots(1,2, dpi = 200, figsize=(10, 4))
# ax2 = ax[0].twinx()
# ax3 = ax[1].twinx()

# momentum = np.zeros_like(spherical_IVP_solution_no_aging.t)
# momentum_whole_galaxy = np.zeros_like(spherical_IVP_solution_whole_galaxy_no_aging.t)
# tau = GetRegionData(gal_data, region_ID, "tau_RP") * (h_g / (spherical_IVP_solution_no_aging.t/cm_pc))**2
# tau_whole_galaxy = GetRegionData(gal_data, region_ID, "tau_RP") * (h_g / (spherical_IVP_solution_whole_galaxy_no_aging.t/cm_pc))**2

# AHa = GetRegionData(gal_data, region_ID, "AHa")

# Gamma = GetRegionData(gal_data, region_ID, "L_Bol") / GetRegionData(gal_data, region_ID, "L_Edd")

# v_inf = GetRegionData(gal_data, region_ID, "v_inf") / 10**5
# v_esc = v_inf / np.sqrt(Gamma)

# for i, r in enumerate(spherical_IVP_solution_no_aging.t):
#     momentum[i] = GetMC(MC_data, tau[i], np.log10(spherical_IVP_solution_no_aging.y[1])[i], L_Edd_DF, Grain_data, 0)
    
# for i, r in enumerate(spherical_IVP_solution_whole_galaxy_no_aging.t):
#     momentum_whole_galaxy[i] = GetMC(MC_data, tau_whole_galaxy[i], np.log10(spherical_IVP_solution_whole_galaxy_no_aging.y[1])[i], L_Edd_DF, Grain_data, 0)

# no_aging = ax[0].plot(spherical_IVP_solution_no_aging.t/cm_pc, spherical_IVP_solution_no_aging.y[0]/10**5, 'b', label=f"No Stellar Aging, {region_ID} Only")

# # ax[0].plot(spherical_IVP_solution_no_aging.t/cm_pc, tau, 'g', label=r"$\langle\tau_{\rm RP}\rangle$")
# # mo = ax2.plot(spherical_IVP_solution_no_aging.t/cm_pc, momentum, 'r', label="Momentum fraction")
# # ax[1].plot(spherical_IVP_solution_no_aging.y[1], tau, 'g', label=r"$\langle\tau_{\rm RP}\rangle$")
# # ax3.plot(spherical_IVP_solution_no_aging.y[1], momentum, 'r', label="Momentum fraction")
# # ax[0].plot([spherical_IVP_solution_no_aging.t[0]/cm_pc,spherical_IVP_solution_no_aging.t[-1]/cm_pc], [v_inf, v_inf], 'purple', label = r"$V_\infty$")
# # ax[0].plot([spherical_IVP_solution_no_aging.t[0]/cm_pc,spherical_IVP_solution_no_aging.t[-1]/cm_pc], [v_esc, v_esc], 'orange', label = r"$V_{\rm esc}$")

# # ax[0].plot(planar_IVP_solution_no_aging.t/cm_pc, planar_IVP_solution_no_aging.y[0]/10**5, 'b--')
# aging = ax[0].plot(spherical_IVP_solution_aging.t/cm_pc, spherical_IVP_solution_aging.y[0]/10**5, 'k', label=f"Stellar Aging, {region_ID} Only")
# # ax[0].plot(planar_IVP_solution_aging.t/cm_pc, planar_IVP_solution_aging.y[0]/10**5, 'k--')
# ax[1].plot(spherical_IVP_solution_no_aging.y[1], spherical_IVP_solution_no_aging.y[0]/10**5, 'b')
# # ax[1].plot(planar_IVP_solution_no_aging.y[1], planar_IVP_solution_no_aging.y[0]/10**5, 'b--')
# ax[1].plot(spherical_IVP_solution_aging.y[1], spherical_IVP_solution_aging.y[0]/10**5, 'k', label=f"Region {region_ID} only")
# # ax[1].plot(planar_IVP_solution_aging.y[1], planar_IVP_solution_aging.y[0]/10**5, 'k--', label="Planar")



# no_aging_whole = ax[0].plot(spherical_IVP_solution_whole_galaxy_no_aging.t/cm_pc, spherical_IVP_solution_whole_galaxy_no_aging.y[0]/10**5, 'g', label="No Stellar Aging, Whole Galaxy")

# # ax[0].plot(spherical_IVP_solution_whole_galaxy_no_aging.t/cm_pc, tau, 'g', label=r"$\langle\tau_{\rm RP}\rangle$")
# # mo_whole = ax2.plot(spherical_IVP_solution_whole_galaxy_no_aging.t/cm_pc, momentum_whole_galaxy, 'r--', label="Momentum fraction")
# # ax[1].plot(spherical_IVP_solution_whole_galaxy_no_aging.y[1], tau, 'g', label=r"$\langle\tau_{\rm RP}\rangle$")
# # ax3.plot(spherical_IVP_solution_whole_galaxy_no_aging.y[1], momentum_whole_galaxy, 'r--', label="Momentum fraction")
# # ax[0].plot([spherical_IVP_solution_whole_galaxy_no_aging.t[0]/cm_pc,spherical_IVP_solution_whole_galaxy_no_aging.t[-1]/cm_pc], [v_inf, v_inf], 'purple', label = r"$V_\infty$")
# # ax[0].plot([spherical_IVP_solution_whole_galaxy_no_aging.t[0]/cm_pc,spherical_IVP_solution_whole_galaxy_no_aging.t[-1]/cm_pc], [v_esc, v_esc], 'orange', label = r"$V_{\rm esc}$")

# # ax[0].plot(planar_IVP_solution_whole_galaxy_no_aging.t/cm_pc, planar_IVP_solution_whole_galaxy_no_aging.y[0]/10**5, 'b--')
# aging_whole = ax[0].plot(spherical_IVP_solution_whole_galaxy_aging.t/cm_pc, spherical_IVP_solution_whole_galaxy_aging.y[0]/10**5, 'r', label="Stellar Aging, Whole Galaxy")
# # ax[0].plot(planar_IVP_solution_whole_galaxy_aging.t/cm_pc, planar_IVP_solution_whole_galaxy_aging.y[0]/10**5, 'k--')
# ax[1].plot(spherical_IVP_solution_whole_galaxy_no_aging.y[1], spherical_IVP_solution_whole_galaxy_no_aging.y[0]/10**5, 'g')
# # ax[1].plot(planar_IVP_solution_whole_galaxy_no_aging.y[1], planar_IVP_solution_whole_galaxy_no_aging.y[0]/10**5, 'b--')
# ax[1].plot(spherical_IVP_solution_whole_galaxy_aging.y[1], spherical_IVP_solution_whole_galaxy_aging.y[0]/10**5, 'r', label="All clusters")
# # ax[1].plot(planar_IVP_solution_whole_galaxy_aging.y[1], planar_IVP_solution_whole_galaxy_aging.y[0]/10**5, 'k--', label="Planar")


# ax[0].set_xscale('log')
# ax[1].set_xscale('log')

# ax[0].set_yscale('log')
# ax[1].set_yscale('log')
# ax2.set_yscale('log')
# ax3.set_yscale('log')

# ax2.set_ylim(0.001)
# ax3.set_ylim(0.001)

# ax2.axes.yaxis.set_ticklabels([])
# ax[1].axes.yaxis.set_ticklabels([])

# ax[0].set_ylim(1)
# ax[1].set_ylim(1)

# lines = no_aging + aging + no_aging_whole + aging_whole# + mo
# labels = [line.get_label() for line in lines]

# ax[0].legend(lines, labels)
# # ax[1].legend()
# # ax2.legend()

# print(f"AHa: {AHa}")
# print(f"Gamma: {Gamma}")

# AHa_label = r'$A_{\rm H\alpha}$'

# ax[0].set_ylabel("Velocity (km/s)")
# ax3.set_ylabel("Momentum Transfer Function")

# # fig.suptitle(f'Galaxy: NGC 5194    Region: {region_ID}    {AHa_label}: {AHa:.2f}    $\Gamma$: {Gamma:.2f}', fontsize=13)

# ax[0].set_xlabel("Radius (pc)")
# ax[1].set_xlabel("Time (yr)")
# plt.tight_layout()

##############################################################################

## 2 Panels three regions
##############################################################################

# gal_data_M51, continuous_SFR, BPASS_data, Grain_data, kappa_data, L_Edd_DF, yield_data, starmass_data = PrepareGalaxyData("M51", time_slice, BPASS_file, h_g)

# _, _, region_162_spherical_IVP_solution_aging, region_162_planar_IVP_solution_aging = GetRegionVelocity(162, gal_data_M51)
# _, _, region_37_spherical_IVP_solution_aging, region_37_planar_IVP_solution_aging = GetRegionVelocity(37, gal_data_M51)

# fig, ax = plt.subplots(1,2, dpi = 200, figsize=(10, 4))
# ax2 = ax[0].twinx()
# ax3 = ax[1].twinx()

# momentum_162 = np.zeros_like(region_162_spherical_IVP_solution_aging.t)

# tau_162 = GetRegionData(gal_data_M51, 162, "tau_RP") * (h_g / (region_162_spherical_IVP_solution_aging.t/cm_pc))**2

# for i, r in enumerate(region_162_spherical_IVP_solution_aging.t):
#     momentum_162[i] = GetMC(MC_data, tau_162[i], np.log10(region_162_spherical_IVP_solution_aging.y[1])[i], L_Edd_DF, Grain_data, 0)

# ax[0].plot(region_162_planar_IVP_solution_aging.t/cm_pc, region_162_planar_IVP_solution_aging.y[0]/10**5, 'b')
# ax[0].plot(region_162_spherical_IVP_solution_aging.t/cm_pc, region_162_spherical_IVP_solution_aging.y[0]/10**5, 'k', label="Region 162")
# ax[1].plot(region_162_planar_IVP_solution_aging.y[1], region_162_planar_IVP_solution_aging.y[0]/10**5, 'b')
# ax[1].plot(region_162_spherical_IVP_solution_aging.y[1], region_162_spherical_IVP_solution_aging.y[0]/10**5, 'k')

# ax2.plot(region_162_spherical_IVP_solution_aging.t/cm_pc, momentum_162, 'r')
# ax3.plot(region_162_spherical_IVP_solution_aging.y[1], momentum_162, 'r', label = "Region 162")

# momentum_37 = np.zeros_like(region_37_spherical_IVP_solution_aging.t)

# tau_37 = GetRegionData(gal_data_M51, 37, "tau_RP") * (h_g / (region_37_spherical_IVP_solution_aging.t/cm_pc))**2

# for i, r in enumerate(region_37_spherical_IVP_solution_aging.t):
#     momentum_37[i] = GetMC(MC_data, tau_37[i], np.log10(region_37_spherical_IVP_solution_aging.y[1])[i], L_Edd_DF, Grain_data, 0)

# ax[0].plot(region_37_planar_IVP_solution_aging.t/cm_pc, region_37_planar_IVP_solution_aging.y[0]/10**5, 'b--')
# ax[0].plot(region_37_spherical_IVP_solution_aging.t/cm_pc, region_37_spherical_IVP_solution_aging.y[0]/10**5, 'k--', label="Region 37")
# ax[1].plot(region_37_planar_IVP_solution_aging.y[1], region_37_planar_IVP_solution_aging.y[0]/10**5, 'b--')
# ax[1].plot(region_37_spherical_IVP_solution_aging.y[1], region_37_spherical_IVP_solution_aging.y[0]/10**5, 'k--')

# ax2.plot(region_37_spherical_IVP_solution_aging.t/cm_pc, momentum_37, 'r--')
# ax3.plot(region_37_spherical_IVP_solution_aging.y[1], momentum_37, 'r--', label = "Region 37")

# gal_data_NGC6946, continuous_SFR, BPASS_data, Grain_data, kappa_data, L_Edd_DF, yield_data, starmass_data = PrepareGalaxyData("NGC6946", time_slice, BPASS_file, h_g)

# _, _, region_27_spherical_IVP_solution_aging, region_27_planar_IVP_solution_aging = GetRegionVelocity(27, gal_data_NGC6946)

# momentum_27 = np.zeros_like(region_27_spherical_IVP_solution_aging.t)

# tau_27 = GetRegionData(gal_data_NGC6946, 27, "tau_RP") * (h_g / (region_27_spherical_IVP_solution_aging.t/cm_pc))**2

# for i, r in enumerate(region_27_spherical_IVP_solution_aging.t):
#     momentum_27[i] = GetMC(MC_data, tau_27[i], np.log10(region_27_spherical_IVP_solution_aging.y[1])[i], L_Edd_DF, Grain_data, 0)
    
# for i, r in enumerate(region_27_spherical_IVP_solution_aging.t):
#     momentum_27[i] = GetMC(MC_data, tau_27[i], np.log10(region_27_spherical_IVP_solution_aging.y[1])[i], L_Edd_DF, Grain_data, 0)

# ax[0].plot(region_27_planar_IVP_solution_aging.t/cm_pc, region_27_planar_IVP_solution_aging.y[0]/10**5, 'b:')
# ax[0].plot(region_27_spherical_IVP_solution_aging.t/cm_pc, region_27_spherical_IVP_solution_aging.y[0]/10**5, 'k:', label="Region 27")
# ax[1].plot(region_27_planar_IVP_solution_aging.y[1], region_27_planar_IVP_solution_aging.y[0]/10**5, 'b:')
# ax[1].plot(region_27_spherical_IVP_solution_aging.y[1], region_27_spherical_IVP_solution_aging.y[0]/10**5, 'k:')

# ax2.plot(region_27_spherical_IVP_solution_aging.t/cm_pc, momentum_27, 'r:')
# ax3.plot(region_27_spherical_IVP_solution_aging.y[1], momentum_27, 'r:', label = "Region 27")

# ax[0].set_xscale('log')
# ax[1].set_xscale('log')
# ax[0].set_yscale('log')
# ax[1].set_yscale('log')
# ax2.set_yscale('log')
# ax3.set_yscale('log')

# ax2.set_ylim(0.001)
# ax3.set_ylim(0.001)

# ax2.axes.yaxis.set_ticklabels([])
# ax[1].axes.yaxis.set_ticklabels([])

# ax[0].set_ylim(1)
# ax[1].set_ylim(1)

# ax[0].legend()
# ax3.legend()

# ax[0].set_ylabel("Velocity (km/s)")
# ax3.set_ylabel("Momentum Transfer Function")

# ax[0].set_xlabel("Radius (pc)")
# ax[1].set_xlabel("Time (yr)")
# plt.tight_layout()

# print("Data for {} pc.".format(h_g))
# print("Region 37")
# print("LBol: {}".format(gal_data_M51[gal_data_M51.ID == 37].L_Bol.values[0]/L_sun_conversion))
# print("AHa: {}".format(gal_data_M51[gal_data_M51.ID == 37].AHa.values[0]))
# print("Mgas: {}".format(gal_data_M51[gal_data_M51.ID == 37].Mass_g.values[0]/g_Msun))
# print("Mstar: {}".format(gal_data_M51[gal_data_M51.ID == 37].Mass_new.values[0]/g_Msun))
# print("Mold: {}".format(gal_data_M51[gal_data_M51.ID == 37].Mass_old.values[0]/g_Msun))
# print("Ratio: {}".format(gal_data_M51[gal_data_M51.ID == 37].L_Bol.values[0]/gal_data_M51[gal_data_M51.ID == 37].L_Edd.values[0]))
# print("Vmax: {}".format(np.max(region_37_spherical_IVP_solution_aging.y[0]/10**5)))
# print("////////////////////////////////////////////////")
# print("Region 162")
# print("LBol: {}".format(gal_data_M51[gal_data_M51.ID == 162].L_Bol.values[0]/L_sun_conversion))
# print("AHa: {}".format(gal_data_M51[gal_data_M51.ID == 162].AHa.values[0]))
# print("Mgas: {}".format(gal_data_M51[gal_data_M51.ID == 162].Mass_g.values[0]/g_Msun))
# print("Mstar: {}".format(gal_data_M51[gal_data_M51.ID == 162].Mass_new.values[0]/g_Msun))
# print("Mold: {}".format(gal_data_M51[gal_data_M51.ID == 162].Mass_old.values[0]/g_Msun))
# print("Ratio: {}".format(gal_data_M51[gal_data_M51.ID == 162].L_Bol.values[0]/gal_data_M51[gal_data_M51.ID == 162].L_Edd.values[0]))
# print("Vmax: {}".format(np.max(region_162_spherical_IVP_solution_aging.y[0]/10**5)))
# print("////////////////////////////////////////////////")
# print("Region 27")
# print("LBol: {}".format(gal_data_NGC6946[gal_data_NGC6946.ID == 27].L_Bol.values[0]/L_sun_conversion))
# print("AHa: {}".format(gal_data_NGC6946[gal_data_NGC6946.ID == 27].AHa.values[0]))
# print("Mgas: {}".format(gal_data_NGC6946[gal_data_NGC6946.ID == 27].Mass_g.values[0]/g_Msun))
# print("Mstar: {}".format(gal_data_NGC6946[gal_data_NGC6946.ID == 27].Mass_new.values[0]/g_Msun))
# print("Mold: {}".format(gal_data_NGC6946[gal_data_NGC6946.ID == 27].Mass_old.values[0]/g_Msun))
# print("Ratio: {}".format(gal_data_NGC6946[gal_data_NGC6946.ID == 27].L_Bol.values[0]/gal_data_NGC6946[gal_data_NGC6946.ID == 27].L_Edd.values[0]))
# print("Vmax: {}".format(np.max(region_27_spherical_IVP_solution_aging.y[0]/10**5)))

##############################################################################

## RP / P_CR / nkT plot
##############################################################################

# lambda_CR = 0.1

# spherical_IVP_solution_no_aging, planar_IVP_solution_no_aging, spherical_IVP_solution_aging, planar_IVP_solution_aging = GetRegionVelocity(region_ID, gal_data)

# t_diff = (3*(spherical_IVP_solution_aging.t)**2 / (c*lambda_CR*cm_pc)) / 31536000
# t_exp = spherical_IVP_solution_aging.t/spherical_IVP_solution_aging.y[0] / 31536000
# t_pion = 5*10**7 / (gal_data[gal_data.ID == region_ID].Mass_g.values[0] / p_mass * 3 / (4*np.pi*spherical_IVP_solution_aging.t)**3)

# t_min = np.minimum(t_diff,t_exp)
# t_min = np.minimum(t_min,t_pion)

# time_slices = np.log10(spherical_IVP_solution_aging.y[1])

# L_ratio = np.interp(time_slices,BPASS_data.drop('WL', axis = 1).sum(axis = 0).index.astype(float), BPASS_data.drop('WL', axis = 1).sum(axis = 0).values)/BPASS_data[str(time_slice)].sum()

# class regionData:
#  	pass

# regionData.M_g = GetRegionData(gal_data, region_ID, "Mass_g")
# regionData.M_new = GetRegionData(gal_data, region_ID, "Mass_new")
# regionData.M_old = GetRegionData(gal_data, region_ID, "Mass_old")
# regionData.Sigma_old = GetRegionData(gal_data, region_ID, "Sigma_star")
# regionData.Sigma_g = GetRegionData(gal_data, region_ID, "Sigma_g")
# regionData.Sigma_new = GetRegionData(gal_data, region_ID, "Sigma_new")
# regionData.tau_i = GetRegionData(gal_data, region_ID, "tau")
# regionData.L_Bol = GetRegionData(gal_data, region_ID, "L_Bol")
# regionData.H_old = GetRegionData(gal_data, region_ID, "H_old")
# regionData.Momentum = GetRegionData(gal_data, region_ID, "Momentum")
# regionData.F_Bol = GetRegionData(gal_data, region_ID, "F_Bol")
# regionData.F_Edd = GetRegionData(gal_data, region_ID, "F_Edd")
# regionData.C = 0.5
# regionData.F_gal = GetRegionData(gal_data, region_ID, "F_Bol")
# regionData.Radius = GetRegionData(gal_data, region_ID, "Dist_to_center")
# regionData.h_g_i = h_g*cm_pc

# # _, _, ratio_rp_lambda_rp = GetRatios(L_Edd_DF, Grain_data, time_slice, grain_type, wl_ref = wl_ref)

# # if includeWholeGalaxy:
# #     L_Bol_new = regionData.L_Bol * (1 - np.exp(-regionData.tau_i*ratio_rp_lambda_rp) + np.sum(gal_data.AHa_ratio_f * np.sin(np.arctan(r/gal_data.dist_to_ref/cm_pc))))
# # else:
#     # L_Bol_new = regionData.L_Bol

# tau_new = (regionData.h_g_i/spherical_IVP_solution_aging.t)**2 * regionData.tau_i

# L_Bol_new = L_ratio*regionData.L_Bol

# momentum_new = np.zeros_like(spherical_IVP_solution_aging.t)

# for i, r in enumerate(spherical_IVP_solution_aging.t):
#     momentum_new[i] = GetMC(MC_data, tau_new[i], time_slices[i], L_Edd_DF, Grain_data, tau_cutoff) + get_tau_IR(r, L_Bol_new,  (regionData.h_g_i/r)**2 * regionData.Sigma_g)

# RP = momentum_new * L_Bol_new / c / (4*np.pi * spherical_IVP_solution_aging.t**2) * 6.242*10**11

# E_dot_CR = 0.1*np.interp(time_slices, yield_data.log_age, yield_data.E_wind.values)/ 10**6 * regionData.M_new/g_Msun*2*10**30

# P_dot_CR = E_dot_CR*t_min/(4*np.pi*(spherical_IVP_solution_aging.t)**3) * 6.242*10**18

# k = 8.617333262 * 10**-5

# T = 10**4

# alpha_rec = 10**-14

# LHa_ratio = np.interp(time_slices, BPASS_data.drop('WL', axis = 1).sum(axis = 0).index.astype(float), BPASS_data[BPASS_data.WL == 656].drop('WL', axis = 1).values[0])/BPASS_data[(BPASS_data.WL == 656)][str(time_slice)].values

# LHa = gal_data[gal_data.ID == region_ID].LHa.values * LHa_ratio

# n = np.sqrt(LHa / (alpha_rec * 4/3*np.pi*(spherical_IVP_solution_aging.t)**3))

# plt.figure(dpi = 200)
# plt.plot(spherical_IVP_solution_aging.t/cm_pc, P_dot_CR, label = r"$P_{\rm CR}$")
# plt.plot(spherical_IVP_solution_aging.t/cm_pc, RP, label = "RP")
# plt.plot(spherical_IVP_solution_aging.t/cm_pc, n*k*T, label = "nkT")
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Distance (pc)')
# plt.legend()
# plt.ylabel(r'P $(\rm{ev/cm}^3)$')


##############################################################################

## Plot L gamma
#############################################################################

# lambda_CR = 0.01

# class regionData:
#  	pass

# regionData.M_new = GetRegionData(gal_data, region_ID, "Mass_new")


# spherical_IVP_solution_no_aging, planar_IVP_solution_no_aging, spherical_IVP_solution_aging, planar_IVP_solution_aging = GetRegionVelocity(region_ID, gal_data)

# t_diff = (3*(spherical_IVP_solution_aging.t)**2 / (c*lambda_CR*cm_pc)) / 31536000
# t_pion = 5*10**7 / (gal_data[gal_data.ID == region_ID].Mass_g.values[0] / p_mass * 3 / (4*np.pi*spherical_IVP_solution_aging.t)**3)

# time_slices = np.log10(spherical_IVP_solution_aging.y[1])

# E_dot_CR = 0.1*np.interp(time_slices, yield_data.log_age, yield_data.E_wind.values)/ 10**6 * regionData.M_new/g_Msun*2*10**30

# L_gamma = E_dot_CR / 3 * np.minimum(np.ones_like(t_diff),np.sqrt(np.divide(t_diff,t_pion))) * 8.241*10**-35

# plt.figure(dpi = 200)
# plt.plot(spherical_IVP_solution_aging.t/cm_pc, L_gamma)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Distance (pc)')
# plt.ylabel(r'$L_\gamma\,(L_{\odot})$')

##############################################################################



## MC result plot
#############################################################################
# plt.figure(dpi = 200)

# _, _, ratio_rp_lambda_rp = GetRatios(L_Edd_DF, Grain_data, '6.0', grain_type, wl_ref = wl_ref)
# _, _, ratio_rp_lambda_rp_3 = GetRatios(L_Edd_DF, Grain_data, '6.5', grain_type, wl_ref = wl_ref)
# _, _, ratio_rp_lambda_rp_10 = GetRatios(L_Edd_DF, Grain_data, '7.0', grain_type, wl_ref = wl_ref)
# _, _, ratio_rp_lambda_rp_100 = GetRatios(L_Edd_DF, Grain_data, '8.0', grain_type, wl_ref = wl_ref)

# directed_result = pd.read_csv('directed MC result.csv')

# plt.plot(MC_data.tau * ratio_rp_lambda_rp, MC_data['6.0'], label = "MC result (isotropic, 1 Myr)")
# plt.plot(MC_data.tau * ratio_rp_lambda_rp_3, MC_data['6.5'], label = "MC result (isotropic, 3 Myr)")
# plt.plot(MC_data.tau * ratio_rp_lambda_rp_10, MC_data['7.0'], label = "MC result (isotropic, 10 Myr)")
# # plt.plot(MC_data.tau * ratio_rp_lambda_rp_100, MC_data['8.0'], label = "MC result (isotropic, 100 Myr)")
# plt.plot(directed_result.tau * ratio_rp_lambda_rp, directed_result['6.0'], 'purple', label = "MC result (beamed, 1 Myr)")
# plt.plot(MC_data.tau * ratio_rp_lambda_rp, 1 - np.exp(-MC_data.tau * ratio_rp_lambda_rp), 'k', label = r"$1-\exp(-\langle\tau_{\rm RP}\rangle)$")

# plt.xscale('log')
# plt.yscale('log')

# plt.grid(which="major", alpha = 0.4, color='k')
# plt.grid(which="minor", alpha = 0.8, linestyle=":")

# plt.xlim(0.1,20)
# plt.ylim(0.1,2)

# plt.xlabel(r"$\langle\tau_{\rm RP}\rangle$")
# plt.ylabel(r"Fractional Momentum Transfer, $f_{(\langle \rm RP \rangle)}$")

# plt.legend(loc = 4)

# plt.tight_layout()

##############################################################################


## Wibking comparison plot
##############################################################################

# def Integrand(x):
    
#     integrand = np.exp(-x)/x
    
#     return integrand

# def Integral(x):
#     integral = inte.quad(Integrand, x, np.inf)
    
#     return integral

# VecInt = np.vectorize(Integral)

# directed_result = pd.read_csv('directed MC result.csv')

# tau_data = MC_data.tau

# expt_t, _ = VecInt(tau_data*ratio_rp_lambda_rp)

# plt.figure(dpi = 200)

# ax = plt.gca()
# ax.set_aspect(0.8)

# plt.plot(tau_data*ratio_rp_lambda_rp, tau_data*ratio_rp_lambda_rp/(1-np.exp(-tau_data*ratio_rp_lambda_rp)), 'b--', label="Beamed single scattering")
# plt.plot(tau_data*ratio_rp_lambda_rp, 3/2 * tau_data * ratio_rp_lambda_rp * (1+np.exp(-tau_data*ratio_rp_lambda_rp)*(tau_data*ratio_rp_lambda_rp/2 - (tau_data*ratio_rp_lambda_rp)**2 /2 -1) +(tau_data*ratio_rp_lambda_rp)**3 / 2 * (expt_t))**-1, 'k', label="isotropic single scattering")
# plt.plot(directed_result.tau*ratio_rp_lambda_rp, directed_result.tau*ratio_rp_lambda_rp /directed_result['6.0'], 'c', label = "Beamed MC Results")
# plt.plot(tau_data*ratio_rp_lambda_rp, tau_data*ratio_rp_lambda_rp/GetMC(MC_data, tau_data, 6.0, 0), 'g', label="Isotropic MC results, 1 Myr")
# _, _, ratio_rp_lambda_rp = GetRatios(L_Edd_DF, Grain_data, '8.0', grain_type, wl_ref = wl_ref)
# plt.plot(tau_data*ratio_rp_lambda_rp, tau_data*ratio_rp_lambda_rp/GetMC(MC_data, tau_data, 8.0, 0), 'r', label="Isotropic MC results, 100 Myr")

# plt.xscale('log')
# plt.yscale('log')

# plt.grid(which="major", alpha = 0.4, color='k')
# plt.grid(which="minor", alpha = 0.8, linestyle=":")

# plt.xlim(0.1,10)
# plt.ylim(0.1,20)

# plt.xlabel(r"$\langle\tau_{\rm RP}\rangle$")
# plt.ylabel(r"Eddington Flux ($gc/\kappa_{\rm F}$)")

# plt.legend(loc = 4)

# plt.tight_layout()

##############################################################################
    
## Opacity / L/M plot
##############################################################################

# time_list = BPASS_data.columns[BPASS_data.columns != 'WL']
# time_list_exp = np.power(10,time_list.astype(float))

# plt.figure(dpi=200)

# fig, ax = plt.subplots(1,2, dpi = 200, sharex=True, figsize=(10, 4))

# ax[0].plot(time_list_exp/10**6, L_Edd_DF.kappa_av_F_Sil*f_dg, 'b--', alpha = 0.3)
# ax[0].plot(time_list_exp/10**6, L_Edd_DF.kappa_av_F_Gra*f_dg, 'g--', alpha = 0.3)
# ax[0].plot(time_list_exp/10**6, L_Edd_DF.kappa_av_F_SiC*f_dg, 'r--', alpha = 0.3)

# ax[0].plot(time_list_exp/10**6, L_Edd_DF.kappa_av_RP_Sil*f_dg, 'b', alpha = 0.3, label = "Sil")
# ax[0].plot(time_list_exp/10**6, L_Edd_DF.kappa_av_RP_Gra*f_dg, 'g', alpha = 0.3, label = "Gra")
# ax[0].plot(time_list_exp/10**6, L_Edd_DF.kappa_av_RP_SiC*f_dg, 'r', alpha = 0.3, label = "SiC")

# ax[0].plot(time_list_exp/10**6, (L_Edd_DF.kappa_av_F_Sil.to_numpy()*grain_mix[0] + 
#                                   L_Edd_DF.kappa_av_F_Gra.to_numpy()*grain_mix[1] +
#                                   L_Edd_DF.kappa_av_F_SiC.to_numpy()*grain_mix[2])*f_dg, 'k--', label = r"$\langle \kappa_{\rm F}\rangle$ dust mix")

# ax[0].plot(time_list_exp/10**6, (L_Edd_DF.kappa_av_RP_Sil.to_numpy()*grain_mix[0] + 
#                                   L_Edd_DF.kappa_av_RP_Gra.to_numpy()*grain_mix[1] +
#                                   L_Edd_DF.kappa_av_RP_SiC.to_numpy()*grain_mix[2])*f_dg, 'k', label = r"$\langle \kappa_{\rm RP}\rangle$ dust mix")

# # ax[0].plot(time_list_exp/10**6, (DF.kappa_av_RP_Sil.to_numpy()*grain_mix[0] + 
# #                                   DF.kappa_av_RP_Gra.to_numpy()*grain_mix[1] +
# #                                   DF.kappa_av_RP_SiC.to_numpy()*grain_mix[2])*f_dg, c = 'g', label = r"$\langle \kappa_{\rm RP}\rangle$ CSFR dust mix")


# # ax[1].plot(time_list_exp/10**6, continuous_SFR_135_300_z_020.drop('WL', axis = 1).sum(axis = 0)/mass, 'k--', alpha = 0.3)
# # ax[1].plot(time_list_exp/10**6, BPASS_data_135_300_z_020.drop('WL', axis = 1).sum(axis = 0)/10**6, 'k', label = '135_300, z = 0.02', linewidth = 1, alpha = 0.7)
# # ax[1].plot(time_list_exp/10**6, continuous_SFR_135_100_z_020.drop('WL', axis = 1).sum(axis = 0)/mass, 'c--', alpha = 0.3)
# # ax[1].plot(time_list_exp/10**6, BPASS_data_135_100_z_020.drop('WL', axis = 1).sum(axis = 0)/10**6, 'c', label = '135_100, z = 0.02', linewidth = 1, alpha = 0.7)
# # ax[1].plot(time_list_exp/10**6, continuous_SFR_100_300_z_020.drop('WL', axis = 1).sum(axis = 0)/mass, 'g--', alpha = 0.3)
# # ax[1].plot(time_list_exp/10**6, BPASS_data_100_300_z_020.drop('WL', axis = 1).sum(axis = 0)/10**6, 'g', label = '100_300, z = 0.02', linewidth = 1, alpha = 0.7)
# # ax[1].plot(time_list_exp/10**6, continuous_SFR_100_300_z_010.drop('WL', axis = 1).sum(axis = 0)/mass, 'r--', alpha = 0.3)
# # ax[1].plot(time_list_exp/10**6, BPASS_data_100_300_z_010.drop('WL', axis = 1).sum(axis = 0)/10**6, 'r', label = '100_300, z = 0.01', linewidth = 1, alpha = 0.7)
# # ax[1].plot(time_list_exp/10**6, continuous_SFR_chab.drop('WL', axis = 1).sum(axis = 0)/mass, 'b--', alpha = 0.3)
# # ax[1].plot(time_list_exp/10**6, BPASS_data_chab.drop('WL', axis = 1).sum(axis = 0)/10**6, 'b', label = 'Chabrier, z = 0.02', linewidth = 1, alpha = 0.7)

# ax[1].plot(time_list_exp/10**6, 4*np.pi*G*c / ((L_Edd_DF.kappa_av_RP_Sil.to_numpy()*grain_mix[0] + 
#                                   L_Edd_DF.kappa_av_RP_Gra.to_numpy()*grain_mix[1] +
#                                   L_Edd_DF.kappa_av_RP_SiC.to_numpy()*grain_mix[2])*f_dg), c = 'k', linestyle = 'dashdot', linewidth = 2)

# # ax[1].plot(time_list_exp/10**6, 4*np.pi*G*c / ((DF.kappa_av_RP_Sil.to_numpy()*grain_mix[0] + 
# #                                   DF.kappa_av_RP_Gra.to_numpy()*grain_mix[1] +
# #                                   DF.kappa_av_RP_SiC.to_numpy()*grain_mix[2])*f_dg), c = 'g', linestyle = 'dashdot', linewidth = 2)


# # ax[1].text(2, 105, r"$\rm L_{\rm EDD}/M$", rotation = 7)
# # ax[1].text(1.5, 70, r"$\rm Instantaneous$", fontsize = 8, rotation = 7)
# # ax[1].text(2, 23, r"$\rm L_{\rm EDD}/M$", rotation = 7)
# # ax[1].text(1.5, 15, r"$\rm Continuous$", fontsize = 8, rotation = 7)
# # ax[1].text(4, 290, "Continuous Star Formation", rotation = -35)
# # ax[1].text(14, 1.1, "Instantaneous Star Formation", rotation = -35)

# ax[0].set_xscale('log')
# ax[1].set_xscale('log')
# ax[0].set_yscale('log')
# ax[1].set_yscale('log')

# # ax[0].set_ylim(40,1000)
# # ax[1].set_ylim(1,10**4)
# # ax[0].set_xlim(1,10**4)
# # ax[1].set_xlim(1,10**4)

# ax[0].legend()
# ax[1].legend()

# ax[0].set_xlabel(r'Time (Myrs)')
# ax[1].set_xlabel(r'Time (Myrs)')

# ax[0].set_ylabel(r'$\langle \kappa_{\rm RP} \rangle$ (${\rm cm}^2/$g)')
# ax[1].set_ylabel(r'L/M ($L_\odot/M_\odot$)')

# plt.tight_layout()

##############################################################################

## Opacity plots by grain sizes
##############################################################################

# grain_min = 0.001
# grain_max = 1
# gal_data, continuous_SFR, BPASS_data, Grain_data, kappa_data, L_Edd_DF_001_1, _ = PrepareGalaxyData(galaxy, time_slice, BPASS_file, h_g)
# grain_min = 0.001
# grain_max = 10
# _, _, _, _, _, L_Edd_DF_001_10, _, _ = PrepareGalaxyData(galaxy, time_slice, BPASS_file, h_g)
# grain_min = 0.01
# grain_max = 1
# _, _, _, _, _, L_Edd_DF_01_1, _, _ = PrepareGalaxyData(galaxy, time_slice, BPASS_file, h_g)
# grain_min = 0.001
# grain_max = 0.1
# _, _, _, _, _, L_Edd_DF_001_01, _, _ = PrepareGalaxyData(galaxy, time_slice, BPASS_file, h_g)
# grain_min = 0.01
# grain_max = 10
# _, _, _, _, _, L_Edd_DF_01_10, _, _ = PrepareGalaxyData(galaxy, time_slice, BPASS_file, h_g)

# time_list = BPASS_data.columns[BPASS_data.columns != 'WL']
# time_list_exp = np.power(10,time_list.astype(float))

# plt.figure(dpi = 200)

# plt.plot(time_list_exp, (L_Edd_DF_001_1.kappa_av_RP_Sil.to_numpy()*grain_mix[0] + 
#                                   L_Edd_DF_001_1.kappa_av_RP_Gra.to_numpy()*grain_mix[1] +
#                                   L_Edd_DF_001_1.kappa_av_RP_SiC.to_numpy()*grain_mix[2])*f_dg, 'k', label = r"$a_{\rm min} = 0.001, a_{\rm max} = 1$")

# plt.plot(time_list_exp, (L_Edd_DF_001_10.kappa_av_RP_Sil.to_numpy()*grain_mix[0] + 
#                                   L_Edd_DF_001_10.kappa_av_RP_Gra.to_numpy()*grain_mix[1] +
#                                   L_Edd_DF_001_10.kappa_av_RP_SiC.to_numpy()*grain_mix[2])*f_dg, label = r"$a_{\rm min} = 0.001, a_{\rm max} = 10$")

# plt.plot(time_list_exp, (L_Edd_DF_01_1.kappa_av_RP_Sil.to_numpy()*grain_mix[0] + 
#                                   L_Edd_DF_01_1.kappa_av_RP_Gra.to_numpy()*grain_mix[1] +
#                                   L_Edd_DF_01_1.kappa_av_RP_SiC.to_numpy()*grain_mix[2])*f_dg, label = r"$a_{\rm min} = 0.01, a_{\rm max} = 1$")

# plt.plot(time_list_exp, (L_Edd_DF_001_01.kappa_av_RP_Sil.to_numpy()*grain_mix[0] + 
#                                   L_Edd_DF_001_01.kappa_av_RP_Gra.to_numpy()*grain_mix[1] +
#                                   L_Edd_DF_001_01.kappa_av_RP_SiC.to_numpy()*grain_mix[2])*f_dg, label = r"$a_{\rm min} = 0.001, a_{\rm max} = 0.1$")

# plt.plot(time_list_exp, (L_Edd_DF_01_10.kappa_av_RP_Sil.to_numpy()*grain_mix[0] + 
#                                   L_Edd_DF_01_10.kappa_av_RP_Gra.to_numpy()*grain_mix[1] +
#                                   L_Edd_DF_01_10.kappa_av_RP_SiC.to_numpy()*grain_mix[2])*f_dg, label = r"$a_{\rm min} = 0.01, a_{\rm max} = 10$")

# plt.xscale('log')
# plt.yscale('log')

# plt.xlim(10**6,10**10)

# plt.legend()

# plt.xlabel(r'Time (yrs)')

# plt.ylabel(r'$\langle \kappa_{\rm RP} \rangle$ (${\rm cm}^2/$g)')

# plt.tight_layout()

#############################################################################

## Analytic ratio plot
##############################################################################

# _, _, ratio_rp_lambda_rp_3 = GetRatios(L_Edd_DF, Grain_data, '6.5', grain_type, wl_ref)
# _, _, ratio_rp_lambda_rp_10 = GetRatios(L_Edd_DF, Grain_data, '7.0', grain_type, wl_ref)
# _, _, ratio_rp_lambda_rp_100 = GetRatios(L_Edd_DF, Grain_data, '8.0', grain_type, wl_ref)

# directed_result = pd.read_csv('directed MC result.csv')

# plt.figure(dpi = 200)
# _, _, ratio_rp_lambda_rp = GetRatios(L_Edd_DF, Grain_data, '6.0', grain_type, wl_ref)
# plt.plot(MC_data.tau*ratio_rp_lambda_rp, MC_data['6.0']/(1-np.exp(-MC_data.tau*ratio_rp_lambda_rp)), label = '1 Myr')
# plt.plot(MC_data.tau*ratio_rp_lambda_rp_3, MC_data['6.5']/(1-np.exp(-MC_data.tau*ratio_rp_lambda_rp_3)), label = '3 Myr')
# plt.plot(MC_data.tau*ratio_rp_lambda_rp_10, MC_data['7.0']/(1-np.exp(-MC_data.tau*ratio_rp_lambda_rp_10)), label = '10 Myr')
# # plt.plot(MC_data.tau*ratio_rp_lambda_rp_100, MC_data['8.0']/(1-np.exp(-MC_data.tau*ratio_rp_lambda_rp_100)), label = '100 Myr')
# _, _, ratio_rp_lambda_rp = GetRatios(L_Edd_DF, Grain_data, '6.0', grain_type, wl_ref)
# plt.plot(directed_result.tau*ratio_rp_lambda_rp, directed_result['6.0']/(1-np.exp(-directed_result.tau*ratio_rp_lambda_rp)), 'purple', label = 'Beamed 1 Myr')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(10**-2,100)
# #plt.ylim(10**-2,2)
# plt.xlabel(r'$\langle \tau_{\rm RP} \rangle$')
# plt.ylabel('Fractional Momentum Transfer / Analytic Result')
# plt.grid(which='major', alpha = 0.4, color = 'k')
# plt.grid(which="minor", alpha = 0.8, linestyle=":")
# plt.legend()
# plt.tight_layout()

##############################################################################

### Galaxy map with Eddington ratio
##############################################################################

# import matplotlib

# plt.figure(dpi = 200)

# norm = matplotlib.colors.TwoSlopeNorm(1,vmin = min(gal_data.L_Bol / gal_data.L_Edd), vmax = max(gal_data.L_Bol / gal_data.L_Edd))
# cmap = "jet"

# plt.scatter(gal_data.xcenter,gal_data.ycenter, s = 1, c = gal_data.L_Bol / gal_data.L_Edd, norm = norm, cmap = cmap)
# plt.axis('off')
# plt.colorbar()
# plt.title("Eddington ratios for subregions of M51")

##############################################################################

### Galaxy Histograms - Planar
##############################################################################

# # M51 data
# gal_data_M51_6_5, continuous_SFR, _, _, _, _, _, _ = PrepareGalaxyData("M51", 6.0, BPASS_file, 5)
# gal_data_M51_6_10, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 6.0, BPASS_file, 10)
# gal_data_M51_6_20, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 6.0, BPASS_file, 20)
# gal_data_M51_6_40, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 6.0, BPASS_file, 40)

# gal_data_M51_7_5, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 7.0, BPASS_file, 5)
# gal_data_M51_7_10, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 7.0, BPASS_file, 10)
# gal_data_M51_7_20, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 7.0, BPASS_file, 20)
# gal_data_M51_7_40, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 7.0, BPASS_file, 40)

# # NGC6946 data
# gal_data_NGC6946_6_5, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 6.0, BPASS_file, 5)
# gal_data_NGC6946_6_10, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 6.0, BPASS_file, 10)
# gal_data_NGC6946_6_20, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 6.0, BPASS_file, 20)
# gal_data_NGC6946_6_40, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 6.0, BPASS_file, 40)

# gal_data_NGC6946_7_5, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 7.0, BPASS_file, 5)
# gal_data_NGC6946_7_10, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 7.0, BPASS_file, 10)
# gal_data_NGC6946_7_20, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 7.0, BPASS_file, 20)
# gal_data_NGC6946_7_40, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 7.0, BPASS_file, 40)

# fig, ax = plt.subplots(nrows = 2, ncols = 2, dpi = 200, sharey=True, figsize=(10,7))

# bins = np.logspace(-3,2,200)

# ymin = 1
# ymax = 100

# # ax[0,0].hist(gal_data_M51_6_5.F_Bol / gal_data_M51_6_5.F_Edd, bins = bins, alpha = 0.5, label = "5 pc")
# ax[0,0].hist(gal_data_M51_6_10.F_Bol / gal_data_M51_6_10.F_Edd, bins = bins, alpha = 0.5, label = "10 pc", color='C1')
# # ax[0,0].hist(gal_data_M51_6_20.F_Bol / gal_data_M51_6_20.F_Edd, bins = bins, alpha = 0.5, label = "20 pc")
# ax[0,0].hist(gal_data_M51_6_40.F_Bol / gal_data_M51_6_40.F_Edd, bins = bins, alpha = 0.5, label = "40 pc", color='C3')

# ax[0,0].plot([1, 1], [ymin, ymax], 'k', alpha = 0.5)

# # ax[0,1].hist(gal_data_M51_7_5.F_Bol / gal_data_M51_7_5.F_Edd, bins = bins, alpha = 0.5, label = "5 pc")
# ax[0,1].hist(gal_data_M51_7_10.F_Bol / gal_data_M51_7_10.F_Edd, bins = bins, alpha = 0.5, label = "10 pc", color='C1')
# # ax[0,1].hist(gal_data_M51_7_20.F_Bol / gal_data_M51_7_20.F_Edd, bins = bins, alpha = 0.5, label = "20 pc")
# ax[0,1].hist(gal_data_M51_7_40.F_Bol / gal_data_M51_7_40.F_Edd, bins = bins, alpha = 0.5, label = "40 pc", color='C3')

# ax[0,1].plot([1, 1], [ymin, ymax], 'k', alpha = 0.5)

# # ax[1,0].hist(gal_data_NGC6946_6_5.F_Bol / gal_data_NGC6946_6_5.F_Edd, bins = bins, alpha = 0.5, label = "5 pc")
# ax[1,0].hist(gal_data_NGC6946_6_10.F_Bol / gal_data_NGC6946_6_10.F_Edd, bins = bins, alpha = 0.5, label = "10 pc", color='C1')
# # ax[1,0].hist(gal_data_NGC6946_6_20.F_Bol / gal_data_NGC6946_6_20.F_Edd, bins = bins, alpha = 0.5, label = "20 pc")
# ax[1,0].hist(gal_data_NGC6946_6_40.F_Bol / gal_data_NGC6946_6_40.F_Edd, bins = bins, alpha = 0.5, label = "40 pc", color='C3')

# ax[1,0].plot([1, 1], [ymin, ymax], 'k', alpha = 0.5)

# # ax[1,1].hist(gal_data_NGC6946_7_5.F_Bol / gal_data_NGC6946_7_5.F_Edd, bins = bins, alpha = 0.5, label = "5 pc")
# ax[1,1].hist(gal_data_NGC6946_7_10.F_Bol / gal_data_NGC6946_7_10.F_Edd, bins = bins, alpha = 0.5, label = "10 pc", color='C1')
# # ax[1,1].hist(gal_data_NGC6946_7_20.F_Bol / gal_data_NGC6946_7_20.F_Edd, bins = bins, alpha = 0.5, label = "20 pc")
# ax[1,1].hist(gal_data_NGC6946_7_40.F_Bol / gal_data_NGC6946_7_40.F_Edd, bins = bins, alpha = 0.5, label = "40 pc", color='C3')

# ax[1,1].plot([1, 1], [ymin, ymax], 'k', alpha = 0.5)

# ax[0,0].set_xscale('log')
# ax[0,1].set_xscale('log')
# ax[1,0].set_xscale('log')
# ax[1,1].set_xscale('log')
# ax[0,0].set_yscale('log')
# ax[0,1].set_yscale('log')
# ax[1,0].set_yscale('log')
# ax[1,1].set_yscale('log')

# ax[0,0].set_ylim(ymin,ymax)
# ax[0,1].set_ylim(ymin,ymax)
# ax[1,0].set_ylim(ymin,ymax)
# ax[1,1].set_ylim(ymin,ymax)
# ax[0,0].set_xlim(0.001)
# ax[0,1].set_xlim(0.001,10)
# ax[1,0].set_xlim(0.001)
# ax[1,1].set_xlim(0.001,10)

# ax[1,0].legend()

# ax[1,0].set_xlabel(r'Eddington Ratio')
# ax[1,1].set_xlabel(r'Eddington Ratio')
# ax[0,0].set_ylabel(r'Count')
# ax[1,0].set_ylabel(r'Count')

# ax[0,0].text(0.05, 0.9, r'NGC 5194 (1 Myr)', fontsize=12, transform=ax[0,0].transAxes)
# ax[0,1].text(0.05, 0.9, r'NGC 5194 (10 Myr)', fontsize=12, transform=ax[0,1].transAxes)
# ax[1,0].text(0.05, 0.9, r'NGC 6946 (1 Myr)', fontsize=12, transform=ax[1,0].transAxes)
# ax[1,1].text(0.05, 0.9, r'NGC 6946 (10 Myr)', fontsize=12, transform=ax[1,1].transAxes)

# plt.tight_layout()

##############################################################################


### Galaxy Histogram - Spherical
##############################################################################

# M51 data
gal_data_M51_6_5, continuous_SFR, _, _, _, _, _, _ = PrepareGalaxyData("M51", 6.0, BPASS_file, 5)
gal_data_M51_6_10, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 6.0, BPASS_file, 10)
gal_data_M51_6_20, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 6.0, BPASS_file, 20)
gal_data_M51_6_40, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 6.0, BPASS_file, 40)

gal_data_M51_7_5, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 7.0, BPASS_file, 5)
gal_data_M51_7_10, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 7.0, BPASS_file, 10)
gal_data_M51_7_20, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 7.0, BPASS_file, 20)
gal_data_M51_7_40, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 7.0, BPASS_file, 40)

# NGC6946 data
gal_data_NGC6946_6_5, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 6.0, BPASS_file, 5)
gal_data_NGC6946_6_10, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 6.0, BPASS_file, 10)
gal_data_NGC6946_6_20, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 6.0, BPASS_file, 20)
gal_data_NGC6946_6_40, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 6.0, BPASS_file, 40)

gal_data_NGC6946_7_5, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 7.0, BPASS_file, 5)
gal_data_NGC6946_7_10, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 7.0, BPASS_file, 10)
gal_data_NGC6946_7_20, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 7.0, BPASS_file, 20)
gal_data_NGC6946_7_40, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 7.0, BPASS_file, 40)

fig, ax = plt.subplots(nrows = 2, ncols = 2, dpi = 200, sharey=True, figsize=(10,7))

bins = np.logspace(-3,2,200)

ymin = 1
ymax = 100

ax[0,0].hist(gal_data_M51_6_5.L_Bol / gal_data_M51_6_5.L_Edd, bins = bins, alpha = 0.5, label = "5 pc")
ax[0,0].hist(gal_data_M51_6_10.L_Bol / gal_data_M51_6_10.L_Edd, bins = bins, alpha = 0.5, label = "10 pc")
ax[0,0].hist(gal_data_M51_6_20.L_Bol / gal_data_M51_6_20.L_Edd, bins = bins, alpha = 0.5, label = "20 pc")
ax[0,0].hist(gal_data_M51_6_40.L_Bol / gal_data_M51_6_40.L_Edd, bins = bins, alpha = 0.5, label = "40 pc")

ax[0,0].plot([1, 1], [ymin, ymax], 'k', alpha = 0.5)

ax[0,1].hist(gal_data_M51_7_5.L_Bol / gal_data_M51_7_5.L_Edd, bins = bins, alpha = 0.5, label = "5 pc")
ax[0,1].hist(gal_data_M51_7_10.L_Bol / gal_data_M51_7_10.L_Edd, bins = bins, alpha = 0.5, label = "10 pc")
ax[0,1].hist(gal_data_M51_7_20.L_Bol / gal_data_M51_7_20.L_Edd, bins = bins, alpha = 0.5, label = "20 pc")
ax[0,1].hist(gal_data_M51_7_40.L_Bol / gal_data_M51_7_40.L_Edd, bins = bins, alpha = 0.5, label = "40 pc")

ax[0,1].plot([1, 1], [ymin, ymax], 'k', alpha = 0.5)

ax[1,0].hist(gal_data_NGC6946_6_5.L_Bol / gal_data_NGC6946_6_5.L_Edd, bins = bins, alpha = 0.5, label = "5 pc")
ax[1,0].hist(gal_data_NGC6946_6_10.L_Bol / gal_data_NGC6946_6_10.L_Edd, bins = bins, alpha = 0.5, label = "10 pc")
ax[1,0].hist(gal_data_NGC6946_6_20.L_Bol / gal_data_NGC6946_6_20.L_Edd, bins = bins, alpha = 0.5, label = "20 pc")
ax[1,0].hist(gal_data_NGC6946_6_40.L_Bol / gal_data_NGC6946_6_40.L_Edd, bins = bins, alpha = 0.5, label = "40 pc")

ax[1,0].plot([1, 1], [ymin, ymax], 'k', alpha = 0.5)

ax[1,1].hist(gal_data_NGC6946_7_5.L_Bol / gal_data_NGC6946_7_5.L_Edd, bins = bins, alpha = 0.5, label = "5 pc")
ax[1,1].hist(gal_data_NGC6946_7_10.L_Bol / gal_data_NGC6946_7_10.L_Edd, bins = bins, alpha = 0.5, label = "10 pc")
ax[1,1].hist(gal_data_NGC6946_7_20.L_Bol / gal_data_NGC6946_7_20.L_Edd, bins = bins, alpha = 0.5, label = "20 pc")
ax[1,1].hist(gal_data_NGC6946_7_40.L_Bol / gal_data_NGC6946_7_40.L_Edd, bins = bins, alpha = 0.5, label = "40 pc")

ax[1,1].plot([1, 1], [ymin, ymax], 'k', alpha = 0.5)

ax[0,0].set_xscale('log')
ax[0,1].set_xscale('log')
ax[1,0].set_xscale('log')
ax[1,1].set_xscale('log')
ax[0,0].set_yscale('log')
ax[0,1].set_yscale('log')
ax[1,0].set_yscale('log')
ax[1,1].set_yscale('log')

ax[0,0].set_ylim(ymin,ymax)
ax[0,1].set_ylim(ymin,ymax)
ax[1,0].set_ylim(ymin,ymax)
ax[1,1].set_ylim(ymin,ymax)
ax[0,0].set_xlim(0.001)
ax[0,1].set_xlim(0.001,10)
ax[1,0].set_xlim(0.001)
ax[1,1].set_xlim(0.001,10)

ax[1,0].legend()

ax[1,0].set_xlabel(r'Eddington Ratio')
ax[1,1].set_xlabel(r'Eddington Ratio')
ax[0,0].set_ylabel(r'Count')
ax[1,0].set_ylabel(r'Count')

ax[0,0].text(0.05, 0.9, r'NGC 5194 (1 Myr)', fontsize=12, transform=ax[0,0].transAxes)
ax[0,1].text(0.05, 0.9, r'NGC 5194 (10 Myr)', fontsize=12, transform=ax[0,1].transAxes)
ax[1,0].text(0.05, 0.9, r'NGC 6946 (1 Myr)', fontsize=12, transform=ax[1,0].transAxes)
ax[1,1].text(0.05, 0.9, r'NGC 6946 (10 Myr)', fontsize=12, transform=ax[1,1].transAxes)

plt.tight_layout()

##############################################################################

## Get integral quantities, for use with the above histogram block
#############################################################################

super_Edd_M51_6_5 = gal_data_M51_6_5.L_Bol > gal_data_M51_6_5.L_Edd
super_Edd_M51_6_10 = gal_data_M51_6_10.L_Bol > gal_data_M51_6_10.L_Edd
super_Edd_M51_6_20 = gal_data_M51_6_20.L_Bol > gal_data_M51_6_20.L_Edd
super_Edd_M51_6_40 = gal_data_M51_6_40.L_Bol > gal_data_M51_6_40.L_Edd

super_Edd_M51_7_5 = gal_data_M51_7_5.L_Bol > gal_data_M51_7_5.L_Edd
super_Edd_M51_7_10 = gal_data_M51_7_10.L_Bol > gal_data_M51_7_10.L_Edd
super_Edd_M51_7_20 = gal_data_M51_7_20.L_Bol > gal_data_M51_7_20.L_Edd
super_Edd_M51_7_40 = gal_data_M51_7_40.L_Bol > gal_data_M51_7_40.L_Edd

super_Edd_NGC6946_6_5 = gal_data_NGC6946_6_5.L_Bol > gal_data_NGC6946_6_5.L_Edd
super_Edd_NGC6946_6_10 = gal_data_NGC6946_6_10.L_Bol > gal_data_NGC6946_6_10.L_Edd
super_Edd_NGC6946_6_20 = gal_data_NGC6946_6_20.L_Bol > gal_data_NGC6946_6_20.L_Edd
super_Edd_NGC6946_6_40 = gal_data_NGC6946_6_40.L_Bol > gal_data_NGC6946_6_40.L_Edd

super_Edd_NGC6946_7_5 = gal_data_NGC6946_7_5.L_Bol > gal_data_NGC6946_7_5.L_Edd
super_Edd_NGC6946_7_10 = gal_data_NGC6946_7_10.L_Bol > gal_data_NGC6946_7_10.L_Edd
super_Edd_NGC6946_7_20 = gal_data_NGC6946_7_20.L_Bol > gal_data_NGC6946_7_20.L_Edd
super_Edd_NGC6946_7_40 = gal_data_NGC6946_7_40.L_Bol > gal_data_NGC6946_7_40.L_Edd

M51_6_5_LBol_frac = gal_data_M51_6_5[super_Edd_M51_6_5].L_Bol.sum()/gal_data_M51_6_5.L_Bol.sum()
M51_6_10_LBol_frac = gal_data_M51_6_10[super_Edd_M51_6_10].L_Bol.sum()/gal_data_M51_6_10.L_Bol.sum()
M51_6_20_LBol_frac = gal_data_M51_6_20[super_Edd_M51_6_20].L_Bol.sum()/gal_data_M51_6_20.L_Bol.sum()
M51_6_40_LBol_frac = gal_data_M51_6_40[super_Edd_M51_6_40].L_Bol.sum()/gal_data_M51_6_40.L_Bol.sum()

M51_7_5_LBol_frac = gal_data_M51_7_5[super_Edd_M51_7_5].L_Bol.sum()/gal_data_M51_7_5.L_Bol.sum()
M51_7_10_LBol_frac = gal_data_M51_7_10[super_Edd_M51_7_10].L_Bol.sum()/gal_data_M51_7_10.L_Bol.sum()
M51_7_20_LBol_frac = gal_data_M51_7_20[super_Edd_M51_7_20].L_Bol.sum()/gal_data_M51_7_20.L_Bol.sum()
M51_7_40_LBol_frac = gal_data_M51_7_40[super_Edd_M51_7_40].L_Bol.sum()/gal_data_M51_7_40.L_Bol.sum()

NGC6946_6_5_LBol_frac = gal_data_NGC6946_6_5[super_Edd_NGC6946_6_5].L_Bol.sum()/gal_data_NGC6946_6_5.L_Bol.sum()
NGC6946_6_10_LBol_frac = gal_data_NGC6946_6_10[super_Edd_NGC6946_6_10].L_Bol.sum()/gal_data_NGC6946_6_10.L_Bol.sum()
NGC6946_6_20_LBol_frac = gal_data_NGC6946_6_20[super_Edd_NGC6946_6_20].L_Bol.sum()/gal_data_NGC6946_6_20.L_Bol.sum()
NGC6946_6_40_LBol_frac = gal_data_NGC6946_6_40[super_Edd_NGC6946_6_40].L_Bol.sum()/gal_data_NGC6946_6_40.L_Bol.sum()

NGC6946_7_5_LBol_frac = gal_data_NGC6946_7_5[super_Edd_NGC6946_7_5].L_Bol.sum()/gal_data_NGC6946_7_5.L_Bol.sum()
NGC6946_7_10_LBol_frac = gal_data_NGC6946_7_10[super_Edd_NGC6946_7_10].L_Bol.sum()/gal_data_NGC6946_7_10.L_Bol.sum()
NGC6946_7_20_LBol_frac = gal_data_NGC6946_7_20[super_Edd_NGC6946_7_20].L_Bol.sum()/gal_data_NGC6946_7_20.L_Bol.sum()
NGC6946_7_40_LBol_frac = gal_data_NGC6946_7_40[super_Edd_NGC6946_7_40].L_Bol.sum()/gal_data_NGC6946_7_40.L_Bol.sum()

M51_6_5_Mass_g_frac = gal_data_M51_6_5[super_Edd_M51_6_5].Mass_g.sum()/gal_data_M51_6_5.Mass_g.sum()
M51_6_10_Mass_g_frac = gal_data_M51_6_10[super_Edd_M51_6_10].Mass_g.sum()/gal_data_M51_6_10.Mass_g.sum()
M51_6_20_Mass_g_frac = gal_data_M51_6_20[super_Edd_M51_6_20].Mass_g.sum()/gal_data_M51_6_20.Mass_g.sum()
M51_6_40_Mass_g_frac = gal_data_M51_6_40[super_Edd_M51_6_40].Mass_g.sum()/gal_data_M51_6_40.Mass_g.sum()

M51_7_5_Mass_g_frac = gal_data_M51_7_5[super_Edd_M51_7_5].Mass_g.sum()/gal_data_M51_7_5.Mass_g.sum()
M51_7_10_Mass_g_frac = gal_data_M51_7_10[super_Edd_M51_7_10].Mass_g.sum()/gal_data_M51_7_10.Mass_g.sum()
M51_7_20_Mass_g_frac = gal_data_M51_7_20[super_Edd_M51_7_20].Mass_g.sum()/gal_data_M51_7_20.Mass_g.sum()
M51_7_40_Mass_g_frac = gal_data_M51_7_40[super_Edd_M51_7_40].Mass_g.sum()/gal_data_M51_7_40.Mass_g.sum()

NGC6946_6_5_Mass_g_frac = gal_data_NGC6946_6_5[super_Edd_NGC6946_6_5].Mass_g.sum()/gal_data_NGC6946_6_5.Mass_g.sum()
NGC6946_6_10_Mass_g_frac = gal_data_NGC6946_6_10[super_Edd_NGC6946_6_10].Mass_g.sum()/gal_data_NGC6946_6_10.Mass_g.sum()
NGC6946_6_20_Mass_g_frac = gal_data_NGC6946_6_20[super_Edd_NGC6946_6_20].Mass_g.sum()/gal_data_NGC6946_6_20.Mass_g.sum()
NGC6946_6_40_Mass_g_frac = gal_data_NGC6946_6_40[super_Edd_NGC6946_6_40].Mass_g.sum()/gal_data_NGC6946_6_40.Mass_g.sum()

NGC6946_7_5_Mass_g_frac = gal_data_NGC6946_7_5[super_Edd_NGC6946_7_5].Mass_g.sum()/gal_data_NGC6946_7_5.Mass_g.sum()
NGC6946_7_10_Mass_g_frac = gal_data_NGC6946_7_10[super_Edd_NGC6946_7_10].Mass_g.sum()/gal_data_NGC6946_7_10.Mass_g.sum()
NGC6946_7_20_Mass_g_frac = gal_data_NGC6946_7_20[super_Edd_NGC6946_7_20].Mass_g.sum()/gal_data_NGC6946_7_20.Mass_g.sum()
NGC6946_7_40_Mass_g_frac = gal_data_NGC6946_7_40[super_Edd_NGC6946_7_40].Mass_g.sum()/gal_data_NGC6946_7_40.Mass_g.sum()

M51_6_5_Mass_old_frac = gal_data_M51_6_5[super_Edd_M51_6_5].Mass_old.sum()/gal_data_M51_6_5.Mass_old.sum()
M51_6_10_Mass_old_frac = gal_data_M51_6_10[super_Edd_M51_6_10].Mass_old.sum()/gal_data_M51_6_10.Mass_old.sum()
M51_6_20_Mass_old_frac = gal_data_M51_6_20[super_Edd_M51_6_20].Mass_old.sum()/gal_data_M51_6_20.Mass_old.sum()
M51_6_40_Mass_old_frac = gal_data_M51_6_40[super_Edd_M51_6_40].Mass_old.sum()/gal_data_M51_6_40.Mass_old.sum()

M51_7_5_Mass_old_frac = gal_data_M51_7_5[super_Edd_M51_7_5].Mass_old.sum()/gal_data_M51_7_5.Mass_old.sum()
M51_7_10_Mass_old_frac = gal_data_M51_7_10[super_Edd_M51_7_10].Mass_old.sum()/gal_data_M51_7_10.Mass_old.sum()
M51_7_20_Mass_old_frac = gal_data_M51_7_20[super_Edd_M51_7_20].Mass_old.sum()/gal_data_M51_7_20.Mass_old.sum()
M51_7_40_Mass_old_frac = gal_data_M51_7_40[super_Edd_M51_7_40].Mass_old.sum()/gal_data_M51_7_40.Mass_old.sum()

NGC6946_6_5_Mass_old_frac = gal_data_NGC6946_6_5[super_Edd_NGC6946_6_5].Mass_old.sum()/gal_data_NGC6946_6_5.Mass_old.sum()
NGC6946_6_10_Mass_old_frac = gal_data_NGC6946_6_10[super_Edd_NGC6946_6_10].Mass_old.sum()/gal_data_NGC6946_6_10.Mass_old.sum()
NGC6946_6_20_Mass_old_frac = gal_data_NGC6946_6_20[super_Edd_NGC6946_6_20].Mass_old.sum()/gal_data_NGC6946_6_20.Mass_old.sum()
NGC6946_6_40_Mass_old_frac = gal_data_NGC6946_6_40[super_Edd_NGC6946_6_40].Mass_old.sum()/gal_data_NGC6946_6_40.Mass_old.sum()

NGC6946_7_5_Mass_old_frac = gal_data_NGC6946_7_5[super_Edd_NGC6946_7_5].Mass_old.sum()/gal_data_NGC6946_7_5.Mass_old.sum()
NGC6946_7_10_Mass_old_frac = gal_data_NGC6946_7_10[super_Edd_NGC6946_7_10].Mass_old.sum()/gal_data_NGC6946_7_10.Mass_old.sum()
NGC6946_7_20_Mass_old_frac = gal_data_NGC6946_7_20[super_Edd_NGC6946_7_20].Mass_old.sum()/gal_data_NGC6946_7_20.Mass_old.sum()
NGC6946_7_40_Mass_old_frac = gal_data_NGC6946_7_40[super_Edd_NGC6946_7_40].Mass_old.sum()/gal_data_NGC6946_7_40.Mass_old.sum()

print("M51 1 Myr 5 PC")
print(f"LBol frac: {M51_6_5_LBol_frac:.8f}")
print(f"Mgas frac: {M51_6_5_Mass_g_frac:.8f}")
print(f"Mold frac: {M51_6_5_Mass_old_frac:.8f}")
print("----------")
print("M51 1 Myr 10 PC")
print(f"LBol frac: {M51_6_10_LBol_frac:.8f}")
print(f"Mgas frac: {M51_6_10_Mass_g_frac:.8f}")
print(f"Mold frac: {M51_6_10_Mass_old_frac:.8f}")
print("----------")
print("M51 1 Myr 20 PC")
print(f"LBol frac: {M51_6_20_LBol_frac:.8f}")
print(f"Mgas frac: {M51_6_20_Mass_g_frac:.8f}")
print(f"Mold frac: {M51_6_20_Mass_old_frac:.8f}")
print("----------")
print("M51 1 Myr 40 PC")
print(f"LBol frac: {M51_6_40_LBol_frac:.8f}")
print(f"Mgas frac: {M51_6_40_Mass_g_frac:.8f}")
print(f"Mold frac: {M51_6_40_Mass_old_frac:.8f}")

print("------------------------------")

print("M51 10 Myr 5 PC")
print(f"LBol frac: {M51_7_5_LBol_frac:.8f}")
print(f"Mgas frac: {M51_7_5_Mass_g_frac:.8f}")
print(f"Mold frac: {M51_7_5_Mass_old_frac:.8f}")
print("----------")
print("M51 10 Myr 10 PC")
print(f"LBol frac: {M51_7_10_LBol_frac:.8f}")
print(f"Mgas frac: {M51_7_10_Mass_g_frac:.8f}")
print(f"Mold frac: {M51_7_10_Mass_old_frac:.8f}")
print("----------")
print("M51 10 Myr 20 PC")
print(f"LBol frac: {M51_7_20_LBol_frac:.8f}")
print(f"Mgas frac: {M51_7_20_Mass_g_frac:.8f}")
print(f"Mold frac: {M51_7_20_Mass_old_frac:.8f}")
print("----------")
print("M51 10 Myr 40 PC")
print(f"LBol frac: {M51_7_40_LBol_frac:.8f}")
print(f"Mgas frac: {M51_7_40_Mass_g_frac:.8f}")
print(f"Mold frac: {M51_7_40_Mass_old_frac:.8f}")

print("------------------------------")

print("NGC6946 1 Myr 5 PC")
print(f"LBol frac: {NGC6946_6_5_LBol_frac:.8f}")
print(f"Mgas frac: {NGC6946_6_5_Mass_g_frac:.8f}")
print(f"Mold frac: {NGC6946_6_5_Mass_old_frac:.8f}")
print("----------")
print("NGC6946 1 Myr 10 PC")
print(f"LBol frac: {NGC6946_6_10_LBol_frac:.8f}")
print(f"Mgas frac: {NGC6946_6_10_Mass_g_frac:.8f}")
print(f"Mold frac: {NGC6946_6_10_Mass_old_frac:.8f}")
print("----------")
print("NGC6946 1 Myr 20 PC")
print(f"LBol frac: {NGC6946_6_20_LBol_frac:.8f}")
print(f"Mgas frac: {NGC6946_6_20_Mass_g_frac:.8f}")
print(f"Mold frac: {NGC6946_6_20_Mass_old_frac:.8f}")
print("----------")
print("NGC6946 1 Myr 40 PC")
print(f"LBol frac: {NGC6946_6_40_LBol_frac:.8f}")
print(f"Mgas frac: {NGC6946_6_40_Mass_g_frac:.8f}")
print(f"Mold frac: {NGC6946_6_40_Mass_old_frac:.8f}")

print("------------------------------")

print("NGC6946 10 Myr 5 PC")
print(f"LBol frac: {NGC6946_7_5_LBol_frac:.8f}")
print(f"Mgas frac: {NGC6946_7_5_Mass_g_frac:.8f}")
print(f"Mold frac: {NGC6946_7_5_Mass_old_frac:.8f}")
print("----------")
print("NGC6946 10 Myr 10 PC")
print(f"LBol frac: {NGC6946_7_10_LBol_frac:.8f}")
print(f"Mgas frac: {NGC6946_7_10_Mass_g_frac:.8f}")
print(f"Mold frac: {NGC6946_7_10_Mass_old_frac:.8f}")
print("----------")
print("NGC6946 10 Myr 20 PC")
print(f"LBol frac: {NGC6946_7_20_LBol_frac:.8f}")
print(f"Mgas frac: {NGC6946_7_20_Mass_g_frac:.8f}")
print(f"Mold frac: {NGC6946_7_20_Mass_old_frac:.8f}")
print("----------")
print("NGC6946 10 Myr 40 PC")
print(f"LBol frac: {NGC6946_7_40_LBol_frac:.8f}")
print(f"Mgas frac: {NGC6946_7_40_Mass_g_frac:.8f}")
print(f"Mold frac: {NGC6946_7_40_Mass_old_frac:.8f}")

##############################################################################


## Integral Quantities planar
##############################################################################

# super_Edd_M51_6_5 = gal_data_M51_6_5.F_Bol > gal_data_M51_6_5.F_Edd
# super_Edd_M51_6_10 = gal_data_M51_6_10.F_Bol > gal_data_M51_6_10.F_Edd
# super_Edd_M51_6_20 = gal_data_M51_6_20.F_Bol > gal_data_M51_6_20.F_Edd
# super_Edd_M51_6_40 = gal_data_M51_6_40.F_Bol > gal_data_M51_6_40.F_Edd

# super_Edd_M51_7_5 = gal_data_M51_7_5.F_Bol > gal_data_M51_7_5.F_Edd
# super_Edd_M51_7_10 = gal_data_M51_7_10.F_Bol > gal_data_M51_7_10.F_Edd
# super_Edd_M51_7_20 = gal_data_M51_7_20.F_Bol > gal_data_M51_7_20.F_Edd
# super_Edd_M51_7_40 = gal_data_M51_7_40.F_Bol > gal_data_M51_7_40.F_Edd

# super_Edd_NGC6946_6_5 = gal_data_NGC6946_6_5.F_Bol > gal_data_NGC6946_6_5.F_Edd
# super_Edd_NGC6946_6_10 = gal_data_NGC6946_6_10.F_Bol > gal_data_NGC6946_6_10.F_Edd
# super_Edd_NGC6946_6_20 = gal_data_NGC6946_6_20.F_Bol > gal_data_NGC6946_6_20.F_Edd
# super_Edd_NGC6946_6_40 = gal_data_NGC6946_6_40.F_Bol > gal_data_NGC6946_6_40.F_Edd

# super_Edd_NGC6946_7_5 = gal_data_NGC6946_7_5.F_Bol > gal_data_NGC6946_7_5.F_Edd
# super_Edd_NGC6946_7_10 = gal_data_NGC6946_7_10.F_Bol > gal_data_NGC6946_7_10.F_Edd
# super_Edd_NGC6946_7_20 = gal_data_NGC6946_7_20.F_Bol > gal_data_NGC6946_7_20.F_Edd
# super_Edd_NGC6946_7_40 = gal_data_NGC6946_7_40.F_Bol > gal_data_NGC6946_7_40.F_Edd

# M51_6_5_FBol_frac = gal_data_M51_6_5[super_Edd_M51_6_5].F_Bol.sum()/gal_data_M51_6_5.F_Bol.sum()
# M51_6_10_FBol_frac = gal_data_M51_6_10[super_Edd_M51_6_10].F_Bol.sum()/gal_data_M51_6_10.F_Bol.sum()
# M51_6_20_FBol_frac = gal_data_M51_6_20[super_Edd_M51_6_20].F_Bol.sum()/gal_data_M51_6_20.F_Bol.sum()
# M51_6_40_FBol_frac = gal_data_M51_6_40[super_Edd_M51_6_40].F_Bol.sum()/gal_data_M51_6_40.F_Bol.sum()

# M51_7_5_FBol_frac = gal_data_M51_7_5[super_Edd_M51_7_5].F_Bol.sum()/gal_data_M51_7_5.F_Bol.sum()
# M51_7_10_FBol_frac = gal_data_M51_7_10[super_Edd_M51_7_10].F_Bol.sum()/gal_data_M51_7_10.F_Bol.sum()
# M51_7_20_FBol_frac = gal_data_M51_7_20[super_Edd_M51_7_20].F_Bol.sum()/gal_data_M51_7_20.F_Bol.sum()
# M51_7_40_FBol_frac = gal_data_M51_7_40[super_Edd_M51_7_40].F_Bol.sum()/gal_data_M51_7_40.F_Bol.sum()

# NGC6946_6_5_FBol_frac = gal_data_NGC6946_6_5[super_Edd_NGC6946_6_5].F_Bol.sum()/gal_data_NGC6946_6_5.F_Bol.sum()
# NGC6946_6_10_FBol_frac = gal_data_NGC6946_6_10[super_Edd_NGC6946_6_10].F_Bol.sum()/gal_data_NGC6946_6_10.F_Bol.sum()
# NGC6946_6_20_FBol_frac = gal_data_NGC6946_6_20[super_Edd_NGC6946_6_20].F_Bol.sum()/gal_data_NGC6946_6_20.F_Bol.sum()
# NGC6946_6_40_FBol_frac = gal_data_NGC6946_6_40[super_Edd_NGC6946_6_40].F_Bol.sum()/gal_data_NGC6946_6_40.F_Bol.sum()

# NGC6946_7_5_FBol_frac = gal_data_NGC6946_7_5[super_Edd_NGC6946_7_5].F_Bol.sum()/gal_data_NGC6946_7_5.F_Bol.sum()
# NGC6946_7_10_FBol_frac = gal_data_NGC6946_7_10[super_Edd_NGC6946_7_10].F_Bol.sum()/gal_data_NGC6946_7_10.F_Bol.sum()
# NGC6946_7_20_FBol_frac = gal_data_NGC6946_7_20[super_Edd_NGC6946_7_20].F_Bol.sum()/gal_data_NGC6946_7_20.F_Bol.sum()
# NGC6946_7_40_FBol_frac = gal_data_NGC6946_7_40[super_Edd_NGC6946_7_40].F_Bol.sum()/gal_data_NGC6946_7_40.F_Bol.sum()

# M51_6_5_Mass_g_frac = gal_data_M51_6_5[super_Edd_M51_6_5].Mass_g.sum()/gal_data_M51_6_5.Mass_g.sum()
# M51_6_10_Mass_g_frac = gal_data_M51_6_10[super_Edd_M51_6_10].Mass_g.sum()/gal_data_M51_6_10.Mass_g.sum()
# M51_6_20_Mass_g_frac = gal_data_M51_6_20[super_Edd_M51_6_20].Mass_g.sum()/gal_data_M51_6_20.Mass_g.sum()
# M51_6_40_Mass_g_frac = gal_data_M51_6_40[super_Edd_M51_6_40].Mass_g.sum()/gal_data_M51_6_40.Mass_g.sum()

# M51_7_5_Mass_g_frac = gal_data_M51_7_5[super_Edd_M51_7_5].Mass_g.sum()/gal_data_M51_7_5.Mass_g.sum()
# M51_7_10_Mass_g_frac = gal_data_M51_7_10[super_Edd_M51_7_10].Mass_g.sum()/gal_data_M51_7_10.Mass_g.sum()
# M51_7_20_Mass_g_frac = gal_data_M51_7_20[super_Edd_M51_7_20].Mass_g.sum()/gal_data_M51_7_20.Mass_g.sum()
# M51_7_40_Mass_g_frac = gal_data_M51_7_40[super_Edd_M51_7_40].Mass_g.sum()/gal_data_M51_7_40.Mass_g.sum()

# NGC6946_6_5_Mass_g_frac = gal_data_NGC6946_6_5[super_Edd_NGC6946_6_5].Mass_g.sum()/gal_data_NGC6946_6_5.Mass_g.sum()
# NGC6946_6_10_Mass_g_frac = gal_data_NGC6946_6_10[super_Edd_NGC6946_6_10].Mass_g.sum()/gal_data_NGC6946_6_10.Mass_g.sum()
# NGC6946_6_20_Mass_g_frac = gal_data_NGC6946_6_20[super_Edd_NGC6946_6_20].Mass_g.sum()/gal_data_NGC6946_6_20.Mass_g.sum()
# NGC6946_6_40_Mass_g_frac = gal_data_NGC6946_6_40[super_Edd_NGC6946_6_40].Mass_g.sum()/gal_data_NGC6946_6_40.Mass_g.sum()

# NGC6946_7_5_Mass_g_frac = gal_data_NGC6946_7_5[super_Edd_NGC6946_7_5].Mass_g.sum()/gal_data_NGC6946_7_5.Mass_g.sum()
# NGC6946_7_10_Mass_g_frac = gal_data_NGC6946_7_10[super_Edd_NGC6946_7_10].Mass_g.sum()/gal_data_NGC6946_7_10.Mass_g.sum()
# NGC6946_7_20_Mass_g_frac = gal_data_NGC6946_7_20[super_Edd_NGC6946_7_20].Mass_g.sum()/gal_data_NGC6946_7_20.Mass_g.sum()
# NGC6946_7_40_Mass_g_frac = gal_data_NGC6946_7_40[super_Edd_NGC6946_7_40].Mass_g.sum()/gal_data_NGC6946_7_40.Mass_g.sum()

# print("M51 1 Myr 5 PC")
# print(f"FBol frac: {M51_6_5_FBol_frac:.8f}")
# print(f"Mgas frac: {M51_6_5_Mass_g_frac:.8f}")
# print("----------")
# print("M51 1 Myr 10 PC")
# print(f"FBol frac: {M51_6_10_FBol_frac:.8f}")
# print(f"Mgas frac: {M51_6_10_Mass_g_frac:.8f}")
# print("----------")
# print("M51 1 Myr 20 PC")
# print(f"FBol frac: {M51_6_20_FBol_frac:.8f}")
# print(f"Mgas frac: {M51_6_20_Mass_g_frac:.8f}")
# print("----------")
# print("M51 1 Myr 40 PC")
# print(f"FBol frac: {M51_6_40_FBol_frac:.8f}")
# print(f"Mgas frac: {M51_6_40_Mass_g_frac:.8f}")

# print("------------------------------")

# print("M51 10 Myr 5 PC")
# print(f"FBol frac: {M51_7_5_FBol_frac:.8f}")
# print(f"Mgas frac: {M51_7_5_Mass_g_frac:.8f}")
# print("----------")
# print("M51 10 Myr 10 PC")
# print(f"FBol frac: {M51_7_10_FBol_frac:.8f}")
# print(f"Mgas frac: {M51_7_10_Mass_g_frac:.8f}")
# print("----------")
# print("M51 10 Myr 20 PC")
# print(f"FBol frac: {M51_7_20_FBol_frac:.8f}")
# print(f"Mgas frac: {M51_7_20_Mass_g_frac:.8f}")
# print("----------")
# print("M51 10 Myr 40 PC")
# print(f"FBol frac: {M51_7_40_FBol_frac:.8f}")
# print(f"Mgas frac: {M51_7_40_Mass_g_frac:.8f}")

# print("------------------------------")

# print("NGC6946 1 Myr 5 PC")
# print(f"FBol frac: {NGC6946_6_5_FBol_frac:.8f}")
# print(f"Mgas frac: {NGC6946_6_5_Mass_g_frac:.8f}")
# print("----------")
# print("NGC6946 1 Myr 10 PC")
# print(f"FBol frac: {NGC6946_6_10_FBol_frac:.8f}")
# print(f"Mgas frac: {NGC6946_6_10_Mass_g_frac:.8f}")
# print("----------")
# print("NGC6946 1 Myr 20 PC")
# print(f"FBol frac: {NGC6946_6_20_FBol_frac:.8f}")
# print(f"Mgas frac: {NGC6946_6_20_Mass_g_frac:.8f}")
# print("----------")
# print("NGC6946 1 Myr 40 PC")
# print(f"FBol frac: {NGC6946_6_40_FBol_frac:.8f}")
# print(f"Mgas frac: {NGC6946_6_40_Mass_g_frac:.8f}")

# print("------------------------------")

# print("NGC6946 10 Myr 5 PC")
# print(f"FBol frac: {NGC6946_7_5_FBol_frac:.8f}")
# print(f"Mgas frac: {NGC6946_7_5_Mass_g_frac:.8f}")
# print("----------")
# print("NGC6946 10 Myr 10 PC")
# print(f"FBol frac: {NGC6946_7_10_FBol_frac:.8f}")
# print(f"Mgas frac: {NGC6946_7_10_Mass_g_frac:.8f}")
# print("----------")
# print("NGC6946 10 Myr 20 PC")
# print(f"FBol frac: {NGC6946_7_20_FBol_frac:.8f}")
# print(f"Mgas frac: {NGC6946_7_20_Mass_g_frac:.8f}")
# print("----------")
# print("NGC6946 10 Myr 40 PC")
# print(f"FBol frac: {NGC6946_7_40_FBol_frac:.8f}")
# print(f"Mgas frac: {NGC6946_7_40_Mass_g_frac:.8f}")

##############################################################################

### Spectra plot
##############################################################################

# plt.figure(dpi = 200)
# plt.plot(BPASS_data.WL*10**-4, BPASS_data["6.0"], label = "1 Myr")
# plt.plot(BPASS_data.WL*10**-4, BPASS_data["7.0"], label = "10 Myr")
# plt.plot([wl_ref,wl_ref], [0,10**9])

# plt.xscale('log')
# plt.yscale('log')

# plt.ylim(1,10**8)
# plt.xlim(10**-2)

# plt.xlabel("Wavelength")

##############################################################################


### Galaxy data by Edd ratio
##############################################################################

# # M51 data
# gal_data_M51_6_5, continuous_SFR, _, _, _, _, _, _ = PrepareGalaxyData("M51", 6.0, BPASS_file, 5)
# gal_data_M51_6_10, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 6.0, BPASS_file, 10)
# # gal_data_M51_6_20, _, _, _, _, _, _ = PrepareGalaxyData("M51", 6.0, BPASS_file, 20)

# gal_data_M51_7_5, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 7.0, BPASS_file, 5)
# gal_data_M51_7_10, _, _, _, _, _, _, _ = PrepareGalaxyData("M51", 7.0, BPASS_file, 10)
# # gal_data_M51_7_20, _, _, _, _, _, _ = PrepareGalaxyData("M51", 6.0, BPASS_file, 20)

# # NGC6946 data
# gal_data_NGC6946_6_5, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 6.0, BPASS_file, 5)
# gal_data_NGC6946_6_10, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 6.0, BPASS_file, 10)
# # gal_data_NGC6946_6_20, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 6.0, BPASS_file, 20)

# gal_data_NGC6946_7_5, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 7.0, BPASS_file, 5)
# gal_data_NGC6946_7_10, _, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 7.0, BPASS_file, 10)
# # gal_data_NGC6946_7_20, _, _, _, _, _, _ = PrepareGalaxyData("NGC6946", 6.0, BPASS_file, 20)

# # Build the plot
# fig, ax = plt.subplots(nrows = 2, ncols = 2, dpi = 200, figsize=(10,7))

# ymin = 0.01
# ymax = 100

# ax[0,0].scatter(gal_data_M51_6_5.AHa, gal_data_M51_6_5.L_Bol / gal_data_M51_6_5.L_Edd, alpha = 0.2, marker = ".", label = "5 pc, 1 Myr")
# ax[0,0].scatter(gal_data_M51_6_10.AHa, gal_data_M51_6_10.L_Bol / gal_data_M51_6_10.L_Edd, alpha = 0.2, marker = ".", label = "10 pc, 1 Myr")
# # ax[0,0].scatter(gal_data_M51_6_20.AHa, gal_data_M51_6_20.L_Bol / gal_data_M51_6_20.L_Edd, alpha = 0.2, marker = ".", label = "20 pc")

# ax[0,0].scatter(gal_data_M51_7_5.AHa, gal_data_M51_7_5.L_Bol / gal_data_M51_7_5.L_Edd, alpha = 0.2, marker = ".", label = "5 pc, 10 Myr")
# ax[0,0].scatter(gal_data_M51_7_10.AHa, gal_data_M51_7_10.L_Bol / gal_data_M51_7_10.L_Edd, alpha = 0.2, marker = ".", label = "10 pc, 10 Myr")
# # ax[0,0].scatter(gal_data_M51_7_20.AHa, gal_data_M51_7_20.L_Bol / gal_data_M51_7_20.L_Edd, alpha = 0.2, marker = ".", label = "20 pc, 10 Myr")

# ax[0,0].plot([0.00001, 10**6], [1, 1], 'k', alpha = 0.5)

# ax[0,1].scatter(gal_data_M51_6_5.Mass_tot/g_Msun, gal_data_M51_6_5.L_Bol / gal_data_M51_6_5.L_Edd, alpha = 0.2, marker = ".", label = "5 pc, 1 Myr")
# ax[0,1].scatter(gal_data_M51_6_10.Mass_tot/g_Msun, gal_data_M51_6_10.L_Bol / gal_data_M51_6_10.L_Edd, alpha = 0.2, marker = ".", label = "10 pc, 1 Myr")
# # ax[0,1].scatter(gal_data_M51_6_20.Mass_tot/g_Msun, gal_data_M51_6_20.L_Bol / gal_data_M51_6_20.L_Edd, alpha = 0.2, marker = ".", label = "20 pc")

# ax[0,1].scatter(gal_data_M51_7_5.Mass_tot/g_Msun, gal_data_M51_7_5.L_Bol / gal_data_M51_7_5.L_Edd, alpha = 0.2, marker = ".", label = "5 pc, 10 Myr")
# ax[0,1].scatter(gal_data_M51_7_10.Mass_tot/g_Msun, gal_data_M51_7_10.L_Bol / gal_data_M51_7_10.L_Edd, alpha = 0.2, marker = ".", label = "10 pc, 10 Myr")
# # ax[0,1].scatter(gal_data_M51_7_20.Mass_tot/g_Msun, gal_data_M51_7_20.L_Bol / gal_data_M51_7_20.L_Edd, alpha = 0.2, marker = ".", label = "20 pc, 10 Myr")

# ax[0,1].plot([0.00001, 10**6], [1, 1], 'k', alpha = 0.5)

# ax[1,0].scatter(gal_data_NGC6946_6_5.AHa, gal_data_NGC6946_6_5.L_Bol / gal_data_NGC6946_6_5.L_Edd, alpha = 0.2, marker = ".", label = "5 pc, 1 Myr")
# ax[1,0].scatter(gal_data_NGC6946_6_10.AHa, gal_data_NGC6946_6_10.L_Bol / gal_data_NGC6946_6_10.L_Edd, alpha = 0.2, marker = ".", label = "10 pc, 1 Myr")
# # ax[1,0].scatter(gal_data_NGC6946_6_20.AHa, gal_data_NGC6946_6_20.L_Bol / gal_data_NGC6946_6_20.L_Edd, alpha = 0.2, marker = ".", label = "20 pc")

# ax[1,0].scatter(gal_data_NGC6946_7_5.AHa, gal_data_NGC6946_7_5.L_Bol / gal_data_NGC6946_7_5.L_Edd, alpha = 0.2, marker = ".", label = "5 pc, 10 Myr")
# ax[1,0].scatter(gal_data_NGC6946_7_10.AHa, gal_data_NGC6946_7_10.L_Bol / gal_data_NGC6946_7_10.L_Edd, alpha = 0.2, marker = ".", label = "10 pc, 10 Myr")
# # ax[1,0].scatter(gal_data_NGC6946_7_20.AHa, gal_data_NGC6946_7_20.L_Bol / gal_data_NGC6946_7_20.L_Edd, alpha = 0.2, marker = ".", label = "20 pc, 10 Myr")

# ax[1,0].plot([0.00001, 10**6], [1, 1], 'k', alpha = 0.5)

# ax[1,1].scatter(gal_data_NGC6946_6_5.Mass_tot/g_Msun, gal_data_NGC6946_6_5.L_Bol / gal_data_NGC6946_6_5.L_Edd, alpha = 0.2, marker = ".", label = "5 pc, 1 Myr")
# ax[1,1].scatter(gal_data_NGC6946_6_10.Mass_tot/g_Msun, gal_data_NGC6946_6_10.L_Bol / gal_data_NGC6946_6_10.L_Edd, alpha = 0.2, marker = ".", label = "10 pc, 1 Myr")
# # ax[1,1].scatter(gal_data_NGC6946_6_20.Mass_tot/g_Msun, gal_data_NGC6946_6_20.L_Bol / gal_data_NGC6946_6_20.L_Edd, alpha = 0.2, marker = ".", label = "20 pc")

# ax[1,1].scatter(gal_data_NGC6946_7_5.Mass_tot/g_Msun, gal_data_NGC6946_7_5.L_Bol / gal_data_NGC6946_7_5.L_Edd, alpha = 0.2, marker = ".", label = "5 pc, 10 Myr")
# ax[1,1].scatter(gal_data_NGC6946_7_10.Mass_tot/g_Msun, gal_data_NGC6946_7_10.L_Bol / gal_data_NGC6946_7_10.L_Edd, alpha = 0.2, marker = ".", label = "10 pc, 10 Myr")
# # ax[1,1].scatter(gal_data_NGC6946_7_20.Mass_tot/g_Msun, gal_data_NGC6946_7_20.L_Bol / gal_data_NGC6946_7_20.L_Edd, alpha = 0.2, marker = ".", label = "20 pc, 10 Myr")

# ax[1,1].plot([0.00001, 10**6], [1, 1], 'k', alpha = 0.5)

# ax[0,0].set_xscale('log')
# ax[0,1].set_xscale('log')
# ax[1,0].set_xscale('log')
# ax[1,1].set_xscale('log')
# ax[0,0].set_yscale('log')
# ax[0,1].set_yscale('log')
# ax[1,0].set_yscale('log')
# ax[1,1].set_yscale('log')

# ax[0,0].set_ylim(ymin, ymax)
# ax[0,1].set_ylim(ymin, ymax)
# ax[1,0].set_ylim(ymin, ymax)
# ax[1,1].set_ylim(ymin, ymax)

# ax[0,0].set_xlim(0.001, 10)
# ax[0,1].set_xlim(500, 10**6)
# ax[1,0].set_xlim(0.001, 10)
# ax[1,1].set_xlim(500, 10**6)

# ax[1,0].set_xlabel(r'$A_{\rm H\alpha}$')
# ax[1,1].set_xlabel(r'Total Mass ($M_\odot$)')
# ax[0,0].set_ylabel(r'Eddington Ratio')
# ax[1,0].set_ylabel(r'Eddington Ratio')

# ax[0,0].text(0.05, 0.05, r'NGC 5194', fontsize=12, transform=ax[0,0].transAxes)
# ax[0,1].text(0.05, 0.05, r'NGC 5194', fontsize=12, transform=ax[0,1].transAxes)
# ax[1,0].text(0.05, 0.05, r'NGC 6946', fontsize=12, transform=ax[1,0].transAxes)
# ax[1,1].text(0.05, 0.05, r'NGC 6946', fontsize=12, transform=ax[1,1].transAxes)

# ax[1,0].legend(loc = (0.002, 0.17))

# plt.tight_layout()

##############################################################################


### Galaxy map with L / M
##############################################################################

# plt.figure(dpi = 200)

# # norm = matplotlib.colors.TwoSlopeNorm(1,vmin = min(gal_data.L_Bol / gal_data.Mass_tot), vmax = max(gal_data.L_Bol / gal_data.Mass_tot))

# plt.scatter(gal_data.xcenter,gal_data.ycenter, s = 1, c = gal_data.L_Bol / gal_data.Mass_tot, cmap = "brg")
# plt.axis('off')
# plt.colorbar()

##############################################################################

## Old plot stuff, maybe not needed.
##############################################################################
# fig, ax = plt.subplots(1,2)

# ax[0].plot(solved.t/cm_pc, solved.y[0]/10**5)
# ax[1].plot(solved.y[1] + 10**time_slice, solved.y[0]/10**5)

# ax[0].set_xscale('log')
# ax[1].set_xscale('log')
# ax[0].set_yscale('log')
# ax[1].set_yscale('log')

# plt.figure(dpi = 200)

# host = host_subplot(111, axes_class=AA.Axes)

# par1 = host.twinx()

# p0, = host.plot(solved.t/cm_pc, solved.y[0]/10**5, label = "Solve IVP")
# # p2, = host.plot(solved.t/cm_pc, np.sqrt(v_0 + 8*Momentum*F_Bol*(solved.t-r[0])/(Sigma_g*c) - 4*G*Sigma_g*(solved.t-r[0]) - 4*G*Sigma_new*(solved.t-r[0]) - 2*G*Sigma_old*(solved.t-r[0])**2/H_old)/(2*10**5), '--', label = "Analytic solution")
# host.set_xscale('log')
# plt.yscale('log')
# host.set_ylim(0.1, 1.2*max(solved.y[0])/10**5)
# host.set_xlabel('Radius (pc)')
# host.set_ylabel('Velocity (km/s)')
# plt.title(r'Region {} velocity'.format(region_ID))

# offset = 60
# new_fixed_axis = par1.get_grid_helper().new_fixed_axis
# par1.axis["right"] = new_fixed_axis(loc="right",
#                                     axes=par1,
#                                     offset=(0, 0))

# p1, = par1.plot(solved.t/cm_pc, solved.y[1], label="Time")
# par1.set_ylim(solved.y[1][0])
# par1.set_ylabel('Time (years)')
# par1.set_yscale('log')
# host.axis["left"].label.set_color(p0.get_color())
# par1.axis["right"].label.set_color(p1.get_color())
# plt.legend()

# # plt.savefig(f'Coupled ODE results region {region_ID} aged.png')

# New_time_slice = BPASS_data.drop('WL', axis = 1).columns[np.abs(BPASS_data.drop('WL', axis = 1).columns.astype(float) - np.log10(solved.y[1][np.abs(solved.t - h_g).argmin()] + 10**time_slice)).argmin()]
# if New_time_slice == str(time_slice):
#     print("time slice guessed correctly!")
# else:
#     print(f'New time slice is {New_time_slice}.  Keep iterating.')

# print(f'Done. Time taken: {time.time() - start_time:.2f}')

# Gamma_UV = gal_data.L_Bol/(4*np.pi*G*c*gal_data.Mass_tot/L_Edd_DF.kappa_av_RP_Gra[L_Edd_DF.time == time_slice].to_numpy())
# Gamma_SS =  gal_data.L_Bol/(G*c*gal_data.Mass_g * gal_data.Mass_tot/h_g**2)

# gal_data['v_inf_3'] = np.sqrt(v_0**2 + 2*G*(gal_data.Mass_tot)/
#                               h_g * (2*Gamma_SS*gal_data.R_UV/h_g*(1-h_g/(2*gal_data.R_UV)) - 1))

# plt.figure(dpi = 200)

# That momentum plot
# momentum = GetMC(np.linspace(0.001,10, 5000), 6.4)
# plt.plot(np.linspace(0.001,10, 5000)*ratio_rp_lambda_rp, momentum, label= "MC Result")
# plt.plot(np.linspace(0.001,10, 5000)*ratio_rp_lambda_rp, 1-np.exp(-np.linspace(0.001,10, 5000)*ratio_rp_lambda_rp), label = r"$1-e^{-\tau_{\rm RP}}$")
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.ylabel("Momentum Transferred (fraction of initial)")
# plt.xlabel(r"Atmospheric depth ($\tau_{\rm RP}$)")
# plt.grid(which = 'both', linestyle = ':')
# plt.tick_params(labelright=True, right = True, grid_color = 'black', grid_alpha = 0.9)
# plt.savefig("Updated MC result plot 7-1-2021.png", dpi = 200)

# plt.scatter(solved.t/cm_pc,solved.y[0]/10**5, label = 'ODE solution', color = 'blue', marker = 'x')
# # plt.plot(x,y, label = 'ODE solution', color = 'orange')
# # plt.plot([solved.t[0]/cm_pc,solved.t[-1]/cm_pc], (gal_data.v_inf_1[mask].to_list()[0]/10**5,gal_data.v_inf_1[mask].to_list()[0]/10**5), label = r'$\sqrt{v_{\rm esc}}\sqrt{L_{\rm bol}/L_{\rm Edd}}$', color ='orange')
# # plt.plot([solved.t[0]/cm_pc,solved.t[-1]/cm_pc], (gal_data.v_inf_2[mask].to_list()[0]/10**5,gal_data.v_inf_2[mask].to_list()[0]/10**5), label = r'$\sqrt{(4R_{\rm UV}L)/(M_{\rm g}c)}$', color ='red')
# # plt.plot([solved.t[0]/cm_pc,solved.t[-1]/cm_pc], (gal_data.v_inf_3[mask].to_list()[0]/10**5,gal_data.v_inf_3[mask].to_list()[0]/10**5), label = 'Analytic solution', color ='green')
# # plt.yscale('log')
# # plt.xscale('log')
# plt.xlabel(r'$h_{\rm g}$ (pc)')
# plt.ylabel('v (km/s)')
# # plt.legend()

# title = 'velocity data for region ' + str(region_ID)
# plt.title(title)

# plt.savefig('region ' + str(region_ID) + ' ' + str(int(h_g/cm_pc)) + ' pc velocity non log.png', dpi = 200)

# bins = []

# for i in range(0,len(solved.t)-1):
#     bins += [solved.t[i+1]-solved.t[i]]

# times = []
# for i, distance in enumerate(bins):
#     times += [distance/((solved.y[0][i]+solved.y[0][i+1])/2)]
    
# total_time = sum(times)/60/60/24/365


# func = lambda r : L_Bol/((G * c * M_g * (M_g + M_new + Sigma_old * 2/3 * np.pi * r**3 /h_old))/(r**2 * 
#                 (np.interp((h_g/r)**2*tau_i, MC_data.tau.values, MC_data.momentum.values)
#                   + get_tau_IR(r, L_Bol, (h_g/r)**2 * Sigma_g)))) - 1

# plt.figure(dpi = 200)

# host = host_subplot(111, axes_class=AA.Axes)

# par1 = host.twinx()
# par2 = host.twinx()
# par3 = host.twinx()
# par4 = host.twinx()

# offset = 60
# new_fixed_axis = par2.get_grid_helper().new_fixed_axis
# par1.axis["right"] = new_fixed_axis(loc="right",
#                                     axes=par1,
#                                     offset=(0, 0))


# par2.axis["right"] = new_fixed_axis(loc="right",
#                                     axes=par2,
#                                     offset=(offset, 0))

# par3.axis["right"] = new_fixed_axis(loc="right",
#                                     axes=par3,
#                                     offset=(2*offset, 0))

# par4.axis["left"] = new_fixed_axis(loc="left",
#                                     axes=par4,
#                                     offset=(-offset, 0))

# par2.axis["right"].toggle(all=True)

# host.set_xlabel('radius (pc)')
# host.set_xscale('log')
# host.set_ylabel('Eddington ratio')
# host.set_ylim(0,2)
# host.set_yscale('log')
# par1.set_ylabel(r' Old Stellar Mass $(M_\odot)$')
# par1.set_yscale('log')
# par2.set_ylabel(r'$\langle \tau_{\rm RP} \rangle$')
# par2.set_yscale('log')
# par2.set_ylim(min((h_g/r)**2*tau_i*ratio_rp_lambda_rp),max((h_g/r)**2*tau_i*ratio_rp_lambda_rp))
# par3.set_ylabel('Momentum transfer')
# par3.set_yscale('log')
# par3.set_ylim(min((h_g/r)**2*tau_i*ratio_rp_lambda_rp),max((h_g/r)**2*tau_i*ratio_rp_lambda_rp))
# par4.set_ylabel('Velocity (km/s)')


# p0, = host.plot(r/cm_pc, EddRatio, label = 'Eddington ratio')
# host.plot([r[0]/cm_pc,r[-1]/cm_pc], [1,1], color = 'black')

# plt.title('Region ' +  str(region_ID) + r', with $h_{\rm g} \, =$ ' + str(h_g/cm_pc) + ' pc.')
# p1, = par1.plot(r/cm_pc,  (Sigma_old * 2/3 * np.pi * r**3 /h_old)/g_Msun)
# p2, = par2.plot(r/cm_pc,  (h_g/r)**2*tau_i*ratio_rp_lambda_rp)
# p3, = par3.plot(r/cm_pc,  np.interp((h_g/r)**2*tau_i, MC_data.tau.values, MC_data[str(time_slice)].values) + get_tau_IR(r, L_Bol,  (h_g/r)**2 * Sigma_g))
# p4, = par4.plot(solved.t/cm_pc,solved.y[0]/10**5)

# host.axis["left"].label.set_color(p0.get_color())
# par1.axis["right"].label.set_color(p1.get_color())
# par2.axis["right"].label.set_color(p2.get_color())
# par3.axis["right"].label.set_color(p3.get_color())
# par4.axis["left"].label.set_color(p4.get_color())

# plt.yscale('symlog')
# plt.xscale('log')
# plt.ylabel('Eddington ratio')
# plt.xlabel('radius (pc)')
# plt.grid(True, which="both")
# plt.savefig('Eddigton ratio by radius region ' + str(region_ID) + ' ' + str(int(h_g/cm_pc)) + '.png', dpi = 200)


# for distance in r:
#     T_Eff = (gal_data.L_Bol/(4*np.pi* distance**2*sigma_SB))**0.25
#     Kappa_R = 2.4*10**-4 * T_Eff**2
#     tau_IR = Kappa_R * gal_data.Sigma_g
#     Temp = (3/4*(tau_IR + 2/3)*T_Eff**4)**0.25
#     tau_IR[tau_IR > 1] = 2.4*10**-4 * Temp**2 * gal_data.Sigma_g

# gal_data['L_Edd_IR'] = (G * c * gal_data['Mass_g'] * gal_data.Mass_tot  / (h_g**2 * (gal_data['Momentum'] + tau_IR)))

# fig, [ax1, ax2, ax3] = plt.subplots(1,3, sharex=True)
# fig.set_figheight(5)
# fig.set_figwidth(15)

# # ax1.scatter(gal_data.tau[gal_data.L_Bol > gal_data.L_Edd_IR], tau_IR[gal_data.L_Bol > gal_data.L_Edd_IR], label = r'$\tau_{\rm IR}$')
# # ax1.scatter(gal_data.tau[gal_data.L_Bol > gal_data.L_Edd_IR], gal_data.Momentum[gal_data.L_Bol > gal_data.L_Edd_IR], label = 'MC result')

# ax1.scatter(gal_data.tau, tau_IR, label = r'$\tau_{\rm IR}$ included')
# ax1.scatter(gal_data.tau, gal_data.Momentum, label = 'MC result only', marker='x')
# ax1.set_yscale('log')
# ax1.set_xscale('log')
# ax1.legend()
# ax1.set_xlabel(r'$\tau_{\rm RP, \, \lambda = H\alpha}$')
# ax1.set_ylabel('Momentum transfer')

# ax2.scatter(gal_data.tau, gal_data.L_Edd_IR/gal_data.L_Edd)
# ax2.set_ylabel(r'$L_{\rm Edd,\, IR}/L_{\rm Edd}$')
# ax2.set_xlabel(r'$\tau_{\rm RP, \lambda = H\alpha}$')
# ax2.set_xscale('log')
# ax2.set_yscale('log')

# ax3.scatter(gal_data[gal_data.L_Bol > gal_data.L_Edd_IR].tau, gal_data.L_Bol[gal_data.L_Bol > gal_data.L_Edd_IR]/gal_data.L_Edd_IR[gal_data.L_Bol > gal_data.L_Edd_IR], label = 'IR Edd ratio')
# ax3.scatter(gal_data[gal_data.L_Bol > gal_data.L_Edd].tau, gal_data.L_Bol[gal_data.L_Bol > gal_data.L_Edd]/gal_data.L_Edd[gal_data.L_Bol > gal_data.L_Edd], label = 'Edd ratio', marker='x')
# ax3.set_xlabel(r'$\tau_{\rm RP, \, \lambda = H\alpha}$')
# ax3.set_ylabel('Eddington ratio')

# plt.tight_layout()
# plt.savefig('MC result with and without tau IR.png', dpi = 200)

# guess = h_g
# solution = fsolve(func, guess)

# print('Best guess for super Eddington split: ' + str(round(solution[0]/cm_pc,2)) + ' pc')
# print('Mass of gas shell: ' + str(round(M_g/g_Msun,2)))