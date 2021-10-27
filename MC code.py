# Imports
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from hoki import load
from scipy import integrate as inte
from scipy.interpolate import interpn
# from scipy.optimize import fsolve
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import sys

from read_BPASS import GetGrainData


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
time_slice = 7.0

# Declare the number of bins, photons, and atmospheres.
num_photons = 20000000
num_bins = 20000
num_atm = 100
photon_reduction = 1000

wl_min = 0.001
wl_max = 10

# "mix", "Sil", "Gra", or "SiC"
grain_type = 'mix'
grain_min = 0.001
grain_max = 1

# Define the mixture of [Sil, Gra, SiC].  These should add to 1.
grain_mix = [0.5,0.45,0.05]

min_tau = 10**-3
max_tau = 100

# Select the BPASS files to use.  Uncomment the file to use
BPASS_file = 'spectra-bin-imf135_300.z020.dat'
# BPASS_file = 'spectra-bin-imf135_100.z020.dat'
# BPASS_file = 'spectra-bin-imf100_300.z020.dat'
# BPASS_file = 'spectra-bin-imf100_300.z010.dat'
# BPASS_file = 'spectra-bin-imf_chab300.z020.dat'
ionizing_file = BPASS_file.replace('spectra','ionizing')

MC_data = pd.read_csv('detailed MC {}csv'.format(BPASS_file.replace('.z',' z').replace('dat','')))

# tau_list = np.logspace(-3, 2, num_atm, base=10)
tau_list = MC_data.tau.to_numpy()
tau_list_calc = tau_list[(tau_list >= min_tau) & (tau_list <= max_tau)]
num_atm = len(tau_list_calc)

# Flag for the boundary.  it can be 'absorb', 'reemit', or 'reflect'.
boundary = 'reflect'

# Flag for the scattering type.  'hg', 'iso', 'draine'
scatter = 'hg'

# Boolean flags for randomizing mu at the start and turning off scattering.
rand_mu = 1
abs_only = 0

# monochromatic light, set to 0 to draw from an SED.
monochromatic = 0

# cm per pc
cm_pc = 3.086*10**18

# g per solar mass
g_Msun = 1.988*10**33

# (ergs/s) / L_Sun
L_sun_conversion = 3.826*10**33

h_old = 1000
h_old = h_old * cm_pc

# Height of the gas column in parsecs
h_g = 1

# Sigma g scaling factor
h_g_scale = 1/h_g

# Convert to cgs
h_g = h_g * cm_pc

region_ID = 162

# dust to gas ratio
f_dg = 1/100

# Starting velocity in cm/s and time in years.
v_0 = 100
t_0 = 10**6

# Define the files to read
gal_file_1 = 'NGC5194_Full_Aperture_table_2arc.dat'
gal_file_2 = 'M51_3p6um_avApertures_MjySr.txt'

# The center of the galaxy, obtained from http://ned.ipac.caltech.edu/
# [RA, Dec] Numbers here are assuming that our dataset is in the J2000.0 frame.
gal_center = [202.484167, 47.230556]

# Distance to the galaxy in pc
Distance = 8.34*10**6

# Resolution in arcseconds
AngularResolution = 1

# Spatial resolution in cm
Resolution = Distance * np.tan(AngularResolution*np.pi/648000) * cm_pc

# Define the column scale for the old stellar mass (r_star / h_star)
h_scale = 7.3

# Determine wether to age the stellar population as the galaxy simulation runs.
age = True

# determine which model to use
model = "planar"

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

start_time = time.time()

# Load the BPASS data
BPASS_data = load.model_output(BPASS_file)
# Make a copy of the BPASS data to downsample.
BPASS_data_r = BPASS_data.copy()
# Load the ionizing photon data.
ion_data = load.model_output(ionizing_file)

H_Alpha_ratio = (np.power(10,ion_data.halpha[ion_data.log_age == time_slice]))/L_sun_conversion/(BPASS_data[str(time_slice)].sum())

# Convert the BPASS data to microns.
BPASS_data_r.WL *= 10**-4

BPASS_data_r = BPASS_data_r[ (BPASS_data_r.WL >= wl_min) & (BPASS_data_r.WL <= wl_max) ]

if grain_type == "mix":
    Sil_data, Sil_kappa = GetGrainData('Sil', grain_min, grain_max, BPASS_data_r, time_slice, wl_ref)
    Gra_data, Gra_kappa = GetGrainData('Gra', grain_min, grain_max, BPASS_data_r, time_slice, wl_ref)
    SiC_data, SiC_kappa = GetGrainData('SiC', grain_min, grain_max, BPASS_data_r, time_slice, wl_ref)
    
    Grain_data = Sil_data*grain_mix[0] + Gra_data*grain_mix[1] + SiC_data*grain_mix[2]
    kappa_data = Sil_kappa*grain_mix[0] + Gra_kappa*grain_mix[1] + SiC_kappa*grain_mix[2]
    
else:
    Grain_data, kappa_data = GetGrainData(grain_type, grain_min, grain_max, BPASS_data_r, time_slice, wl_ref)

# kappa_data = pd.read_csv(kappa_file)

kappa_data.WL = kappa_data.WL.round(4)
Grain_data.WL = Grain_data.WL.round(4)

Grain_data = pd.merge(Grain_data, kappa_data, left_on = 'WL', right_on = 'WL')


# Not doing absorption only right now.
# if abs_only == 1:
#     kappa = pd.read_csv('abs 1 kappa by wl.csv')
#     L_Edd = pd.read_csv('abs 1 L_Edd dataframe.csv')
# else:
    
# kappa = pd.read_csv('kappa by wl.csv', index_col=0)



## Generate photon counts
# -----------------------------------------------------

BPASS_data_r = BPASS_data_r.iloc[::100,:]

# BPASS_data_r.WL *= 10**-4

h = 6.6261*(10**(-27))

c = 2.99792458*(10**10)
G = 6.67259*(10**-8)
sigma_SB = 5.67037441918442945397*10**-5

# BPASS_data_r.iloc[:,1:-1] *= 10**8

photon = BPASS_data_r[str(time_slice)] * BPASS_data_r.WL**2 / (h*c)

# -----------------------------------------------------
    
## Find CDF
# -----------------------------------------------------

norm = np.trapz(photon, x = BPASS_data_r.WL)
CDF = np.zeros_like(BPASS_data_r.WL)

for i, _ in enumerate(BPASS_data_r.WL):
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


def GetMC(tau, new_time):
    
    if np.any(new_time == MC_data.drop('tau', axis=1).columns.values.astype(float)):
        MC_result = np.interp(tau, MC_data.tau, MC_data[str(new_time)])    
    else:
        MC_result = interpn((MC_data.tau.values, MC_data.drop('tau', axis = 1).columns.values.astype(float))
                            ,MC_data.drop('tau', axis = 1).to_numpy(), (tau, new_time))

    return MC_result

VecMC = np.vectorize(GetMC)

def ODEs(r, X):
    ##########################################################################
    # Returns the ODE for dv/dr and dt/dr.
    #
    # r -- A number giving the physical size of the region in cm.
    # X -- A vector containing [v, t] in cm/s and years.
    ##########################################################################
    
    v, t = X
    
    tau_new = (h_g/r)**2*tau_i
    
    new_time = np.log10(t)
    
    L_ratio = np.interp(new_time,BPASS_data.drop('WL', axis = 1).sum(axis = 0).index.astype(float), BPASS_data.drop('WL', axis = 1).sum(axis = 0).values)/BPASS_data[str(time_slice)].sum()
    
    L_Bol_new = L_Bol
    
    if age:
        L_Bol_new *= L_ratio
        momentum_new = GetMC(tau_new, new_time) + get_tau_IR(r, L_Bol_new,  (h_g/r)**2 * Sigma_g)
    else:
        momentum_new = GetMC(tau_new, time_slice) + get_tau_IR(r, L_Bol_new,  (h_g/r)**2 * Sigma_g)
    
    # LEdd = (G*c*M_g*(M_g + M_new + M_old * (r/h_g)**3))/(momentum_new * r**2)
    
    dvdr = (-G*(M_g + M_new + M_old*(r/h_g)**3)/r**2 + momentum_new*L_Bol_new/(c*M_g))/v

    dtdr = 1/(v*31556952)

    return dvdr, dtdr

def FluxODEs(r, X):
    ##########################################################################
    # Returns the ODE for dv/dr and dt/dr.
    #
    # r -- A number giving the physical size of the region in cm.
    # X -- A vector containing [v, t] in cm/s and years.
    ##########################################################################
    
    v, t = X
    
    F_g = G / 2 * ( Sigma_g + Sigma_new + Sigma_old * min(r/H_old,1))

    new_time = np.log10(t)
    
    Flux = F_Bol
    
    if age:
        L_ratio = np.interp(new_time,BPASS_data.drop('WL', axis = 1).sum(axis = 0).index.astype(float), BPASS_data.drop('WL', axis = 1).sum(axis = 0).values)/BPASS_data[str(time_slice)].sum()
    
        Momentum_new = GetMC(tau_i, new_time) + get_tau_IR(r, L_Bol*L_ratio,  Sigma_g)

        Flux *= Momentum_new*L_ratio
    else:
        Flux *= Momentum
    
    if r > Radius*cm_pc:
        Flux += C * F_gal * ((Radius*cm_pc-H_old)/r)**2
    elif r > H_old:
        Flux += C * F_gal * ((r-H_old)/Radius*cm_pc)**2
    
    dvdr = (Flux/(c*Sigma_g) - F_g)/v
    
    dtdr = 1/(v*31556952)
    
    # if np.random.rand() > 0.9:
    #     print("r: {}, Edd: {}, dvdr: {}, time: {}".format(r/cm_pc,Flux/F_Edd, dvdr, new_time))
    
    return dvdr, dtdr

def runMC(BPASS_file, time_slice, num_photons, photon_reduction=0, boundary = 'reflect', scatter = 'hg', save_MC = True):
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
    
    BPASS_data_r = BPASS_data_r.iloc[::100,:]

    photon = BPASS_data_r[str(time_slice)] * BPASS_data_r.WL**2 / (h*c)
    
    # -----------------------------------------------------
        
    ## Find CDF
    # -----------------------------------------------------
    
    norm = np.trapz(photon, x = BPASS_data_r.WL)
    CDF = np.zeros_like(BPASS_data_r.WL)
    
    for i, _ in enumerate(BPASS_data_r.WL):
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
        escaped_mu, boundary_momentum, escaped_momentum, initial_momentum, _ = RunPhotons(tau_atm, num_photons, boundary, scatter, PDF, Range)
    
        momentum_method_1[atm_i] = (initial_momentum - escaped_momentum + boundary_momentum)/initial_momentum
    
    momentum_transfer = np.zeros(len(tau_list))
    momentum_transfer[tau_list < min_tau] = tau_list[tau_list < min_tau]*np.mean(momentum_method_1[0:10]/(1-np.exp(-tau_list_calc[0:10])))
    momentum_transfer[(tau_list >= min_tau) & (tau_list <= max_tau)] = momentum_method_1
    momentum_transfer[tau_list > max_tau] = np.mean(momentum_method_1[-5:])
    
    if save_MC:
        MC_data[str(time_slice)] = momentum_transfer
        MC_data.to_csv('detailed MC {}csv'.format(BPASS_file.replace('.z',' z').replace('dat','')), index = False)
    
    return momentum_transfer

def ContinuousStarFormationSED(BPASS_data, time_list_exp, delta_t, age, rate):
    
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

time_list = BPASS_data.columns[BPASS_data.columns != 'WL']

time_list_exp = np.power(10,time_list.astype(float))


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
        
        SED = ContinuousStarFormationSED(BPASS_data, time_list_exp, delta_t, age, rate)
        
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

L_Edd_DF = pd.read_csv('L_Edd dataframe {}.csv'.format(BPASS_file.replace('.z',' z').replace('dat','')))

ratio_lambda_f_lambda_rp = Grain_data.Kappa_F[Grain_data.WL.round(decimals=4) == wl_ref].to_numpy()/Grain_data.Kappa_RP[Grain_data.WL.round(decimals = 4) == wl_ref].to_numpy()

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

# ## Combine galaxy data
# ## ---------------------------------------------------------------------------

# Open and read the files
df1 = pd.read_csv(gal_file_1, delim_whitespace = True)
df2 = pd.read_csv(gal_file_2, delim_whitespace = True)

# Merge the files and delete the extra id column.
gal_data = pd.merge(df1,df2, left_on = 'ID', right_on = 'id')
gal_data = gal_data.drop(columns='id')

# calculate the distance to galactic center.
gal_data['Dist_to_center'] = Distance * np.sqrt(np.tan(np.radians(gal_data['RA'] - gal_center[0]))**2 + np.tan(np.radians(gal_data['Dec'] - gal_center[1]))**2)

# Calculate the old stellar column height.

gal_data['H_old'] = gal_data.Dist_to_center/h_scale * cm_pc

# gal_data['H_old'] = h_old

# Define new columns
# Find bolometric luminosty: L_Ha / 0.00724 (Kennicutt & Evans)
gal_data['L_Bol'] = gal_data.LHa/H_Alpha_ratio.values

gal_data['F_Bol'] = gal_data.L_Bol / (np.pi * Resolution**2)

# Find Sigma_old, area density of old stars.
# in solar masses/pc^2, then converted to cgs units.
gal_data['Sigma_star'] = 350*gal_data['3.6um_aperture_average'] / cm_pc**2 * g_Msun * np.cosh(h_g/(2*(gal_data['H_old'])))**-2
gal_data['Sigma_g'] = gal_data.AHa/(1.086 * Grain_data.Kappa_F[Grain_data.WL.round(decimals=4) == wl_ref].to_numpy() * f_dg)

# ## -------------------------------------------------------------------------


## Get Eddington ratios for galaxies
## ---------------------------------------------------------------------------

# gal_file = 'M51 MC Complete.csv'
# gal_file = 'M51.csv'

Range = BPASS_data_r.WL.to_numpy()

# gal_data = pd.read_csv(gal_file)

gal_data['Momentum'] = 0

# gal_data['tau'] = gal_data.AHa/(1.086 * ratio_lambda_f_lambda_rp.values[0])

gal_data['tau'] = gal_data['Sigma_g'] * Grain_data.Kappa_RP[Grain_data.WL.round(decimals=4) == wl_ref].to_numpy() * f_dg

L_over_M = L_Edd_DF[L_Edd_DF.time == time_slice].L_bol_BPASS/L_Edd_DF[L_Edd_DF.time == time_slice].Mass

# Mass of new stars, in grams.
gal_data['Mass_new'] = gal_data.L_Bol/(L_over_M.to_numpy()[0] * L_sun_conversion  / g_Msun)
gal_data['Sigma_new'] = gal_data['Mass_new'] / (np.pi * Resolution**2)


# check which model to use for the galaxy (not the MC simulation).
# Mass of old stars in M_sun.
gal_data['Mass_old'] = gal_data.Sigma_star * 2/3 * np.pi * h_g**3 / (gal_data['H_old'])

# Mass of gas in M_sun.
gal_data['Mass_g'] = gal_data.AHa/(1.086 * Grain_data.Kappa_F[Grain_data.WL.round(decimals=4) == wl_ref].to_numpy() * f_dg)*4*np.pi*h_g**2

gal_data['Mass_tot'] = (gal_data.Mass_g + gal_data.Mass_old + gal_data.Mass_new)

momentum_method_1 = np.zeros(num_atm)
# escaped_mu = np.zeros_like(momentum_method_1)
scatter_count = np.zeros_like(momentum_method_1)



# for i, row in gal_data.iterrows():

#     tau_max = row.tau
    
#     escaped_mu, boundary_momentum, escaped_momentum, initial_momentum, scatter_count = RunPhotons(tau_max , num_photons, boundary, scatter, PDF, Range)
    
#     momentum = (initial_momentum - escaped_momentum + boundary_momentum)/initial_momentum
    
#     gal_data.loc[gal_data.ID == row.ID, 'Momentum'] = momentum
    
# MC_data = pd.read_csv('detailed MC.csv')
momentum = GetMC(gal_data.tau.values, time_slice)
gal_data['Momentum'] = momentum


gal_data['L_Edd'] = (G * c * gal_data['Mass_g'] * gal_data.Mass_tot  / (h_g**2 * gal_data['Momentum'] + get_tau_IR(h_g, gal_data.L_Bol, gal_data.Sigma_g)))

gal_data['F_Edd'] = (2 * np.pi * G * c * gal_data['Sigma_g'] * (gal_data['Sigma_star'] * np.minimum(h_g /gal_data['H_old'],1)) + gal_data['Sigma_g'] + gal_data['Sigma_new'])/ (gal_data['Momentum'])

gal_data['tau_RP'] = gal_data.tau * ratio_rp_lambda_rp

Edd_ratio = gal_data.L_Bol/gal_data.L_Edd-1
Edd_ratio[Edd_ratio < 0] = 0

# gal_data['v_inf_1'] = np.sqrt(2*G*gal_data.Mass_tot/h_g)*np.sqrt(Edd_ratio)

# gal_data['R_UV'] = (L_Edd.kappa_av_RP_Sil[L_Edd.time == time_slice].to_numpy()[0] * f_dg * gal_data.Mass_g/(4*np.pi))**0.5
# gal_data['v_inf_2'] = ((4 * gal_data.R_UV * gal_data.L_Bol)/(gal_data.Mass_g * c))**0.5

# ratio = gal_data.L_Bol[gal_data.L_Bol > gal_data.L_Edd].sum()/gal_data.L_Bol.sum()
# print(str(h_g/cm_pc) + ' pc, Super Eddington luminosity ratio: ' + str(ratio))

# superEdd = gal_data.L_Bol > gal_data.L_Edd
# opThick = gal_data.tau_RP >= 1

# evolve region velocity
# ---------------------------------------------------------------------------
M_g = gal_data.Mass_g[gal_data.ID == region_ID].values[0]
M_new = gal_data.Mass_new[gal_data.ID == region_ID].values[0]
M_old = gal_data.Mass_old[gal_data.ID == region_ID].values[0]
Sigma_old = gal_data.Sigma_star[gal_data.ID == region_ID].values[0]
Sigma_g = gal_data.Sigma_g[gal_data.ID == region_ID].values[0]
Sigma_new = gal_data.Sigma_new[gal_data.ID == region_ID].values[0]
tau_i = gal_data.tau[gal_data.ID == region_ID].values[0]
L_Bol = gal_data.L_Bol[gal_data.ID == region_ID].values[0]
H_old = (gal_data.H_old[gal_data.ID == region_ID].values[0])
Momentum = gal_data.Momentum[gal_data.ID == region_ID].values[0]
F_Bol = gal_data.F_Bol[gal_data.ID == region_ID].values[0]
F_Edd = gal_data.F_Edd[gal_data.ID == region_ID].values[0]
C = 0.5
F_gal = gal_data.F_Bol.sum()
Radius = gal_data.Dist_to_center[gal_data.ID == region_ID].values[0]


EddRatioFunc = lambda r : L_Bol/((G * c * M_g * (M_g + M_new + Sigma_old * 2/3 * np.pi * r**3 /H_old))/(r**2 * 
                (GetMC((h_g/r)**2*tau_i, time_slice)
                  + get_tau_IR(r, L_Bol, (h_g/r)**2 * Sigma_g))))

EddFluxRatioLowr = lambda r : F_Bol / ((G * c * Sigma_g * (Sigma_old * r /H_old) + Sigma_g + Sigma_new)/ (2 * Momentum))


# gal_data = pd.read_csv('tau test.csv')

mask = gal_data.ID == region_ID

r = np.linspace(h_g_scale*h_g, 100*H_old,num_bins)

Edd_Ratio = EddFluxRatioLowr(r)

delta_r = (25-h_g_scale)*h_g/num_bins

print(f'Galaxy data setup done.  Time taken: {time.time() - start_time:.5}')

# try:
#     r_span = [r[Edd_Ratio > 1][0],r[-1]]
# except:
#     print('Region {} is not super-Eddington.'.format(region_ID))
#     sys.exit(0)

# # dvdr(r, v, M_g, M_new, M_old_i, L_Bol, tau_i, Sigma_g)

# if model == "spherical":
#     solved = inte.solve_ivp(ODEs, r_span, [v_0, t_0], method='Radau')
# elif model == "planar":
#     solved = inte.solve_ivp(FluxODEs, r_span, [v_0, t_0], method='Radau', max_step = cm_pc)
# else:
#     print('invalid model selection, use "spherical" or "planar".')
#     sys.exit(0)

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
# # host.set_xscale('log')
# # plt.yscale('log')
# host.set_ylim(0, 1.2*max(solved.y[0])/10**5)
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

# Gamma_UV = gal_data.L_Bol/(4*np.pi*G*c*gal_data.Mass_tot/L_Edd.kappa_av_RP_Gra[L_Edd.time == time_slice].to_numpy())
# Gamma_SS =  gal_data.L_Bol/(G*c*gal_data.Mass_g * gal_data.Mass_tot/h_g**2)

# gal_data['v_inf_3'] = np.sqrt(v_0[0]**2 + 2*G*
#                               (gal_data.Mass_tot)/
#                               h_g * (2*Gamma_SS*gal_data.R_UV/h_g*(1-h_g/(2*gal_data.R_UV)) - 1))

# plt.figure(dpi = 200)

# # That momentum plot
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
#                  + get_tau_IR(r, L_Bol, (h_g/r)**2 * Sigma_g)))) - 1

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