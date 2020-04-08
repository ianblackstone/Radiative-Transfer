# imports
import numpy as np
import matplotlib.pyplot as plt

# Later I want to convert this to CUDA, since I have ~3000 times the FLOPS on the GPU.
# import cupy as cu

# atmosphere depth, numpy array from 0 to 10 with N_atm evenly spaced samples
N_atm = 100
tau_atm = np.logspace(-2,1,N_atm,base=10)
# tau_atm = np.array([0.01,0.03,0.1,0.3,1,3,10])

# The number of photons to simulate for each optical depth and wavelength
N_photons = 100
# The number of different wavelengths to simulate
N_wavelengths = 10


# Create a distribution of wavelengths, obviously not the actual distribution
# Ideally this will eventually be a PDF and the function will take the PDF as an input.
Photon_Wavelengths = np.linspace(0.1,2,N_wavelengths)

# # Number of bins in the mu and phi directions.
# # mu ranges from 0 to 1 and phi ranges from 0 to 2pi
# N_mu = 100
# N_phi = 100

# photons starts at z = 0, moving upward.
# Making space for the x and y coords even if not being used now.
tau_i = [0,0,0]
mu_i = 1
phi_i = 0

g = [-1,-0.5,0.001,0.5,1]
a = 0

# We will use units of h*nu/c = 1, We can change this or iterate over a list of frequencies later.
photon_momentum = 1

# Total momentum
momentum_i = photon_momentum*N_photons

momentum_transfer = momentum_i*np.ones([N_wavelengths,N_atm])

# Counter for the number of photons that get absorbed
N_absorbed = np.zeros([N_wavelengths,N_atm])

def TakeStep(tau,mu,phi):
    # Travel some step size in the direction it was heading in.
    # tau = current position of the photon, stored as a numpy array [x,y,z].
    # mu = cos(theta), the cosine of the angle between the velocity vector of the photon and the z axis.
    # phi = the azimuthal angle of the photon's velocity.
    
    # calculate the step size before an interaction.
    step =  -np.log(1-np.random.rand())
    # tau[0] += mu*np.cos(phi)*step
    # tau[1] += mu*np.sin(phi)*step
    tau[2] += mu*step
    return tau, step


def Scatter(g):
    # decide on a new random direction to scatter in.
    s = 2*np.random.rand() - 1
    
    # This is the Henyey Greenstein relation for anisotropic scattering.
    mu = 1/(2*g)*(1+g**2-((1-g**2)/(1+g*s))**2)
    phi = 2*np.pi*np.random.rand()
    
    return mu, phi

def Absorb(wavelength):
    # Determine the probability of absorbtion at a given wavelength:
    absorb_chance = 1-np.exp(-wavelength)
    return absorb_chance


def PerformRun(g):
    # Total momentum
    momentum_i = photon_momentum*N_photons

    momentum_transfer = momentum_i*np.ones_like(tau_atm)
    
    # Loop over each atmospheric depth.  atm will be the atmospheric depth, atm_i is an iteration variable.
    for atm_i,atm in enumerate(tau_atm):
        # Loop over each wavelength
        for wave_i,wavelength in enumerate(Photon_Wavelengths):
            absorb_chance = Absorb(wavelength)
            for phot_i in range(N_photons):
                # Set the initial conditions for the photons
                tau = tau_i[:]
                mu = mu_i
                phi = phi_i
                
                # Keep making steps until the photon is absorbed or transmitted.
                while 1:
                    tau, step = TakeStep(tau,mu,phi)
                   
                    if tau[2] >= atm:
                        momentum_transfer[wave_i,atm_i] -= mu*photon_momentum
                        break
                    elif tau[2] < 0:
                        N_absorbed[wave_i,atm_i] += 1
                        momentum_transfer[wave_i,atm_i] -= mu*photon_momentum
                        tau[2]= tau[2]*(-1)
                    
                    # Not sure this is the right relationship for absorption but adding this for now.
                    if np.random.rand() > absorb_chance:
                        # Check if the photon was absorbed, then add the initial momentum back if it was.
                        N_absorbed[wave_i,atm_i] += 1
                        momentum_transfer[wave_i,atm_i] += photon_momentum
                        break
                
                # If the photon did not escape or get absorbed decide on a new scattering angle.
                mu,phi = Scatter(g)
    
    # Calculate the actual fraction transmitted and the theoretical line for comparison.
    frac_transmitted = 1-N_absorbed/N_photons
    theory = 1/(1+tau_atm/2)
    
    return frac_transmitted, theory, momentum_transfer

gdata = []

# for each in g:
#     frac_transmitted, theory, momentum_transfer = PerformRun(each)
#     # gdata.append([frac_transmitted, theory, momentum_transfer])
#     plt.plot(np.log10(tau_atm),momentum_transfer/N_photons,label=each)

# # Plot data
# # plt.plot(np.log10(tau_atm), frac_transmitted, label="Simulation data")
# # plt.plot(np.log10(tau_atm),theory,label = r'$\frac{1}{1+\frac{\tau}{2}}$')
# # plt.legend()
# # plt.xlabel(r'$log_{10}(\tau_{atm})$')
# # plt.ylabel('Fraction of transmitted photons')
# # plt.savefig('1D_Plot.png',dpi=200)


# plt.xlabel(r'$log_{10}(\tau_{atm})$')
# plt.ylabel('Momentum transferred per photon')
# plt.title('Momentum transfer with reflecting boundary')
# plt.legend(title="anisotropic weighting")
# plt.savefig('1D_Momentum_Plot_With_Absorption.png',dpi=200)