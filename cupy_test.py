# imports
import cupy as cp
import matplotlib.pyplot as plt
import time

a = time.time()

# atmosphere depth, numpy array from 0 to 10 with N_atm evenly log spaced samples
N_atm = 100
tau_atm = cp.logspace(-2,1,N_atm,base=10)
# tau_atm = np.array([0.01,0.03,0.1,0.3,1,3,10])

# The number of photons to simulate for each optical depth
N_photons = 1000

# # Set a counter for the number of photons absorbed.  Not used for momentum calculation.
# N_absorbed = cp.zeros(N_atm)

# Henyeye-Greenley parameters
g = [-1,-0.5,0.001,0.5,1]

# g = [1]

# Create an array of wavelengths.  Units in nm.
Wavelengths = cp.linspace(100,500,50)

# Pick a uniform grain size, units in micrometers.

a = 1



# We will use units of h*nu/c = 1, We can change this or iterate over a list of frequencies later.
photon_momentum = 1

# Total initial momentum in the photons
momentum_i = photon_momentum*N_photons

# Troubleshooting flag
flag = 0

# step counter, used in troubleshooting.
steps = 0

# Define a kernel to quickly multiply and add arrays together.
#############################################################################
# mu -> CUpy array of cosine(theta), where theta is the angle with respect to the z-axis.
# tau -> CUpy array of the z positions of each photon.
# step -> CUpy array of the random step sizes being taken by each photon
#
# newtau -> output as a CUpy array, represents the next z position after taking a step.
#############################################################################
TakeStep = cp.ElementwiseKernel('T mu, T tau, T step','T newtau', 'newtau = tau + mu*step','TakeStep')


def WalkLikeAPhoton(tau, mu, hg, atm, momentum_transfer):
#############################################################################
# A function that takes a CUpy array of photon locations and an atmosphere depth and
# makes steps for the photons until they escape the atmosphere.
#
# tau -> CUpy array of the initial positions of each photon.  This should be all zeroes since all photons start at z = 0.
# mu -> A CUpy array of initial cosine(theta) for each photon.  This should just be 1 since all photons are initially upward moving.
# hg -> A float containing the value of g in the Henyey-Greenley function
# momentum_transfer -> A CUpy array that holds the momentum transferred.
    while len(tau) > 0:
        # global steps
        # steps += 1
        
        # Determine a random step size for each photon
        step = -cp.log(1-cp.random.rand(*mu.shape))
        
        # Have each photon take a step
        newtau = TakeStep(mu,tau,step)
        
        direction = cp.sign(newtau-tau)
        
        tau = newtau
        
        newtau = None
        
        tau[cp.where(tau>=atm)] = cp.NaN

        mask = cp.isnan(tau)
        momentum_transfer[atm_i] -= photon_momentum*float(cp.sum(mu[mask]))
        # if steps>20:#cp.sum(mu[mask])>0:
        #     print(hg)
        #     print(mu)
        #     print(tau)
        #     print(steps)
        #     flag = 1
        #     break
        tau = tau[~mask]
        mu = mu[~mask]
        direction = direction[~mask]

        
        mask = cp.where(tau<0)
        tau[mask] = -tau[mask]
        direction[mask] = -direction[mask]
        momentum_transfer[atm_i] -= photon_momentum*float(cp.sum(mu[mask]))
        
        
        tau[cp.where(cp.random.rand(len(tau))<0.1)] = cp.NaN
        mask = cp.isnan(tau)
        mu = mu[mask]
        momentum_transfer[atm_i] -= photon_momentum*float(cp.sum(mu[cp.where(mu<0)]))
        tau = tau[~mask]
        direction = direction[~mask]
        
        # mu = 2*cp.random.rand(len(tau)) - 1
        
        # decide on a new random direction to scatter in.
        s = 2*cp.random.rand(len(tau)) - 1  
        # This is the Henyey Greenstein relation for anisotropic scattering.
        mu = 1/(2*hg)*(1+hg**2-((1-hg**2)/(1+hg*s))**2)
        mu = mu*direction

    return momentum_transfer

# Loop over all anisotropic Henyey-Greenly parameters.
for hg in g:
    momentum_transfer = momentum_i*cp.ones_like(tau_atm)
    
    for atm_i,atm in enumerate(tau_atm):
        
        tau = cp.zeros(N_photons)
        mu = cp.ones_like(tau)
        
        momentum_transfer = WalkLikeAPhoton(tau, mu, hg, atm, momentum_transfer)
    #     if flag:
    #         break
    # if flag:
    #         break
    plt.plot(cp.asnumpy(cp.log10(tau_atm)),cp.asnumpy(momentum_transfer)/N_photons,label = hg)

b = time.time()

print('time taken: ' + str(b-a))

plt.xlabel(r'$log_{10}(\tau_{atm})$')
plt.ylabel('Momentum transferred per photon')
plt.title('CUpy with reflecting boundary')
plt.legend(title="anisotropic weighting")
plt.savefig('cupy_test.png',dpi=200)


# # Calculate the actual fraction transmitted and the theoretical line for comparison.
# frac_transmitted = 1-N_absorbed/N_photons
# theory = 1/(1+tau_atm/2)

# # Plot data
# plt.plot(cp.asnumpy(cp.log10(tau_atm)), cp.asnumpy(frac_transmitted), label="Simulation data")
# plt.plot(cp.asnumpy(cp.log10(tau_atm)),cp.asnumpy(theory),label = r'$\frac{1}{1+\frac{\tau}{2}}$')
# plt.legend()
# plt.xlabel(r'$log_{10}(\tau_{atm})$')
# plt.ylabel('Fraction of transmitted photons')
# plt.savefig('CUDA_Frac_Transmitted.png',dpi=200)