import numpy as np

# atmosphere depth
tau_atm = 10

# Number of bins in the mu and phi directions.
# mu ranges from 0 to 1 and phi ranges from 0 to 2pi 
N_mu = 100
N_phi = 100

# photons starts at z = 0, moving upward
tau_i = [0,0,0]
mu_i = 0
phi_i = 0

# The number of photons to simulate
N_photons = 100000

# Counter for the number of photons that get absorbed
N_absorbed = 0

def TakeStep(tau,mu,phi):
    # Travel some amount in the direction it was heading in.
    tau[0] += mu*np.cos(phi)
    tau[1] += mu*np.sin(phi)
    tau[2] += mu

    return tau 

    
def Scatter():
    # scatter in a random direction.
    mu = 2*np.random.rand() - 1
    phi = 2*np.pi*np.random.rand()
    
    return mu, phi

#  See if the photon escaped or was reabsorbed.
def CheckStep(tau):
    if tau[2] >= tau_atm:
        return 1
    if tau[2] < -1:
        global N_absorbed
        N_absorbed += 1
        return 0
    return 2

step_count = 0


for i in range(N_photons):
    tau = tau_i[:]
    mu = mu_i
    phi = phi_i
    
    i += 1
    
    while 1:
        tau = TakeStep(tau,mu,phi)
        mu,phi = Scatter()
        
        step_count += 1
        
        escape = CheckStep(tau)
        
        if escape == 1:
            # print('escaped')
            break
        
        elif escape == 0:
            # print('absorbed')
            break
        
        else:
            pass