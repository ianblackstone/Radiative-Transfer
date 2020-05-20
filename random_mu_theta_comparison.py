import numpy as np
import matplotlib.pyplot as plt


# Generate random values in the distribution
ran_mu = 2*np.random.rand(500)-1
ran_theta = np.pi*np.random.rand(500)

# generate linearly spaced for comparison
# ran_mu = np.linspace(-1,1,100)
# ran_theta = np.linspace(0,cp.pi,100)

# Need the angles directly.
ran_mu_theta = np.arccos(ran_mu)


# Choose a radius
r = 10

# Select which one to plot
ran = ran_theta

plt.figure(1)
# Plot each line from [0,0] to the endpoint.
for theta in ran:
    x = [0,r * np.cos(theta).tolist()]
    y = [0,r * np.sin(theta).tolist()]
    plt.plot(x,y,color='green')
    plt.xlim(-10,10)
    plt.ylim(0,10)

plt.figure(2)
plt.hist(ran,bins='auto')
plt.xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi],labels=['0',r"$\frac{\pi}{4}$",r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$'])