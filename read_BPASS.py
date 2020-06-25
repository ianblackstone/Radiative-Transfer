import pandas as pd
import matplotlib.pyplot as plt


row = 9000
column = 10

labels = ["Wavelength"]

for n in range(1,52,1):
    labels.append(10**(6+0.1*(n-2)))

data_file = pd.read_csv("spectra-bin-imf135_300.z020.dat", header = None, delimiter="   ", names = labels, engine='python')

data_file.Wavelength *= 10**-4

# plot_data = data_file.iloc[row][1:-1]

# plot_data.plot(logx=True)
# plt.title("Flux by year for wavelength " + str(data_file.iloc[row][0]) + " microns")
# plt.ylabel('flux (solar luminosity per micron)')
# plt.xlabel('log year')

# data_file[data_file['Wavelength'].between(0.01,10)].plot(x=labels[0], y=labels[column], legend=False, loglog=True)
# plt.title('Spectral distribution at time t = ' + f"{labels[column]:.2f}" + ' years')
# plt.xlabel('log Wavelength (microns)')
# plt.ylabel('Log Flux (Solar luminosity per angstrom)')

integral = data_file.sum(axis=0)*10**-10

integral[1:-1].plot(loglog=True)
plt.ylabel('log total luminosity (solar luminosity)')
plt.xlabel('log Age (years)')
plt.title('Total luminosity by age')