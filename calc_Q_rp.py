import numpy as np
from hoki import load
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

start_time = time.time()


# specify which BPASS file to use
# BPASS_file = 'spectra-bin-imf135_300.z020.dat'
BPASS_file = 'spectra-bin-imf135_100.z002.dat'

count = 0

# Load the BPASS data
BPASS_data = load.model_output(BPASS_file)

# Convert the BPASS data to microns.  The flux is in luminosity per angstrom, so it is converted by multiplying
# by angstroms/micron (10^4).  The wavelength is in angstrom so it needs to be converted by multiplying by
# microns/angstrom (10^-4), but this is easiest done by converting the whole data set at once then correcting
# the wavelength conversion by another factor of 10^-4.
# BPASS_data *= 10**4
# BPASS_data.WL *= 10**-4

time_list = BPASS_data.columns[BPASS_data.columns != 'WL']

time_list_exp = np.power(10,time_list.astype(float))

delta_t = np.zeros_like(time_list_exp)

for i in range(len(delta_t)):
    if i == 0:
        delta_t[i] = 10**6.05
    else:
        delta_t[i] = 10**(6.15 + 0.1*(i-1))-10**(6.05 + 0.1*(i-1))

def ContinuousStarFormationSED(BPASS_data, time_list_exp, delta_t, age, rate):
    
    SED = BPASS_data[time_list[0]]*delta_t[0]
    
    for j, time_slice in enumerate(time_list_exp[1:]):
        if time_slice <= age:
            SED += BPASS_data[time_list[j]]*delta_t[j]*rate
    
    SED = SED/(10**6)
    
    return SED

# def ContinuousStarFormationSED(BPASS_data, time_list_exp, delta_t, age, rate):
    
#     for j, time_slice in enumerate(time_list_exp[0:]):
#         if time_slice == age:
#             i = j
    
#     SED = BPASS_data[time_list[i]].copy()
    
#     if age != 10**6:
#         for k in range(0,i+1):
#             SED += BPASS_data[time_list[i-k]]*delta_t[k]*rate/10**6
    
#     return SED

def CalculateKappa(data_source, BPASS_data, min_grain_size, max_grain_size):
    # Create a generator for the files to use for Draine.
    # This method will change later.
    Draine_list = os.listdir(data_source)
        
    flag = 0
    
    for k, each in enumerate(Draine_list):
        number = each.split('_')[-2:]
        grain = float(number[0] + '.' + number[1])
        if flag == 0 and grain >= min_grain_size:
            flag = 1
            grain_min_index = k
        if grain >= max_grain_size:
            grain_max_index = k+1
            break
        
    Draine_list = Draine_list[grain_min_index:grain_max_index]
    
    # container for average Q_rp and grain sizes
    Q_rp_av = np.zeros((len(time_list),len(Draine_list)))
    Q_F_av = np.zeros_like(Q_rp_av)
    a = np.zeros(len(Draine_list))
    
    j = 0
    
    for file in Draine_list:
        
        Draine_file = os.path.join(data_source,file)
        
        # print(Draine_file)
        # Import the data from a file for astronomical silicate.
        # [ wavelength (microns) , Q_abs , Q_scatt , g = <cos(theta)> ]
        Draine_data = np.genfromtxt(Draine_file, skip_header=1)
        
        
        
        grain_size = file.split('_')[-2] + '.' +file.split('_')[-1]
        # print(grain_size)
        a[j] = float(grain_size)
        
        Q_rp = np.zeros(len(time_list))
        Q_F = np.zeros_like(Q_rp)
        L_tot = np.zeros_like(Q_rp)
        
        i = 0
    
        # generate a mask to exclude the data in regions not sampled by Draine.
        mask = BPASS_data.WL >= np.min(Draine_data[:,0])
        wavelengths = BPASS_data.WL[mask]
        
        # # If we are looking for the Hydrogen alpha extinction we can use this instead.
        # wavelengths = BPASS_data.WL[6564]
        
        # Use linear interpolation to get Draine values for each of the BPASS wavelengths.
        interpolated_Q_abs = np.interp(wavelengths, np.flip(Draine_data[:,0]), np.flip(Draine_data[:,1]))
        interpolated_Q_scatt = np.interp(wavelengths, np.flip(Draine_data[:,0]), np.flip(Draine_data[:,2]))
        interpolated_g = np.interp(wavelengths, np.flip(Draine_data[:,0]), np.flip(Draine_data[:,3]))
        
        # Calculate the relative contributions to the integrand
        Q_contribution_rp = interpolated_Q_abs + (1 - interpolated_g)*interpolated_Q_scatt
        Q_contribution_F = interpolated_Q_abs + interpolated_Q_scatt
        
        # Calculate the albedo
        albedo = interpolated_Q_scatt/(interpolated_Q_abs+interpolated_Q_scatt)
        
        # Loop over each time slice
        for time_slice in time_list:
    
            # Get the L_lambda term
            L_contribution = BPASS_data[mask][time_slice]
            
            # The known parts of the Q contribution
            # Draine_Q_rp = Draine_data[:,1] + (1 - Draine_data[:,3])*Draine_data[:,2]
            
            # The full integrand is the product of the two contributions
            integrand_rp = Q_contribution_rp * L_contribution
            integrand_F = Q_contribution_F * L_contribution
            
            # The total luminosity
            L = np.sum(L_contribution)
            L_tot[i] = L
            
            Q_rp[i] = np.trapz(integrand_rp)/L
            Q_F[i] = np.trapz(integrand_F)/L
            
            # iterate.
            i += 1
        
        Q_rp_av[:,j] = np.pi * Q_rp*(a[j]**-1.5)
        Q_F_av[:,j] = np.pi * Q_F*(a[j]**-1.5)
        
        j += 1
        
        # plt.plot(time_list_exp,Q_rp, label = grain_size)
        # plt.plot(wavelengths,albedo, label = grain_size)
        # print("a = " + grain_size + ", albedo = " + str(np.average(albedo)))
    
    K_F = np.trapz(Q_F_av,x = a,axis=1)/(np.trapz(4/3*np.pi*a**-0.5*3*10**-12, x=a))*10**-8
    K_rp = np.trapz(Q_rp_av,x = a,axis=1)/(np.trapz(4/3*np.pi*a**-0.5*3*10**-12, x=a))*10**-8
    
    
    return K_F, K_rp, albedo, L_tot

def CalculateHaExtinction(data_source, min_grain_size, max_grain_size):
    
    # Define the hydrogen alpha line
    Ha_lambda = 0.6565
    
    # Get a list of all files in the directory.
    Draine_list = os.listdir(data_source)
    
    # Set a flag for use in the next loop.
    flag = 0
    
    # Use the minimum and maximum grain size to find all the relevant files.
    for k, each in enumerate(Draine_list):
        number = each.split('_')[-2:]
        grain = float(number[0] + '.' + number[1])
        if flag == 0 and grain >= min_grain_size:
            flag = 1
            grain_min_index = k
        if grain >= max_grain_size:
            grain_max_index = k+1
            break
        
    # Trim out only the grain sizes not included.
    Draine_list = Draine_list[grain_min_index:grain_max_index]
    
    # Initialize some empty lists.    
    numerator = np.zeros(len(Draine_list))
    denominator = np.zeros_like(numerator)
    a = np.zeros(len(Draine_list))
    
    # Loop over each file.
    for j, file in enumerate(Draine_list):
        
        # Get the file path and name.
        Draine_file = os.path.join(data_source,file)
        
        # Find the grain size and store it in an array.
        grain_size = file.split('_')[-2] + '.' + file.split('_')[-1]
        a[j] = float(grain_size)
        
        # Import the data from a file for astronomical silicate.
        # [ wavelength (microns) , Q_abs , Q_scatt , g = <cos(theta)> ]
        Draine_data = np.genfromtxt(Draine_file, skip_header=1)
        
        # Interpolate the values of Q_abs and Q_scatt at the hydrogen alpha line from the Draine file.
        Q_abs = np.interp(Ha_lambda, np.flip(Draine_data[:,0]), np.flip(Draine_data[:,1]))
        Q_scatt = np.interp(Ha_lambda, np.flip(Draine_data[:,0]), np.flip(Draine_data[:,2]))
        
        # Get the numerator and denominator.
        numerator[j] = np.pi * a[j]**(-1.5) * (Q_abs + Q_scatt)
        denominator[j] = 4/3 * np.pi * a[j]**(-0.5) * 3 * 10**-12
    
    # Integrate the numerator and denominator.
    Extinction = np.trapz(numerator, x = a)/np.trapz(denominator, x = a) * 10**-8
    
    return Extinction



## Generate photon counts
# -----------------------------------------------------

# BPASS_data = BPASS_data.iloc[::100,:]

# # BPASS_data.WL *= 10**-4

# h = 6.6261*(10**(-27))

# c = 2.99792458*(10**10)

# # BPASS_data.iloc[:,1:-1] *= 10**8

# photon = BPASS_data['6.0'] * BPASS_data.WL**2 / (h*c)

# plt.plot(BPASS_data.WL, photon)
# plt.title('Photon Count by Wavelength')
# plt.xlabel('Wavelength (cm)')
# plt.ylabel('photon / s')
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(1)
# plt.xlim(5*10**-7,10**-3)
# plt.savefig('Photon counts.png', DPI = 200)
# -----------------------------------------------------
    
# ## Find CDF
# # -----------------------------------------------------

# norm = np.trapz(photon, x = BPASS_data.WL)
# CDF = np.zeros_like(BPASS_data.WL)

# for i, _ in enumerate(BPASS_data.WL):
#     phot = photon[0:i]
#     WL = BPASS_data.WL[0:i]
#     CDF[i] = np.trapz(phot, x = WL) / norm

# # plt.plot(BPASS_data.WL, CDF)
# # plt.ylabel('Continuous Distribution')
# # plt.xlabel('Wavelength (micron)')
# # plt.title('CDF by wavelength')
# # plt.savefig('CDF.png', dpi = 200)

# # -----------------------------------------------------



# ## Generate PDF from CDF.
# # -----------------------------------------------------

# PDF = np.gradient(CDF)

# plt.plot(BPASS_data.WL*10**-8,PDF)
# plt.xscale('log')
# plt.ylabel('Probability')
# plt.xlabel('Wavelength (cm)')
# plt.title('PDF for BPASS SED at 1 MYr')
# plt.savefig('PDF.png', dpi = 200)


# # -----------------------------------------------------

# ## Verify the Accept/Reject function works
# # -----------------------------------------------------
# fig, ax1 = plt.subplots()

# color = 'tab:blue'
# ax1.hist(x, bins = BPASS_data.WL.to_numpy())
# ax1.set_xlabel('Wavelength (angstroms)')
# ax1.set_ylabel('Accept/Reject results')
# ax1.xscale('log')
# ax1.yscale('log')
# ax1.tick_params(axis='y', labelcolor = color)

# ax2 = ax1.twinx()

# color = 'tab:red'
# ax2.plot(BPASS_data.WL, PDF, color = 'r')
# ax2.set_ylabel('PDF')
# ax2.xscale('log')
# ax2.yscale('log')
# ax2.tick_params(axis='y', labelcolor = color)


# fig.tight_layout()

# plt.savefig('Accept-Reject verification.png', dpi = 200)
# #------------------------------------------------------


# max_grain_size = 1.0
# min_grain_size = 0.001

# data_source = 'Draine data Gra/'

# Extinction = CalculateHaExtinction(data_source, min_grain_size, max_grain_size)

# print(Extinction)

# x = np.linspace(10**6,10**10,10000)

# time_labels = np.log10(x).astype('str')

# for i in range(len(BPASS_data.WL.iloc[::10])):
#     interp = np.interp(x, time_list_exp, BPASS_data.iloc[i*10][1:].values).reshape(1,-1)
    
#     if i == 0:
#         BPASS_int = pd.DataFrame(interp, columns = time_labels)
#     else:
#         BPASS_int = BPASS_int.append(pd.DataFrame(interp, columns = time_labels))

# BPASS_int *= 10**4

# continuous_SFR = BPASS_int.cumsum(axis=1)

# continuous_SFR.insert(0, 'WL', BPASS_data.WL.values[::10])

# plt.plot(BPASS_int.WL, CSFR_SED.iloc[:,-1], label = 'Continuous SFR')
# plt.plot(BPASS_data.WL, BPASS_data.iloc[:,41]*10**4, label = 'instantaneous')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.002,10)
# plt.ylim(0.00001)
# plt.legend(loc = 4)
# plt.title('Comparison of continuous vs instantaneous SFR SED at 10 GYr')
# plt.xlabel('Wavelength (microns)')
# plt.ylabel('Flux (Solar luminosity per micron)')
# plt.savefig('interpolated SED comparison zoomed.png', dpi = 200)

# age_list = np.logspace(6, 10, num = 50, base=10.0)

# rate_list = [1]

# for rate in rate_list:
#     for i, age in enumerate(time_list_exp):
        
#         SED = ContinuousStarFormationSED(BPASS_data, time_list_exp, delta_t, age, rate)
        
#         if age == 10**6:
#             continuous_SFR = SED
#         else:
#             continuous_SFR = pd.concat([continuous_SFR, SED.rename(time_list[i])], axis=1)
    
#     continuous_SFR.insert(0, 'WL', BPASS_data.WL)
    
#     mass = delta_t.copy()
#     mass[1:] *= rate
#     mass = np.cumsum(mass)
    
#     # age_list = np.insert(age_list, [0], 10**6)
    
#     # plt.plot(time_list_exp, continuous_SFR.sum(axis = 0)[1:]/mass, label = str(rate) + ' solar M/year')
#     # plt.plot(time_list_exp, continuous_SFR.sum(axis = 0)[1:], label = str(rate) + ' solar M/year')


max_grain_size = 1.0
min_grain_size = 0.001

# data_source = 'Draine data SiC/'
# K_F_SiC, K_rp_SiC, albedo_SiC, L_CSFR = CalculateKappa(data_source, continuous_SFR, min_grain_size, max_grain_size)
# data_source = 'Draine data Sil/'
# K_F_Sil, K_rp_Sil, albedo_Sil, L = CalculateKappa(data_source, continuous_SFR, min_grain_size, max_grain_size)
# data_source = 'Draine data Gra/'
# K_F_Gra, K_rp_Gra, albedo_Gra, L = CalculateKappa(data_source, continuous_SFR, min_grain_size, max_grain_size)


# data_source = 'Draine data SiC/'
# K_F_SiC2, K_rp_SiC2, albedo_SiC, L = CalculateKappa(data_source, BPASS_data, min_grain_size, max_grain_size)
# data_source = 'Draine data Sil/'
# K_F_Sil, K_rp_Sil, albedo_Sil, L = CalculateKappa(data_source, BPASS_data, min_grain_size, max_grain_size)
# data_source = 'Draine data Gra/'
# K_F_Gra2, K_rp_Gra2, albedo_Gra, L = CalculateKappa(data_source, BPASS_data, min_grain_size, max_grain_size)

# plt.plot(time_list_exp, BPASS_data.iloc[:,1:].sum(axis=0), label = 'BPASS data')
# # plt.plot(time_list_exp, L, label = 'L verified method')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(10**5.9, 10**10)
# plt.xlabel('Time (years)')
# plt.ylabel('L (solar luminosity per angstrom)')
# plt.title('Luminosity for continuous star formation')
# plt.legend()
# plt.tight_layout()
# plt.savefig('L for continuous SFR method 2.png', dpi = 200)

# plt.plot(BPASS_data.WL,SED)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.001,10)
# plt.ylim(1,10**14)
# plt.title(r'SED for a continuously forming binary population at $10^8$ years')
# plt.xlabel('wavelength (microns)')
# plt.ylabel('Flux (Solar luminosity per angstrom)')
# plt.savefig('SED for continuous formation.png', dpi=200)

# max_grain_size = 1.0
# min_grain_size = 0.001

# data_source = 'Draine data SiC/'
# K_F_SiC, K_rp_SiC, albedo_SiC, L = CalculateKappa(data_source, BPASS_data, min_grain_size, max_grain_size)
# data_source = 'Draine data Sil/'
K_F_Sil, K_rp_Sil, albedo_Sil, L = CalculateKappa(data_source, BPASS_data, min_grain_size, max_grain_size)
# data_source = 'Draine data Gra/'
# K_F_Gra, K_rp_Gra, albedo_Gra, L = CalculateKappa(data_source, BPASS_data, min_grain_size, max_grain_size)

# L_edd_SiC = 2001*4*np.pi*150 / K_rp_SiC * 2/3.84
L_edd_Sil = 2001*4*np.pi*150 / K_rp_Sil * 2/3.84
# L_edd_Gra = 2001*4*np.pi*150 / K_rp_Gra * 2/3.84


# Set the ratios of the 3 materials.  First guess is that it is 50% silicate, an aditional amount of silicon
# is present in the form of silicon-Carbide, equal to 5% of the silicate.  The rest is graphene.
# f_Sil = 0.5
# f_SiC = f_Sil*0.05
# f_Gra = 1 - f_Sil - f_SiC

# kappa_rp = f_Sil*K_rp_Sil + f_SiC*K_rp_SiC + f_Gra*K_rp_Gra
# kappa_F = f_Sil*K_F_Sil + f_SiC*K_F_SiC + f_Gra*K_F_Gra

# plt.plot(time_list_exp, kappa_rp, 'navy', label='CSFR RP')
# plt.plot(time_list_exp, kappa_F, 'blue', label='CSFR F')

# kappa_rp2 = f_Sil*K_rp_Sil2 + f_SiC*K_rp_SiC2 + f_Gra*K_rp_Gra2
# kappa_F2 = f_Sil*K_F_Sil2 + f_SiC*K_F_SiC2 + f_Gra*K_F_Gra2

# plt.plot(time_list_exp, kappa_rp2, 'red', label='Instantaneous RP')
# plt.plot(time_list_exp, kappa_F2, 'orange', label='Instantaneous F')

# plt.plot(time_list_exp,K_rp_Gra, 'gold', label = 'Gra RP')
# plt.plot(time_list_exp,K_F_Gra, 'yellow', label = 'Gra F')

# plt.plot(time_list_exp,K_rp_Sil, 'green', label = 'Sil RP')
# plt.plot(time_list_exp,K_F_Sil, 'limegreen', label = 'Sil F')

# plt.plot(time_list_exp,K_rp_SiC, 'darkred', label = 'SiC RP')
# plt.plot(time_list_exp,K_F_SiC, 'red', label = 'SiC F')

# plt.xlim(10**6,10**10)
# plt.xscale('log')
# plt.xlabel('Time (years)')
# plt.yscale('log')
# plt.legend()
# plt.ylabel(r'$\langle \kappa \rangle$')
# plt.title(r'$\langle \kappa \rangle$ for grains from ' + str(min_grain_size) + ' micron to ' + str(max_grain_size) + ' micron for a continuous SFR.')
# plt.tight_layout()
# plt.savefig('Kappa all for dust mixture cont SFR.png', dpi = 200)


# # These are common to all Draine files so I'm just hardcoding them.
# grain_sizes = [1.259e-03, 1.413e-03, 1.585e-03, 1.778e-03,
#         1.995e-03, 2.239e-03, 2.512e-03, 2.818e-03, 3.162e-03, 3.548e-03,
#         3.981e-03, 4.467e-03, 5.012e-03, 5.623e-03, 6.310e-03, 7.079e-03,
#         7.943e-03, 8.913e-03, 1.000e-02, 1.122e-02, 1.259e-02, 1.413e-02,
#         1.585e-02, 1.778e-02, 1.995e-02, 2.239e-02, 2.512e-02, 2.818e-02,
#         3.162e-02, 3.548e-02, 3.981e-02, 4.467e-02, 5.012e-02, 5.623e-02,
#         6.310e-02, 7.079e-02, 7.943e-02, 8.913e-02, 1.000e-01, 1.122e-01,
#         1.259e-01, 1.413e-01, 1.585e-01, 1.778e-01, 1.995e-01, 2.239e-01,
#         2.512e-01, 2.818e-01, 3.162e-01, 3.548e-01, 3.981e-01, 4.467e-01,
#         5.012e-01, 5.623e-01, 6.310e-01, 7.079e-01, 7.943e-01, 8.913e-01,
#         1.000e+00, 1.122e+00, 1.259e+00, 1.413e+00, 1.585e+00,
#         1.778e+00, 1.995e+00, 2.239e+00, 2.512e+00, 2.818e+00, 3.162e+00,
#         3.548e+00, 3.981e+00, 4.467e+00, 5.012e+00, 5.623e+00, 6.310e+00,
#         7.079e+00, 7.943e+00, 8.913e+00]

# SiC_rp_data = np.zeros_like(grain_sizes)
# Sil_rp_data = np.zeros_like(grain_sizes)
# Gra_rp_data = np.zeros_like(grain_sizes)
# SiC_F_data = np.zeros_like(grain_sizes)
# Sil_F_data = np.zeros_like(grain_sizes)
# Gra_F_data = np.zeros_like(grain_sizes)

# for z, size in enumerate(grain_sizes[0:-3]):
#     max_grain_size = 8.913e+00
#     min_grain_size = size
    
#     data_source = 'Draine data SiC/'
#     K_F_SiC, K_rp_SiC, albedo_SiC, L = CalculateKappa(data_source, BPASS_data, min_grain_size, max_grain_size)
#     data_source = 'Draine data Sil/'
#     K_F_Sil, K_rp_Sil, albedo_Sil, L = CalculateKappa(data_source, BPASS_data, min_grain_size, max_grain_size)
#     data_source = 'Draine data Gra/'
#     K_F_Gra, K_rp_Gra, albedo_Gra, L = CalculateKappa(data_source, BPASS_data, min_grain_size, max_grain_size)
#     # data_source = 'Draine data SUV/'
#     # K_F_SUV, K_rp_SUV, albedo_SUV, time_list_exp, L = CalculateKappa(data_source, BPASS_data, max_grain_size)
    
#     L_edd_SiC = 2001*4*np.pi*150 / K_rp_SiC * 2/3.84
#     L_edd_Sil = 2001*4*np.pi*150 / K_rp_Sil * 2/3.84
#     L_edd_Gra = 2001*4*np.pi*150 / K_rp_Gra * 2/3.84
    
#     SiC_rp_data[z] = K_rp_SiC[0]
#     Sil_rp_data[z] = K_rp_Sil[0]
#     Gra_rp_data[z] = K_rp_Gra[0]
#     SiC_F_data[z] = K_F_SiC[0]
#     Sil_F_data[z] = K_F_Sil[0]
#     Gra_F_data[z] = K_F_Gra[0]
    
#     count += 1
#     print(count)

# stop_time = time.time()

# print(stop_time-start_time)

# # slope = 1/np.sqrt(grain_sizes)

# plt.plot(grain_sizes,SiC_rp_data, label = 'SiC_rp')
# plt.plot(grain_sizes,Sil_rp_data, label = 'Sil_rp')
# plt.plot(grain_sizes,Gra_rp_data, label = 'Gra_rp')
# plt.plot(grain_sizes,SiC_F_data, label = 'SiC_F')
# plt.plot(grain_sizes,Sil_F_data, label = 'Sil_F')
# plt.plot(grain_sizes,Gra_F_data, label = 'Gra_F')
# # plt.plot(grain_sizes, 95000*slope, 'k--')
# plt.yscale('log')
# plt.xscale('log')
# plt.xlim(0.001,1)
# plt.title('Scaling check at 1 MYr')
# plt.xlabel(r'$a_{min} \, (\mu$m)')
# plt.ylabel(r'$\langle \kappa \rangle$')
# plt.legend()
# plt.tight_layout()
# plt.savefig('Scaling check for amin.png', dpi = 200)

# plt.plot(time_list_exp,SiC_1_0/SiC_0_1, label = '1.0/0.1')
# plt.xscale('log')
# plt.xlabel('time (years)')
# plt.title(r'Scaling check for SiC, $a_{max}=1.0$/$a_{max}=0.1$')
# plt.xlim(10**6,10**10)
# plt.tight_layout()
# plt.savefig('Scaling check.png', dpi = 200)


# V_esc_SiC = 10*np.sqrt((L/10**6)/L_edd_SiC -1)
# V_esc_Gra = 10*np.sqrt((L/10**6)/L_edd_Gra -1)
# V_esc_Sil = 10*np.sqrt((L/10**6)/L_edd_Sil -1)

# plt.plot(time_list_exp, V_esc_Gra, label = 'Gra')
# plt.plot(time_list_exp, V_esc_Sil, label = 'Sil')
# plt.plot(time_list_exp, V_esc_SiC, label = 'SiC')
# plt.xscale('log')
# plt.xlabel('time (years)')
# plt.title(r'$V_{inf}$ over time for grain sizes ' + str(min_grain_size) + ' micron to ' + str(max_grain_size) + ' micron, $V_{esc}$ = 10 km/s')
# plt.xlim(10**6,2.5*10**7)
# plt.legend()
# plt.ylabel('km/s')
# plt.tight_layout()
# plt.savefig('V_inf.png', dpi = 200)


# plt.plot(time_list_exp,Q_rp_av)
# plt.xscale('log')
# plt.xlabel('time (years)')
# plt.title(r'$\langle\frac{Q_{rp}}{a}\rangle$ over time')
# plt.xlim(10**6,10**10)
# plt.ylabel(r'$\langle\frac{Q_{rp}}{a}\rangle$')
# plt.savefig('Q_rp_av over a over time.png', dpi = 200)

# plt.plot(time_list_exp,Q_rp_av_int)
# plt.xscale('log')
# plt.xlabel('time (years)')
# plt.yscale('log')
# plt.title(r'$\langle \kappa_F \rangle$ over time for grain sizes 0.00035 micron to 0.001 micron')
# plt.xlim(10**6,10**10)
# plt.ylabel(r'$\langle \kappa_F \rangle (cm^2/g)$')
# plt.savefig('kappa_F over time.png', dpi = 200)

# plt.plot(time_list_exp,K_rp_Gra, label = 'Gra')
# plt.plot(time_list_exp,K_rp_Sil, label = 'Sil')
# plt.plot(time_list_exp,K_rp_SiC, label = 'SiC')
# plt.plot(time_list_exp,K_rp_SUV, label = 'SUV')
# plt.xscale('log')
# plt.xlabel('time (years)')
# plt.yscale('log')
# plt.title(r'$\langle \kappa_{rp} \rangle$ over time for grain sizes 0.001 micron to ' + str(max_grain_size) + ' micron')
# plt.xlim(10**6,10**10)
# plt.legend()
# plt.ylabel(r'$\langle \kappa_{rp} \rangle (cm^2/g)$')
# plt.tight_layout()
# plt.savefig('kappa_rp gra vs sil vs SiC comparison over time.png', dpi = 200)

# plt.plot(time_list_exp,K_F_Gra, label = 'Gra')
# plt.plot(time_list_exp,K_F_Sil, label = 'Sil')
# plt.plot(time_list_exp,K_F_SiC, label = 'SiC')
# # plt.plot(time_list_exp,K_F_SUV, label = 'SUV')
# plt.xscale('log')
# plt.xlabel('time (years)')
# plt.yscale('log')
# plt.title(r'$\langle \kappa_{F} \rangle$ over time for grain sizes ' + str(min_grain_size) + ' micron to ' + str(max_grain_size) + ' micron')
# plt.xlim(10**6,10**10)
# plt.legend()
# plt.ylabel(r'$\langle \kappa_{F} \rangle (cm^2/g)$')
# plt.tight_layout()
# plt.savefig('kappa_F gra vs sil vs SiC comparison over time.png', dpi = 200)

# plt.plot(time_list_exp, L_edd_Gra, label = 'Gra')
plt.plot(time_list_exp, L_edd_Sil/10**6, label = r'$L_{Edd}/M (Sil)$')
# plt.plot(time_list_exp, L_edd_SiC, label = 'SiC')
plt.plot(time_list_exp, L/10**6, 'k--', label = 'L/M (BPASS)')
plt.xscale('log')
plt.xlabel('Time (years)')
plt.yscale('log')
plt.title(r'$\langle L_{edd}/M \rangle$ over time for grain sizes ' + str(min_grain_size) + ' micron to ' + str(max_grain_size) + ' micron')
# plt.xlim(10**6,10**10)
# plt.ylim(10,2000)
plt.legend()
plt.ylabel(r'$ L/M \; (L_{\odot}/M_{\odot}) $')
plt.tight_layout()
plt.savefig('L_edd over M.png', dpi = 200)

# plt.plot(time_100,K_100, label='0.02', linewidth=0.5)
# plt.plot(time_list_exp,Q_rp_av_int, label='0.002', linewidth=0.5)
# plt.xscale('log')
# plt.xlabel('time (years)')
# plt.yscale('log')
# plt.title(r'$\langle \kappa_F \rangle$ over time for grain sizes 0.001 micron to 0.2512 micron')
# plt.xlim(10**6,10**10)
# plt.ylabel(r'$\langle \kappa_F \rangle (cm^2/g)$')
# plt.legend(title = 'z')
# plt.savefig('z 02 vs 002 comparison.png', dpi = 200)


# plt.ylabel(r'$Q_{rp}$')
# plt.xlabel('time (years)')
# plt.xscale('log')
# plt.xlim(10**6,10**10)
# plt.legend(title = "a", ncol = 2, loc = 7)
# plt.title(r'$Q_{rp}$ over time')
# plt.savefig('Q_rp late universe cropped.png', dpi = 200)

# plt.plot(a,np.pi * (10**-3)**3.5 * (a**-0.5)*Q_rp_av[0,:])
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('fig 1.png', dpi=200)


    
# plt.plot(x, BPASS_int.sum(axis=0)[1:]/(10*x), label = 'CSFR')
# plt.plot(time_list_exp, BPASS_data.sum(axis = 0)[1:]/10**6, label = 'Instantaneous')
# plt.plot(x, np.cumsum(interp*10)/x, label = 'linear bin')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('time (years)')
# plt.ylabel('L/M (solar luminosity per solar mass)')
# plt.xlim(10**6,10**10)
# plt.ylim(10)
# plt.legend()
# plt.title(r'L/M over time for continuous SFR, 1 $M_{\odot}$/Yr')
# plt.savefig('L over M for CSFR.png', dpi = 200)



# ## Plot Ha luminosity
# ## ---------------------------------------------------------------

time_list = BPASS_data.columns[BPASS_data.columns != 'WL']

time_list_exp = np.power(10,time_list.astype(float))

Bol = BPASS_data[ BPASS_data.columns[BPASS_data.columns != 'WL'] ].sum()
Ha = BPASS_data[ BPASS_data.columns[BPASS_data.columns != 'WL'] ].iloc[6564]

plt.plot(time_list_exp, Ha, label= r'H$\alpha$')
plt.plot(time_list_exp, Bol, label = 'Bolometric')
plt.title('luminosity over time')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Luminosity (solar luminosity / angstrom)')
plt.xlabel('Time (years)')
plt.legend()
plt.savefig('L over time.png', dpi = 200)
# ## ----------------------------------------------------------------


# ## Plot Bol / Ha luminosity
# ## ---------------------------------------------------------------

time_list = BPASS_data.columns[BPASS_data.columns != 'WL']

time_list_exp = np.power(10,time_list.astype(float))

Ha = BPASS_data[ BPASS_data.columns[BPASS_data.columns != 'WL'] ].iloc[6564]

Bol = BPASS_data[ BPASS_data.columns[BPASS_data.columns != 'WL'] ].sum()

plt.plot(time_list_exp, Bol/Ha)
plt.title(r'Bolometric Luminosity / H$\alpha$ luminosity over time')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$L_{bol}/L_{H\alpha}$')
plt.xlabel('Time (years)')
plt.savefig('Bol over Ha.png', dpi = 200)
# ## ----------------------------------------------------------------