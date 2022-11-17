import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from hoki import load
import numpy as np
import os


# def GetKappa(folder, grain_min, grain_max, wl_list, RPF):

    
    
#     Draine_list = os.listdir(folder)
#     N_grains = len(Draine_list)
#     N_wl = len(wl_list)
    
#     Q_ext = np.zeros((N_grains,N_wl))
#     a = np.zeros(N_grains)
    
#     for i, file in enumerate(Draine_list):
#         Draine_data = pd.read_csv(folder + file, delim_whitespace=True)
#         Draine_data.columns = ['wl', 'Q_abs', 'Q_sca', 'g']
        
#         Q_abs = np.interp(wl_list, np.flip(Draine_data.wl.to_numpy()), np.flip(Draine_data.Q_abs.to_numpy()))
#         Q_sca = np.interp(wl_list, np.flip(Draine_data.wl.to_numpy()), np.flip(Draine_data.Q_sca.to_numpy()))
#         g = np.interp(wl_list, np.flip(Draine_data.wl.to_numpy()), np.flip(Draine_data.g.to_numpy()))
        
#         if RPF == 'F':
#             Q_ext[i,:] = Q_abs + Q_sca
#         else:
#             Q_ext[i,:] = Q_abs + (1-g)*Q_sca
        
#         grain_size = file.split('_')[-2] + '.' +file.split('_')[-1]
#         a[i] = float(grain_size)
    

#     grain_mask = (a <= grain_max) & (a >= grain_min)
    
#     a = a[grain_mask]
#     Q_ext = Q_ext[grain_mask,:]
    
#     density = 3*10**-4
    
#     kappa = np.trapz(Q_ext * a[:, None]**-1.5 * np.pi, x = a, axis = 0) / np.trapz(4*np.pi/3 * a**-0.5 * density, x = a , axis = 0)
    
#     return kappa, a

def GetGrainData(grain_type, grain_min, grain_max, BPASS_data_r, time_slice, wl_ref):
    Grain_File = grain_type + ' Average Values.csv'
    folder = 'Draine data ' + grain_type + '/'
    kappa_data, _, _ = GetTauScaling(folder, grain_min, grain_max, BPASS_data_r.WL.to_numpy(), BPASS_data_r, time_slice, wl_ref)
    Grain_data = pd.read_csv(Grain_File)

    return Grain_data, kappa_data

def GetKappas(folder, grain_min, grain_max, wl_list, Spectra, time_slice):
    ##########################################################################
    # This function calculates the various kappas and returns each of them as
    # numpy array.  It also returns the total luminosity and grain size.
    #
    # folder = string pointing to the folder where the Draine data is stored.
    #       Each Draine dataset is split into 81 files, one for each grain size
    #       with a name in the format draine_data_Sil_0_001.
    # grain_min - float giving the minimum grain size to consider.
    # m_max - float giving the maximum grain size to consider.
    # wl_list - numpy list containing all the wavelengths to be used in microns.
    # BPASS_data - a pandas dataframe containing the BPASS data.
    # time_slice - String containing which time slice to use ('6.0', '7.8', etc)
    ##########################################################################
    
    # Read the list of files in the directory for the Draine data.
    # Then find how many grain sizes are in that folder (should be 81 but for the
    # sake of flexibility it's done programatically).  Then get the length of the
    # wavelength list.
    Draine_list = os.listdir(folder)
    N_grains = len(Draine_list)
    N_wl = len(wl_list)
    
    # Create 3 arrays filled with zero to hold the Q_ext and grain size data.
    Q_ext_F = np.zeros((N_grains,N_wl))
    Q_ext_RP = np.zeros_like(Q_ext_F)
    a = np.zeros(N_grains)
    
    # Go through the list of Draine files, enumerate() creates an index matching
    # that file's position in the list
    for i, file in enumerate(Draine_list):
        
        # Read the data from each file into a pandas dataframe.
        Draine_data = pd.read_csv(folder + file, delim_whitespace=True)
        # The headers in the file are hard to reference as written so we rename the columns
        Draine_data.columns = ['wl', 'Q_abs', 'Q_sca', 'g']
        
        # Interpolate the Draine data at the points in wl_list.  np.flip is used to put the data in ascending order
        Q_abs = np.interp(wl_list, np.flip(Draine_data.wl.to_numpy()), np.flip(Draine_data.Q_abs.to_numpy()))
        Q_sca = np.interp(wl_list, np.flip(Draine_data.wl.to_numpy()), np.flip(Draine_data.Q_sca.to_numpy()))
        g = np.interp(wl_list, np.flip(Draine_data.wl.to_numpy()), np.flip(Draine_data.g.to_numpy()))
        
        # Calculate the Q_ext for all wavelengths for this grain size.
        Q_ext_F[i,:] = Q_abs + Q_sca
        Q_ext_RP[i,:] = Q_abs + (1-g)*Q_sca
        
        # # Set Q_ext = Q_abs for the purposes of doing absoprtion only
        # Q_ext_F[i,:] = Q_abs
        # Q_ext_RP[i,:] = Q_abs
        
        # The grain size is stored only in the file name, so we read the name,
        # split it into ['draine','data','0','001'] and take the last two entries
        # then we convert that string to a float and store it.
        grain_size = file.split('_')[-2] + '.' +file.split('_')[-1]
        a[i] = float(grain_size)
    
    # Since we didn't have a list of all the grain sizes before now we couldn't
    # avoid interpolating all grain sizes (it's fast anyways).  So now we define
    # the boolean mask to filter data to being inside the grain size range.
    grain_mask = (a <= grain_max) & (a >= grain_min)
    
    a = a[grain_mask]
    Q_ext_F = Q_ext_F[grain_mask,:]
    Q_ext_RP = Q_ext_RP[grain_mask,:]
    
    # Extract the flux from the BPASS data.
    if str(time_slice) in Spectra.columns:
        L_integrand = Spectra[str(time_slice)].to_numpy()
    else:
        new_column = np.empty(len(Spectra.WL))
        new_column[:] = np.NaN
        Spectra = Spectra.drop('WL', axis = 1)
        Spectra.insert(1, str(time_slice), new_column)
        Spectra.columns = Spectra.columns.astype(float)
        Spectra.sort_index(axis=1, inplace=True)
        Spectra.interpolate(axis = 1, inplace = True)
        L_integrand = np.interp(time_slice, )
    # print(Spectra.shape)
    
    # get the total luminosity/angstrom (units will not matter)
    L = np.sum(L_integrand)
    
    # Do the first integral for the averaged values.  The [None,:] indexing needs
    # to be done to make Q_ext and L_integrand both be vertical arrays.
    # print('Q_ext_F: {}, L_int: {}'.format(np.shape(Q_ext_F),np.shape(L_integrand[None,:])))
    Q_ext_int_F = np.trapz(Q_ext_F * L_integrand[None,:], x = wl_list, axis = 1)*10**4
    Q_ext_int_RP = np.trapz(Q_ext_RP * L_integrand[None,:], x = wl_list, axis = 1)*10**4
    
    # Define the density.  This is in units of g/(cm^2 * um).  This is identical
    # to using g/um^3 then converting to cm^2/g at the end.
    if "Sil" in folder:
        density = 3.3*10**-4
    elif "Gra" in folder:
        density = 2.26*10**-4
    elif "SiC" in folder:
        density = 3.22*10**-4
    else:
        Exception("GetKappas() | Gain Type for {} not Recognized.".format(folder))
        quit
    
    # If we only want one grain size we don't do an integral so we have an if here.
    if len(a) > 1:
        # Use the trapezoidal method to integrate the 4 kappas over grain size.
        kappa_av_RP = np.trapz(Q_ext_int_RP * a**-1.5 * np.pi, x = a) / np.trapz(4*np.pi/3 * a**-0.5 * density, x = a) / L
        kappa_av_F = np.trapz(Q_ext_int_F * a**-1.5 * np.pi, x = a) / np.trapz(4*np.pi/3 * a**-0.5 * density, x = a) / L
        kappa_RP = np.trapz(Q_ext_RP * a[:, None]**-1.5 * np.pi, x = a, axis = 0) / np.trapz(4*np.pi/3 * a**-0.5 * density, x = a)
        kappa_F = np.trapz(Q_ext_F * a[:, None]**-1.5 * np.pi, x = a, axis = 0) / np.trapz(4*np.pi/3 * a**-0.5 * density, x = a)
    else:
        kappa_av_RP = (Q_ext_int_RP * a**-1.5 * np.pi) / (4*np.pi/3 * a**-0.5 * density) / L
        kappa_av_F = (Q_ext_int_F * a**-1.5 * np.pi) / (4*np.pi/3 * a**-0.5 * density) / L
        kappa_RP = (Q_ext_RP * a**-1.5 * np.pi) / (4*np.pi/3 * a**-0.5 * density)
        kappa_F = (Q_ext_F * a**-1.5 * np.pi) / (4*np.pi/3 * a**-0.5 * density)
    
    # Return our calculations.
    return kappa_av_RP, kappa_av_F, kappa_RP, kappa_F, L, a

def GetTauScaling(folder, grain_min, grain_max, wl_list, BPASS_data, time_slice, wl_ref):
    
    kappa_av_RP, kappa_av_F, kappa_RP, kappa_F, _, _ = GetKappas(folder, grain_min, grain_max, wl_list, BPASS_data, time_slice)
    
    index = wl_list.round(decimals = 4) == wl_ref
    
    scale = kappa_RP / kappa_RP[index]
    
    ext_scale = kappa_F / kappa_F[index]

    df = pd.DataFrame({'WL':wl_list, 'Kappa_RP':kappa_RP, 'Kappa_F':kappa_F, 'Scale':scale, 'ExtScale':ext_scale})
    
    return df, kappa_av_RP, kappa_av_F

def GetLEddDataFrame(BPASS_file, continuous_SFR, grain_min, grain_max, wl_min, wl_max, f_dg):
    #########################################################################
    # Gets the existing L_Edd data frame csv data or builds a new one if one
    # has not been previously generated.
    #
    # BPASS_file -- String with the name of the BPASS file to read
    # grain_min -- Float declaring the minimum grain size.
    # grain_max -- Float declaring the maximum grain size.
    # wl_min -- Float declaring the minimum wavelength.
    # wl_max -- Float declaring the maximum wavelength.
    # f_dg -- Float declaring the dust to gas ratio.
    #########################################################################
    
    L_Edd_file = 'L_Edd data/L_Edd dataframe {} a {} to {}.csv'.format(BPASS_file.replace('.z',' z').replace('.dat',''), grain_min, grain_max)
    
    if os.path.exists(L_Edd_file):
        L_Edd_DF = pd.read_csv(L_Edd_file)
    else:
        print("File does not exist, creating L Edd data file")
        SM_file = BPASS_file.replace('spectra','starmass')
    
        BPASS_data = load.model_output(BPASS_file)
        BPASS_data.WL *= 10**-4
    
        time_list = BPASS_data.columns[BPASS_data.columns != 'WL']
    
        time_list_exp = np.power(10,time_list.astype(float))
    
        BPASS_data = BPASS_data[ (BPASS_data.WL >= wl_min) & (BPASS_data.WL <= wl_max) ]
        wl_list = BPASS_data.WL.to_numpy()
    
        kappa_av_RP = np.zeros_like(time_list)
        kappa_av_F = np.zeros_like(time_list)
        L = np.zeros_like(time_list)
    
        folder = 'Draine data Sil/'
        for i, time_slice in enumerate(time_list):
            kappa_av_RP[i], kappa_av_F[i], _, _, L[i], a = GetKappas(folder, grain_min, grain_max, wl_list, BPASS_data, time_slice)
    
        L_edd_Sil = 1.299*10**4 / (kappa_av_RP*f_dg)
    
        L_Edd_DF = pd.DataFrame({'time':time_list,'time exp':time_list_exp, 'L_bol_BPASS':L, 'kappa_av_RP_Sil':kappa_av_RP, 'kappa_av_F_Sil':kappa_av_F, 'L_Edd_Sil':L_edd_Sil})

        folder = 'Draine data SiC/'
        for i, time_slice in enumerate(time_list):
            kappa_av_RP[i], kappa_av_F[i], _, _, _, _ = GetKappas(folder, grain_min, grain_max, wl_list, BPASS_data, time_slice)
    
        L_edd_SiC =  1.299*10**4 / (kappa_av_RP*f_dg)
    
        L_Edd_DF['kappa_av_RP_SiC'] = kappa_av_RP
        L_Edd_DF['kappa_av_F_SiC'] = kappa_av_F
        L_Edd_DF['L_Edd_SiC'] = L_edd_SiC
    
        folder = 'Draine data Gra/'
        for i, time_slice in enumerate(time_list):
            kappa_av_RP[i], kappa_av_F[i], _, _, _, _ = GetKappas(folder, grain_min, grain_max, wl_list, BPASS_data, time_slice)
    
        L_edd_Gra = 1.299*10**4 / (kappa_av_RP*f_dg)
    
        L_Edd_DF['kappa_av_RP_Gra'] = kappa_av_RP
        L_Edd_DF['kappa_av_F_Gra'] = kappa_av_F
        L_Edd_DF['L_Edd_Gra'] = L_edd_Gra

        Mass = load.model_output(SM_file)
    
        M = Mass.stellar_mass + Mass.remnant_mass
    
        L_Edd_DF['Mass'] = M
    
        L_Edd_DF.to_csv(L_Edd_file, index = False)
        
        # This seems silly but is done for type safe reasons.
        L_Edd_DF = pd.read_csv(L_Edd_file)
        
    L_Edd_CSFR_file = 'L_Edd data/L_Edd dataframe CSFR {} a {} to {}.csv'.format(BPASS_file.replace('.z',' z').replace('.dat',''), grain_min, grain_max)
    
    if os.path.exists(L_Edd_CSFR_file):
        L_Edd_CSFR_DF = pd.read_csv(L_Edd_CSFR_file)
    else:
        print("File does not exist, creating L Edd CSFR data file")
        SM_file = BPASS_file.replace('spectra','starmass')
    
        time_list = continuous_SFR.columns[continuous_SFR.columns != 'WL']
    
        time_list_exp = np.power(10,time_list.astype(float))
    
        continuous_SFR = continuous_SFR[ (continuous_SFR.WL >= wl_min) & (continuous_SFR.WL <= wl_max) ]
        wl_list = continuous_SFR.WL.to_numpy()
    
        kappa_av_RP = np.zeros_like(time_list)
        kappa_av_F = np.zeros_like(time_list)
        L = np.zeros_like(time_list)
    
        folder = 'Draine data Sil/'
        for i, time_slice in enumerate(time_list):
            kappa_av_RP[i], kappa_av_F[i], _, _, L[i], a = GetKappas(folder, grain_min, grain_max, wl_list, continuous_SFR, time_slice)
    
        L_edd_Sil = 1.299*10**4 / (kappa_av_RP*f_dg)
    
        L_Edd_CSFR_DF = pd.DataFrame({'time':time_list,'time exp':time_list_exp, 'L_bol_BPASS':L, 'kappa_av_RP_Sil':kappa_av_RP, 'kappa_av_F_Sil':kappa_av_F, 'L_Edd_Sil':L_edd_Sil})

        folder = 'Draine data SiC/'
        for i, time_slice in enumerate(time_list):
            kappa_av_RP[i], kappa_av_F[i], _, _, _, _ = GetKappas(folder, grain_min, grain_max, wl_list, continuous_SFR, time_slice)
    
        L_edd_SiC =  1.299*10**4 / (kappa_av_RP*f_dg)
    
        L_Edd_CSFR_DF['kappa_av_RP_SiC'] = kappa_av_RP
        L_Edd_CSFR_DF['kappa_av_F_SiC'] = kappa_av_F
        L_Edd_CSFR_DF['L_Edd_SiC'] = L_edd_SiC
    
        folder = 'Draine data Gra/'
        for i, time_slice in enumerate(time_list):
            kappa_av_RP[i], kappa_av_F[i], _, _, _, _ = GetKappas(folder, grain_min, grain_max, wl_list, continuous_SFR, time_slice)
    
        L_edd_Gra = 1.299*10**4 / (kappa_av_RP*f_dg)
    
        L_Edd_CSFR_DF['kappa_av_RP_Gra'] = kappa_av_RP
        L_Edd_CSFR_DF['kappa_av_F_Gra'] = kappa_av_F
        L_Edd_CSFR_DF['L_Edd_Gra'] = L_edd_Gra
    
        L_Edd_CSFR_DF.to_csv(L_Edd_CSFR_file, index = False)
        
        # This seems silly but is done for type safe reasons.
        L_Edd_CSFR_DF = pd.read_csv(L_Edd_CSFR_file)
    
    return L_Edd_DF, L_Edd_CSFR_DF

# # Get the tau scaling and save it to a csv
# # -------------------------------------------------------------------

# folder = 'Draine data Sil/'
# wl_ref = 0.547
# grain_min = 0.001
# grain_max = 1

# BPASS_file = 'spectra-bin-imf100_300.z010.dat'
# time_slice = '6.0'

# BPASS_data = load.model_output(BPASS_file)
# BPASS_data.WL *= 10**-4

# wl_min = 0.001
# wl_max = 10

# BPASS_data = BPASS_data[ (BPASS_data.WL >= wl_min) & (BPASS_data.WL <= wl_max) ]
# wl_list = BPASS_data.WL.to_numpy()

# df, kappa_av_RP, kappa_av_F = GetTauScaling(folder, grain_min, grain_max, wl_list, BPASS_data, time_slice, wl_ref)

# # name = 'lambda ' + str(wl_ref).replace('.','_') + ' a ' + str(grain_min).replace('.','_') + ' ' + str(grain_max).replace('.','_') + ' ' + folder[-4:-1] +  ' time ' + time_slice + '.csv'

# df.to_csv(name)
## -------------------------------------------------------------------

## Plot tau scaling
## ---------------------------------------------------------------------
# plt.plot(df.WL, df.Scale, label = 'RP')
# plt.plot(df.WL, df.ExtScale, label = 'Ext')
# plt.legend()
# plt.xscale('log')
# plt.xlabel('Wavelength (microns)')
# plt.ylabel(r'$\tau$ scale')
# plt.title(r'scale for V band')
# plt.savefig('V band tau scaling.png', dpi = 200)
## ---------------------------------------------------------------------

# Plot L_Edd
# ------------------------------------------------------------------

# grain_min = 0.001
# grain_max = 1

# BPASS_file = 'spectra-bin-imf100_300.z001.dat'
# SM_file = BPASS_file.replace('spectra','starmass')

# BPASS_data = load.model_output(BPASS_file)
# BPASS_data.WL *= 10**-4

# time_list = BPASS_data.columns[BPASS_data.columns != 'WL']

# time_list_exp = np.power(10,time_list.astype(float))

# wl_min = 0.001
# wl_max = 10

# BPASS_data = BPASS_data[ (BPASS_data.WL >= wl_min) & (BPASS_data.WL <= wl_max) ]
# wl_list = BPASS_data.WL.to_numpy()

# kappa_av_RP = np.zeros_like(time_list)
# kappa_av_F = np.zeros_like(time_list)
# kappa_RP = np.zeros_like(wl_list)
# kappa_F = np.zeros_like(wl_list)
# L = np.zeros_like(time_list)

# fdg = 1/100

# folder = 'Draine data Sil/'
# for i, time_slice in enumerate(time_list):
    
#     kappa_av_RP[i], kappa_av_F[i], kappa_RP, kappa_F, L[i], a = GetKappas(folder, grain_min, grain_max, wl_list, BPASS_data, time_slice)

# L_edd_Sil = 1.299*10**4 / (kappa_av_RP*fdg)

# DF = pd.DataFrame({'time':time_list,'time exp':time_list_exp, 'L_bol_BPASS':L, 'kappa_av_RP_Sil':kappa_av_RP, 'kappa_av_F_Sil':kappa_av_F, 'L_Edd_Sil':L_edd_Sil})
# DF2 = pd.DataFrame({'WL':wl_list, 'kappa_RP_Sil':kappa_RP, 'kappa_F_Sil':kappa_F})



# folder = 'Draine data SiC/'
# for i, time_slice in enumerate(time_list):
    
#     kappa_av_RP[i], kappa_av_F[i], kappa_RP, kappa_F, _, _ = GetKappas(folder, grain_min, grain_max, wl_list, BPASS_data, time_slice)

# L_edd_SiC =  1.299*10**4 / (kappa_av_RP*fdg)

# DF['kappa_av_RP_SiC'] = kappa_av_RP
# DF['kappa_av_F_SiC'] = kappa_av_F
# DF2['kappa_RP_SiC'] = kappa_RP
# DF2['kappa_F_SiC'] = kappa_F
# DF['L_Edd_SiC'] = L_edd_SiC



# folder = 'Draine data Gra/'
# for i, time_slice in enumerate(time_list):
    
#     kappa_av_RP[i], kappa_av_F[i], kappa_RP, kappa_F, _, _ = GetKappas(folder, grain_min, grain_max, wl_list, BPASS_data, time_slice)

# L_edd_Gra = 1.299*10**4 / (kappa_av_RP*fdg)

# DF['kappa_av_RP_Gra'] = kappa_av_RP
# DF['kappa_av_F_Gra'] = kappa_av_F
# DF2['kappa_RP_Gra'] = kappa_RP
# DF2['kappa_F_Gra'] = kappa_F
# DF['L_Edd_Gra'] = L_edd_Gra


# Mass = load.model_output(SM_file)

# M = Mass.stellar_mass + Mass.remnant_mass

# DF['Mass'] = M

# DF.to_csv('L_Edd dataframe {} .csv'.format(BPASS_file.replace('.z',' z').replace('.dat','')), index = False)
# # DF2.to_csv('kappa by wl.csv', index = False)

# plt.plot(time_list_exp, L_edd_Sil, label = r'$L_{Edd}/M \, (Sil)$')
# plt.plot(time_list_exp, L_edd_SiC, label = r'$L_{Edd}/M \, (SiC)$')
# plt.plot(time_list_exp, L_edd_Gra, label = r'$L_{Edd}/M \, (Gra)$')
# plt.plot(time_list_exp, L/M, 'k--', label = r'$L/M \, (BPASS)$')
# plt.xscale('log')
# plt.xlabel('Time (years)')
# plt.yscale('log')
# # plt.title(r'$\langle L_{edd}/M \rangle$ over time for grain sizes ' + str(min(a)) + ' micron to ' + str(max(a)) + ' micron')
# plt.xlim(10**6,10**9)
# plt.ylim(1,2000)
# plt.legend()
# plt.ylabel(r'$ L/M \; (L_{\odot}/M_{\odot}) $')
# plt.tight_layout()
# # plt.savefig('L_edd over M 3.png', dpi = 200)
# ##--------------------------------------------------------------------


## -------------------------------------------------------------------
# time_slice = '6.0'

# kappa_av_RP, kappa_av_F, kappa_RP, kappa_F, L, a = GetKappas(folder, grain_min, grain_max, wl_list, BPASS_data, time_slice)

# # plt.plot(wl_list, (BPASS_data[time_slice].to_numpy()*10**4*Q_ext_RP*wl_list)[0], label = r'$L_\lambda Q_{\rm ext} \lambda$')
# plt.plot(wl_list, BPASS_data[time_slice].to_numpy()*10**4, label = r'$L_\lambda (L_{\odot}/\mu m)$')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$\lambda$ (microns)')
# plt.ylabel(r'$Q_{ext,\, RP}\lambda$L (solar luminosity)')
# plt.ylim(10**-2,10**11)
# plt.legend()
# plt.savefig('Q lambda F by wl with L.png', dpi = 200)

## -------------------------------------------------------------------



# ## Plot tau
# ## --------------------------------------------------------------------
# Sigma_gas = 2*10**-4
# plt.plot(time_list_exp, kappa_rp*Sigma_gas, label = r'$\tau_{rp}$')
# plt.plot(time_list_exp, kappa_F*Sigma_gas, label = r'$\tau_{F}$')
# plt.xscale('log')
# plt.yscale('log')
# plt.ylabel(r'$\tau$')
# plt.xlabel('Time (years)')
# plt.legend()
# plt.savefig('tau rp and F.png', dpi = 200)
# ## --------------------------------------------------------------------

## Read ML ratio
## --------------------------------------------------------------------
# ML_ratio_file = 'mlratio-bin-imf135_300.z020.dat'

# cols = ['Time (log(years))', 'M/L (K band)', 'M/L (V band)']

# ML_DF = pd.read_csv(ML_ratio_file, skiprows=1, names = cols, delim_whitespace=True)

# time_list_exp = np.power(10,ML_DF['Time (log(years))'])

# plt.plot(time_list_exp, 1/ML_DF['M/L (V band)'] + 1/ML_DF['M/L (K band)'])
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(10**6,10**10)
## --------------------------------------------------------------------

## Read Surviving mass file
## -------------------------------------------------------------------

# SM_file = 'starmass-bin-imf135_300.z020.dat'

# Mass = load.model_output(SM_file)

# time_list_exp = np.power(10,Mass['log_age'])

# plt.plot(time_list_exp, Mass['stellar_mass'], label = 'Stellar mass')
# plt.plot(time_list_exp, Mass['remnant_mass'], label = 'Remnant mass')
# plt.xlabel('Time (years)')
# plt.ylabel(r'Mass ($M_\odot$)')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(10**6,10**11)
# plt.ylim(1,2*10**6)
# plt.legend()
# plt.title('Stellar and remnant mass over time')
# plt.savefig('Stellar mass.png', dpi = 200)

## -------------------------------------------------------------------



# # Plot < kappa_rp >
# # -------------------------------------------------------------------
# plt.plot(time_list_exp, kappa_av_RP, label = r'$\langle \kappa_{RP} \rangle$')

# plt.xscale('log')
# plt.xlabel('Time (years)')
# plt.yscale('log')
# plt.title(r'$\kappa$ for grains from ' + str(min(a)) + ' micron to ' + str(max(a)) + ' micron')
# plt.xlim(10**6,10**10)
# plt.legend()
# plt.ylabel(r'$ \langle \kappa_{rp} \rangle \; (cm^2 /g) $')
# plt.tight_layout()
# plt.savefig('kappa rp over time.png', dpi = 200)
# # -------------------------------------------------------------------




# # Plot kappa as a function of wavelength
# # --------------------------------------------------------
# folder = 'Draine data Sil/'

# grain_min = 0.001
# grain_max = 10

# wl_list = np.arange(0.001, 10.001, 0.001)


# kappa = np.zeros_like(wl_list)

# kappa, a = GetKappa(folder, grain_min, grain_max, wl_list, 'RP')


# plt.plot(wl_list, kappa, label = r'$\kappa$')
# # # plt.plot(wl_list, k_rp, label = r'$\kappa_{rp}$')
# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.xscale('log')
# plt.yscale('log')
# plt.ylabel(r'$\langle \kappa \rangle (cm^2 / g)$')
# plt.title(r'$\langle \kappa \rangle$ over wavelength for grain sizes ' + str(min(a)) + ' micron to ' + str(max(a)) + ' micron')
# plt.legend()
# plt.savefig('kappa by wavelength multiple a.png', dpi = 200)

# # ---------------------------------------------------------

# # Plot kappa as a function of wavelength
# # --------------------------------------------------------
# folder = 'Draine data Sil/'


# grain_min = 0.001

# wl_list = np.arange(0.001, 1000.001, 0.001)

# a_list = [1.259e-03, 1.413e-03, 1.585e-03, 1.778e-03,
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

# kappa = np.zeros((len(a_list),len(wl_list)))



# for i, grain_max in enumerate(a_list):
#     kappa[i, :] = GetKappa(folder, grain_min, grain_max, wl_list, 'RP')


# plt.plot(wl_list, kappa[18,:], label = r'a_{max} = ' + str(a_list[18]))
# plt.plot(wl_list, kappa[28,:], label = r'a_{max} = ' + str(a_list[28]))
# plt.plot(wl_list, kappa[38,:], label = r'a_{max} = ' + str(a_list[38]))
# plt.plot(wl_list, kappa[48,:], label = r'a_{max} = ' + str(a_list[48]))


# # # plt.plot(wl_list, k_rp, label = r'$\kappa_{rp}$')
# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.xscale('log')
# plt.yscale('log')
# plt.ylabel(r'$\langle \kappa \rangle (cm^2 / g)$')
# plt.title(r'$\langle \kappa \rangle$ over wavelength')
# plt.legend()
# plt.savefig('kappa by wavelength multiple a.png', dpi = 200)

# ---------------------------------------------------------


## Plot multiple time slices
## -----------------------------------------------------------
# BPASS_file = 'spectra-bin-imf135_300.z020.dat'

# BPASS_data = load.model_output(BPASS_file)
# BPASS_data.WL *= 10**-4

# time_list = BPASS_data.columns[BPASS_data.columns != 'WL']

# time_list_exp = np.power(10,time_list.astype(float))

# plt.plot(BPASS_data.WL, BPASS_data['6.0'], label = '1 MYr')
# plt.plot(BPASS_data.WL, BPASS_data['7.0'], label = '10 MYr')
# plt.plot(BPASS_data.WL, BPASS_data['8.0'], label = '100 MYr')
# plt.plot(BPASS_data.WL, BPASS_data['9.0'], label = '1 GYr')
# plt.plot(BPASS_data.WL, BPASS_data['10.0'], label = '10 GYr')

# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(10**-3,10**7)
# plt.xlim(10**-3,10)
# plt.ylabel('Flux (Solar luminosity per angstrom)')
# plt.title('Flux by wavelength for different age populations')
# plt.legend()
# plt.savefig('multiple spectra.png', dpi = 200)
## -----------------------------------------------------------

## Plot L_lambda / L
## -----------------------------------------------------------
# BPASS_file = 'spectra-bin-imf135_300.z020.dat'

# BPASS_data = load.model_output(BPASS_file)
# BPASS_data.WL *= 10**-4

# time_list = BPASS_data.columns[BPASS_data.columns != 'WL']

# time_list_exp = np.power(10,time_list.astype(float))

# plt.plot(BPASS_data.WL, BPASS_data['6.0']/BPASS_data['6.0'].sum(), label = '1 MYr')

# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(10**-9,3*10**-3)
# plt.xlim(10**-2,10)
# plt.ylabel(r'$L_\lambda / L_{bol}$')
# plt.legend()
# plt.savefig('L contribution by wavelength.png', dpi = 200)
## -----------------------------------------------------------

##  Plot super and sub Eddington regions
## -------------------------------------------------------------------------

# M51 = pd.read_csv('M51.csv')

# NGC6946 = pd.read_csv('NGC6946.csv')

# r = 83/2
# r = 73/2

# L_over_M = M51.LBol/(M51.Sigma_star*np.pi*r**2)/(3.826*10**33)
# L_over_M = NGC6946.LBol/(NGC6946.Sigma_star*np.pi*r**2)/(3.826*10**33)


# plt.scatter(M51.xcenter, M51.ycenter, s = 1, c = L_over_M, norm = matplotlib.colors.DivergingNorm(L_edd_Sil[0]), cmap = 'coolwarm')
# plt.savefig('L over M for M51.png', dpi = 200)

# plt.scatter(NGC6946.xcenter, NGC6946.ycenter, s = 1, c = L_over_M, norm = matplotlib.colors.DivergingNorm(L_edd_Sil[0]), cmap = 'coolwarm')
# plt.savefig('L over M for NGC6946.png', dpi = 200)

## -------------------------------------------------------------------------



# # Plot Q_abs and Q_sca for several grain sizes
## ------------------------------------------------------------------------------
# a_0_001 = pd.read_csv(folder + 'draine_data_Sil_0_001', delim_whitespace = True)

# a_0_01 = pd.read_csv(folder + 'draine_data_Sil_0_01', delim_whitespace = True)

# a_0_1 = pd.read_csv(folder + 'draine_data_Sil_0_1', delim_whitespace = True)

# a_1 = pd.read_csv(folder + 'draine_data_Sil_1_0', delim_whitespace = True)

# a_10_0 = pd.read_csv(folder + 'draine_data_Sil_10_0', delim_whitespace = True)

# a_0_001.columns = ['wl', 'Q_abs', 'Q_sca', 'g']

# a_0_01.columns = ['wl', 'Q_abs', 'Q_sca', 'g']

# a_0_1.columns = ['wl', 'Q_abs', 'Q_sca', 'g']

# a_1.columns = ['wl', 'Q_abs', 'Q_sca', 'g']


# plt.plot(a_10_0.wl, a_10_0.Q_abs, label = 'Q_abs, a = 10')
# plt.plot(a_10_0.wl, a_10_0.Q_sca, label = 'Q_sca, a = 10')
# plt.plot(a_1.wl, a_1.Q_abs, label = 'Q_abs, a = 1')
# plt.plot(a_1.wl, a_1.Q_sca, label = 'Q_sca, a = 1')
# plt.plot(a_0_1.wl, a_0_1.Q_abs, label = 'Q_abs, a = 0.1')
# plt.plot(a_0_1.wl, a_0_1.Q_sca, label = 'Q_sca, a = 0.1')
# plt.plot(a_0_01.wl, a_0_01.Q_abs, label = 'Q_abs, a = 0.01')
# plt.plot(a_0_01.wl, a_0_01.Q_sca, label = 'Q_sca, a = 0.01')
# plt.plot(a_0_001.wl, a_0_001.Q_abs, label = 'Q_abs, a = 0.001')
# plt.plot(a_0_001.wl, a_0_001.Q_sca, label = 'Q_sca, a = 0.001')

# plt.ylabel('Q')
# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend(bbox_to_anchor = (1.05,1), loc = 'upper left')
# plt.savefig('Q over wavelength for select grain sizes.png', dpi = 200, bbox_inches = 'tight')


# # Compare interpolation
# Q_abs = np.interp(wl_list, np.flip(a_1.wl.to_numpy()), np.flip(a_1.Q_abs.to_numpy()))

# plt.plot(wl_list, Q_abs, label = 'Q_abs, a = 1, Interpolated', linewidth = 1)
# plt.plot(a_1.wl, a_1.Q_abs, label = 'Q_abs, a = 1, Draine', linewidth = 0.5)
# plt.ylabel('Q')
# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend(bbox_to_anchor = (1.05,1), loc = 'upper left')

# plt.savefig('interpolation check.png', dip = 200, bbox_inches = 'tight')

# # pick out the row and column to read from
# row = 9000
# column = 10

# # The first column contains wavelength data, the rest have to be calculated manually.
# labels = ["Wavelength"]

# # Get the rest of the column names
# for n in range(1,52,1):
#     labels.append(10**(6+0.1*(n-2)))

# # read in the BPASS data, the engine must be the slower python engine because the C engine doesn't
# # support multiple whitespace dilimiting.
# data_file = pd.read_csv("spectra-bin-imf135_300.z020.dat", header = None, delimiter="   ", names = labels, engine='python')

# # Coinvert the wavelengths from angstroms to microns.
# data_file *= 10**4
# data_file.Wavelength *= 10**-8

# # plot_data = data_file.iloc[row][1:-1]

# # plot_data.plot(logx=True)
# # plt.title("Flux by year for wavelength " + str(data_file.iloc[row][0]) + " microns")
# # plt.ylabel('flux (solar luminosity per micron)')
# # plt.xlabel('log year')

# # data_file[data_file['Wavelength'].between(0.01,10)].plot(x=labels[0], y=labels[column], legend=False, loglog=True)
# # plt.title('Spectral distribution at time t = ' + f"{labels[column]:.2f}" + ' years')
# # plt.xlabel('log Wavelength (microns)')
# # plt.ylabel('Log Flux (Solar luminosity per angstrom)')

# integral = np.trapz(data_file, axis = 0)[1:]

# # integral[1:15].plot(loglog=True)
# # plt.ylabel('log total luminosity (solar luminosity)')
# # plt.xlabel('log Age (years)')
# # plt.title('Total luminosity by age')

# # give the name of the Draine source file.
# source_file = "Draine data/draine_test_2_152"

# # import the data from the file.
# draine_file = pd.read_csv(source_file, delim_whitespace = True)

# # Get the list of wavelengths we will be using.
# lambda_list = draine_file.iloc[:,0]
# lambda_list = lambda_list[lambda_list <= 10]

# Q = np.zeros((data_file.shape[1]-1,2))

# for column in range(0,data_file.shape[1]-1):
#     # Get only the data for the time slice we are interested in
#     time_data = data_file.iloc[:,[0,column+1]]
    
#     # Start at 0
#     Q_rpL = np.zeros(len(lambda_list))
    
#     # For each wavelength in the Draine file for a given grain size.
#     for i, wavelength in enumerate(lambda_list):
#         # Find the closest wavelength in the BPASS data to the wavelength given by Draine.
#         index = abs(time_data['Wavelength'] - wavelength).idxmin()
        
#         # Get the Qs we need and the value of g.
#         Q_abs = draine_file.iloc[i,1]
#         Q_scatt = draine_file.iloc[i,2]
#         g = Q_abs = draine_file.iloc[i,3]
        
#         if wavelength < 10:
#             # Add the contribution from this wavelength.
#             Q_rpL[i] = (Q_abs + (1-g)*Q_scatt)*time_data.iloc[index,1]


#     Q[column,0] = labels[column+1]
#     Q[column,1] = np.trapz(Q_rpL)
    
# plt.plot(np.log10(Q[0:15,0]),np.log10(Q[0:15,1]/integral[1:16]))
# plt.title('Q_rp over time for astronomical silicate')
# plt.xlabel('log Years')
# plt.ylabel('log Q_rp')

# plt.plot(np.log10(Q[0:15,0]),np.log10(Q_1[0:15]/integral[1:16]), label = "0.001")
# plt.plot(np.log10(Q[0:15,0]),np.log10(Q_2[0:15]/integral[1:16]), label = "0.02512")
# plt.plot(np.log10(Q[0:15,0]),np.log10(Q_3[0:15]/integral[1:16]), label = "1.585")
# plt.plot(np.log10(Q[0:15,0]),np.log10(Q_4[0:15]/integral[1:16]), label = "2.152")
# plt.title('Q_rp over time for astronomical silicate')
# plt.legend(loc = 4)
# plt.xlabel('log Years')
# plt.ylabel('log Q_rp')
# plt.savefig('multiple_Q_rp.png', dpi=200)

# integrand = np.zeros(len(lambda_list))
# integrand2 = np.zeros(len(lambda_list))
# integrand3 = np.zeros(len(lambda_list))
# wave_error = np.zeros(len(lambda_list))

# for i, wavelength in enumerate(lambda_list):
#     # Get only the data for the time slice we are interested in
#     time_data = data_file.iloc[:,[0,column]]
    
#     index = abs(time_data['Wavelength'] - wavelength).idxmin()
#     wave_error[i] = abs(time_data['Wavelength'] - wavelength).min()
   
#     Q_abs = draine_file.iloc[i,1]
#     Q_scatt = draine_file.iloc[i,2]
#     g = Q_abs = draine_file.iloc[i,3]
        
#     integrand[i] = (Q_abs + (1-g)*Q_scatt)
    
#     if wavelength < 10:
#         integrand2[i] = time_data.iloc[index,1]
    
#     integrand3[i] = integrand[i]*integrand2[i]
    
# plt.plot(np.log10(lambda_list),np.log10(integrand),label='Q contribution')
# plt.plot(np.log10(lambda_list),np.log10(integrand2),label='L_lambda contribution')
# # plt.plot(np.log10(lambda_list),np.log10(wave_error), label='Error')
# plt.plot(np.log10(lambda_list),np.log10(integrand3),label='Full integrand', color = 'red')
# plt.legend()
# plt.title('integrand for numerator of Q_rp for 2.152 micron dust at t = ' + f'{labels[column]:.2f}' + ' years')
# plt.xlabel('log wavelength (microns)')
# plt.ylabel('log integrand')