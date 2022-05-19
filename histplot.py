nbins = 100
#Fbins_1Myr_5pc = np.logspace(np.log10(min(gal_data_1Myr_5pc.F_Bol/gal_data_1Myr_5pc.F_Edd)), np.log10(max(gal_data_1Myr_5pc.F_Bol/gal_data_1Myr_5pc.F_Edd)), nbins)
#Fbins_10Myr_5pc = np.logspace(np.log10(min(gal_data_10Myr_5pc.F_Bol/gal_data_10Myr_5pc.F_Edd)), np.log10(max(gal_data_10Myr_5pc.F_Bol/gal_data_10Myr_5pc.F_Edd)), nbins)
# Fbins_100Myr_5pc = np.logspace(np.log10(min(gal_data_100Myr_5pc.F_Bol/gal_data_100Myr_5pc.F_Edd)), np.log10(max(gal_data_100Myr_5pc.F_Bol/gal_data_100Myr_5pc.F_Edd)), nbins)
#Fbins_1Myr_10pc = np.logspace(np.log10(min(gal_data_1Myr_10pc.F_Bol/gal_data_1Myr_10pc.F_Edd)), np.log10(max(gal_data_1Myr_10pc.F_Bol/gal_data_1Myr_10pc.F_Edd)), nbins)
#Fbins_10Myr_10pc = np.logspace(np.log10(min(gal_data_10Myr_10pc.F_Bol/gal_data_10Myr_10pc.F_Edd)), np.log10(max(gal_data_10Myr_10pc.F_Bol/gal_data_10Myr_10pc.F_Edd)), nbins)
# Fbins_100Myr_10pc = np.logspace(np.log10(min(gal_data_100Myr_10pc.F_Bol/gal_data_100Myr_10pc.F_Edd)), np.log10(max(gal_data_100Myr_10pc.F_Bol/gal_data_100Myr_10pc.F_Edd)), nbins)
#Fbins_1Myr_20pc = np.logspace(np.log10(min(gal_data_1Myr_20pc.F_Bol/gal_data_1Myr_20pc.F_Edd)), np.log10(max(gal_data_1Myr_20pc.F_Bol/gal_data_1Myr_20pc.F_Edd)), nbins)
#Fbins_10Myr_20pc = np.logspace(np.log10(min(gal_data_10Myr_20pc.F_Bol/gal_data_10Myr_20pc.F_Edd)), np.log10(max(gal_data_10Myr_20pc.F_Bol/gal_data_10Myr_20pc.F_Edd)), nbins)
# Fbins_100Myr_20pc = np.logspace(np.log10(min(gal_data_100Myr_20pc.F_Bol/gal_data_100Myr_20pc.F_Edd)), np.log10(max(gal_data_100Myr_20pc.F_Bol/gal_data_100Myr_20pc.F_Edd)), nbins)

Lbins_1Myr_5pc = np.logspace(np.log10(min(gal_data_1Myr_5pc.L_Bol/gal_data_1Myr_5pc.L_Edd)), np.log10(max(gal_data_1Myr_5pc.L_Bol/gal_data_1Myr_5pc.L_Edd)), nbins)
Lbins_10Myr_5pc = np.logspace(np.log10(min(gal_data_10Myr_5pc.L_Bol/gal_data_10Myr_5pc.L_Edd)), np.log10(max(gal_data_10Myr_5pc.L_Bol/gal_data_10Myr_5pc.L_Edd)), nbins)
# Lbins_100Myr_5pc = np.logspace(np.log10(min(gal_data_100Myr_5pc.L_Bol/gal_data_100Myr_5pc.L_Edd)), np.log10(max(gal_data_100Myr_5pc.L_Bol/gal_data_100Myr_5pc.L_Edd)), nbins)
Lbins_1Myr_10pc = np.logspace(np.log10(min(gal_data_1Myr_10pc.L_Bol/gal_data_1Myr_10pc.L_Edd)), np.log10(max(gal_data_1Myr_10pc.L_Bol/gal_data_1Myr_10pc.L_Edd)), nbins)
Lbins_10Myr_10pc = np.logspace(np.log10(min(gal_data_10Myr_10pc.L_Bol/gal_data_10Myr_10pc.L_Edd)), np.log10(max(gal_data_10Myr_10pc.L_Bol/gal_data_10Myr_10pc.L_Edd)), nbins)
# Lbins_100Myr_10pc = np.logspace(np.log10(min(gal_data_100Myr_10pc.L_Bol/gal_data_100Myr_10pc.L_Edd)), np.log10(max(gal_data_100Myr_10pc.L_Bol/gal_data_100Myr_10pc.L_Edd)), nbins)
Lbins_1Myr_20pc = np.logspace(np.log10(min(gal_data_1Myr_20pc.L_Bol/gal_data_1Myr_20pc.L_Edd)), np.log10(max(gal_data_1Myr_20pc.L_Bol/gal_data_1Myr_20pc.L_Edd)), nbins)
Lbins_10Myr_20pc = np.logspace(np.log10(min(gal_data_10Myr_20pc.L_Bol/gal_data_10Myr_20pc.L_Edd)), np.log10(max(gal_data_10Myr_20pc.L_Bol/gal_data_10Myr_20pc.L_Edd)), nbins)
# Lbins_100Myr_20pc = np.logspace(np.log10(min(gal_data_100Myr_20pc.L_Bol/gal_data_100Myr_20pc.L_Edd)), np.log10(max(gal_data_100Myr_20pc.L_Bol/gal_data_100Myr_20pc.L_Edd)), nbins)

fig, ax = plt.subplots(1,2, figsize=(15,5), sharey = True, sharex = True, dpi = 200)

#ax[0].hist(gal_data_1Myr_5pc.F_Bol/gal_data_1Myr_5pc.F_Edd, bins = Fbins_1Myr_5pc, alpha = 0.5, label = "Planar")
ax[0].hist(gal_data_1Myr_5pc.L_Bol/gal_data_1Myr_5pc.L_Edd, bins = Lbins_1Myr_5pc, alpha = 0.5, label = "5 pc")

#ax[1].hist(gal_data_10Myr_5pc.F_Bol/gal_data_10Myr_5pc.F_Edd, bins = Fbins_10Myr_5pc, alpha = 0.5, label = "Planar")
ax[1].hist(gal_data_10Myr_5pc.L_Bol/gal_data_10Myr_5pc.L_Edd, bins = Lbins_10Myr_5pc, alpha = 0.5, label = "5 pc")

# ax[2].hist(gal_data_100Myr_5pc.F_Bol/gal_data_100Myr_5pc.F_Edd, bins = Fbins_100Myr_5pc, alpha = 0.5, label = "Planar")
# ax[2].hist(gal_data_100Myr_5pc.L_Bol/gal_data_100Myr_5pc.L_Edd, bins = Lbins_100Myr_5pc, alpha = 0.5, label = "5 pc")



#ax[0].hist(gal_data_1Myr_10pc.F_Bol/gal_data_1Myr_10pc.F_Edd, bins = Fbins_1Myr_10pc, alpha = 0.5, label = "Planar")
ax[0].hist(gal_data_1Myr_10pc.L_Bol/gal_data_1Myr_10pc.L_Edd, bins = Lbins_1Myr_10pc, alpha = 0.5, label = "10 pc")

#ax[1].hist(gal_data_10Myr_10pc.F_Bol/gal_data_10Myr_10pc.F_Edd, bins = Fbins_10Myr_10pc, alpha = 0.5, label = "Planar")
ax[1].hist(gal_data_10Myr_10pc.L_Bol/gal_data_10Myr_10pc.L_Edd, bins = Lbins_10Myr_10pc, alpha = 0.5, label = "10 pc")

# ax[2].hist(gal_data_100Myr_10pc.F_Bol/gal_data_100Myr_10pc.F_Edd, bins = Fbins_100Myr_10pc, alpha = 0.5, label = "Planar")
# ax[2].hist(gal_data_100Myr_10pc.L_Bol/gal_data_100Myr_10pc.L_Edd, bins = Lbins_100Myr_10pc, alpha = 0.5, label = "10 pc")



#ax[0].hist(gal_data_1Myr_20pc.F_Bol/gal_data_1Myr_20pc.F_Edd, bins = Fbins_1Myr_20pc, alpha = 0.5, label = "Planar")
ax[0].hist(gal_data_1Myr_20pc.L_Bol/gal_data_1Myr_20pc.L_Edd, bins = Lbins_1Myr_20pc, alpha = 0.5, label = "20 pc")

#ax[1].hist(gal_data_10Myr_20pc.F_Bol/gal_data_10Myr_20pc.F_Edd, bins = Fbins_10Myr_20pc, alpha = 0.5, label = "Planar")
ax[1].hist(gal_data_10Myr_20pc.L_Bol/gal_data_10Myr_20pc.L_Edd, bins = Lbins_10Myr_20pc, alpha = 0.5, label = "20 pc")

# ax[2].hist(gal_data_100Myr_20pc.F_Bol/gal_data_100Myr_20pc.F_Edd, bins = Fbins_100Myr_20pc, alpha = 0.5, label = "Planar")
# ax[2].hist(gal_data_100Myr_20pc.L_Bol/gal_data_100Myr_20pc.L_Edd, bins = Lbins_100Myr_20pc, alpha = 0.5, label = "20 pc")

ax[0].legend()

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
# ax[2].set_xscale('log')
# ax[2].set_yscale('log')

ax[0].set_ylim(1)
ax[0].set_xlim(0.01)

ax[0].set_title("1 Myr")
ax[1].set_title("10 Myr")
# ax[2].set_title("100 Myr")

ax[0].set_ylabel("Count")
ax[1].set_xlabel("Eddington Ratio")