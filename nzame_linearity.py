#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pylab as plt
from matplotlib import rcParams


# In[2]:


rcParams["mathtext.default"] = 'regular'
rcParams['font.size'] = 16

output_write = '/home/users/zosiast/jasmin_plots/no_anthro_ensemble_ssp370/'


# ## Data for temperature and methane, ozone concs in 2050

# Data for AerChemMIP results is from Allen et al 2021 https://doi.org/10.1088/1748-9326/abe06b

# In[3]:


# Difference in methane conc in 2050
d_ch4_nzame_conc = 1560.
d_ch4_aer_conc = 1113.


# In[4]:


# difference in ozone conc (ppb) in 2050
d_ozone_nzame = -5.2
d_ozone_aer = -2.85
d_ozone_ukesm = -3.07
ozone_nzame_err = 0.2
ozone_aer_err = 0.13

#other ensemble members in AerChemMIP
labels = ['GFDL-ESM4','MRI-ESM2-0','EC-Earth3-AerChem','GISS-E2-1-G']
d_o3_ens_2050 = [-3.01,-2.92,-3.63,-1.63]


# In[5]:


# difference in surface temp (K)
d_temp_nzame = -0.96
d_temp_aer = -0.39
d_temp_ukesm = -0.57
temp_aer_err = 0.05
temp_nzame_err = 0.09

# other ens members
d_gmst_ens_2050 = [-0.26,-0.31,-0.46,-0.34]


# ## Ozone linearity test

# In[6]:


# 0,0 = 2015 value for ozone and methane
fig = plt.figure(dpi=200) #200 for printing
ax = plt.axes()

x_vals = [0,d_ch4_aer_conc*2]
y_1 = [ozone_aer_err,ozone_aer_err + d_ozone_aer*2]
y_2 = [-ozone_aer_err,-ozone_aer_err + d_ozone_aer*2]

ax.fill_between(x_vals,y_1,y_2, color='lightblue', label = 'AerChemMIP linear')

ax.errorbar([d_ch4_aer_conc], [d_ozone_aer], yerr = [ozone_aer_err],fmt='+',label='AerChemMIP MMM')
ax.errorbar(d_ch4_nzame_conc, d_ozone_nzame,yerr = [ozone_nzame_err],fmt='+',label=' NZAME', c='orange')

ax.plot([0,d_ch4_aer_conc*2], [0,d_ozone_ukesm*2], 'green',linestyle=':', label = 'UKESM linear')
ax.scatter([d_ch4_aer_conc]*4,d_o3_ens_2050, c='lightblue', marker='x', label='AerChemMIP ensemble')
ax.scatter(d_ch4_aer_conc, d_ozone_ukesm,marker='+',label=' UKESM1', c='green',s=50)



ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#plt.legend(fontsize=13,loc=3)
ax.set_xlim(0,1750)
plt.xlabel('$\Delta$ [CH$_4$] / ppb')
plt.ylabel(f'$\Delta$ [O$_3$] / ppb')
plt.title('2050 $\Delta$[O$_3$] vs $\Delta$[CH$_4$]')
#plt.savefig(f'{output_write}d_o3_d_ch4_2050.png',bbox_inches='tight')


# ## Methane vs temp relationship test

# In[7]:


ssp370_2050_conc_int = np.sqrt(2181.)
nzame_2050_conc = np.sqrt(621.)
diff_nzame =  ssp370_2050_conc_int - nzame_2050_conc

ssp370_2050_conc_mein = np.sqrt(2472.)
lowntcf_2050_conc = np.sqrt(1359.)
diff_aerchemmip =  ssp370_2050_conc_mein - lowntcf_2050_conc


# In[8]:


# 0,0 = 2015 value for ozone and methane
fig = plt.figure(dpi=200)
ax = plt.axes()

ax.errorbar([diff_aerchemmip], [d_temp_aer], yerr = [temp_aer_err],fmt='+',label='AerChemMIP MMM')
ax.errorbar([diff_nzame], [d_temp_nzame], yerr = [temp_nzame_err],fmt='+', label = 'NZAME')
ax.scatter([diff_aerchemmip], [d_temp_ukesm], label = 'UKESM1.0', c='green', marker='x')

ax.plot([0,diff_aerchemmip*2], [0,d_temp_ukesm*2], 'green',linestyle=':', label = 'UKESM1.0 linear')
ax.scatter([diff_aerchemmip]*4,d_gmst_ens_2050, c='lightblue', marker='x', label='AerChemMIP ensemble')

x_vals = [0,diff_aerchemmip*2]
y_1 = [temp_aer_err,temp_aer_err + d_temp_aer*2]
y_2 = [-temp_aer_err,-temp_aer_err + d_temp_aer*2]

ax.fill_between(x_vals,y_1,y_2, color='lightblue', label = 'AerChemMIP linear')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles,labels = ax.get_legend_handles_labels()

handles = [handles[5], handles[1], handles[0],handles[2], handles[4], handles[3]]
labels = [labels[5], labels[1], labels[0], labels[2], labels[4], labels[3]]


plt.legend(handles, labels, fontsize=14,frameon = False)#x_to_anchor = [0.65,0.35])
plt.xlabel('$\Delta\sqrt{[CH_4] / ppb}$')
plt.ylabel(f'$\Delta$ GMST / K')
plt.title('$\Delta$ GMST vs $\sqrt{[CH_4]}$ 2015 - 2050')
#plt.savefig(f'{output_write}d_gmst_d_ch4_2050.png', bbox_inches='tight')


# In[ ]:





# In[ ]:




