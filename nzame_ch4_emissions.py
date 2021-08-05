#!/usr/bin/env python
# coding: utf-8

# ### CH4 emissions from 1985 to 2050 in NZAME and historical simulations

# In[1]:


import numpy as np
import netCDF4 as nc
import pylab as plt
import matplotlib
import numpy.ma as ma
from matplotlib import rcParams
import datetime
import matplotlib.dates as mdates
import pandas as pd
import cftime
import nc_time_axis


# In[2]:


#constants
mr_ch4 = 16.
per_sec_to_per_yr = 60*60*24*360
g_to_Tg = 1e12
n_a = 6.022e23

output_write = '/home/users/zosiast/jasmin_plots/no_anthro_ensemble_ssp370/'
emissions_loc = '/home/users/zosiast/scripts/bl593_bc179_NOAA/emissions_arrays/'

#stash codes
stash_ch4_ems = 'm01s50i306'
stash_wet_ems = 'm01s50i304'
stash_wet_area = 'm01s08i248'


# # 2016-2050: u-by186 and u-be647

# ### u-by186: net zero anthro CH4

# In[8]:


suite_id = 'u-by186'
suite_lab = 'uby186'
dates = '2015_2050'

data_1 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_tot_ems_{dates}.nc')
#extract variables to arrays
ch4_anthro_1 = data_1.variables[stash_ch4_ems][:]
lat_1 = data_1.variables['latitude'][:]
lon_1 = data_1.variables['longitude'][:]

time_1 = data_1.variables['time']
dtime_1 = nc.num2date(time_1[:],time_1.units)
#cf_dtime_1 = nc.num2date(time_1[:],time_1.units, time_1.calendar)
#dtime_1 = np.array([nc_time_axis.CalendarDateTime(item, "360_day") for item in cf_dtime_1])

#wetland ems
data_wet_1 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_wet_ems_{dates}.nc')
#extract variables to arrays
ch4_wet_1 = data_wet_1.variables[stash_wet_ems][:]


# ### u-bz146

# In[9]:


suite_id = 'u-bz146'
suite_lab = 'ubz146'
dates = '2015_2050'

data_4 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_tot_ems_{dates}.nc')
#extract variables to arrays
ch4_anthro_4 = data_4.variables[stash_ch4_ems][:]
lat_4 = data_4.variables['latitude'][:]
lon_4 = data_4.variables['longitude'][:]

time_4 = data_4.variables['time']
dtime_4 = nc.num2date(time_4[:],time_4.units)
#cf_dtime_4 = nc.num2date(time_4[:],time_4.units, time_4.calendar)
#dtime_4 = np.array([nc_time_axis.CalendarDateTime(item, "360_day") for item in cf_dtime_4])

#wetland ems
data_wet_4 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_wet_ems_{dates}.nc')
#extract variables to arrays
ch4_wet_4 = data_wet_4.variables[stash_wet_ems][:]


# ### u-bz473

# In[10]:


suite_id = 'u-bz473'
suite_lab = 'ubz473'
dates = '2015_2050'

data_3 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_tot_ems_{dates}.nc')
#extract variables to arrays
ch4_anthro_3 = data_3.variables[stash_ch4_ems][:]
lat_3 = data_3.variables['latitude'][:]
lon_3 = data_3.variables['longitude'][:]

time_3 = data_3.variables['time']
dtime_3 = nc.num2date(time_3[:],time_3.units)
#cf_dtime_3 = nc.num2date(time_3[:],time_3.units, time_3.calendar)
#dtime_3 = np.array([nc_time_axis.CalendarDateTime(item, "360_day") for item in cf_dtime_3])

#wetland ems
data_wet_3 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_wet_ems_{dates}.nc')
#extract variables to arrays
ch4_wet_3 = data_wet_3.variables[stash_wet_ems][:]


# ### u-bo797

# In[11]:


suite_id = 'u-bo797'
suite_lab = 'ubo797'
dates = '2016_2050'

data_2 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_ems_{dates}.nc')
#extract variables to arrays
ch4_anthro_2 = data_2.variables[stash_ch4_ems][:]
lat_2 = data_1.variables['latitude'][:]
lon_2 = data_1.variables['longitude'][:]

time_2 = data_2.variables['time']
dtime_2 = nc.num2date(time_2[:],time_2.units)
#cf_dtime_2 = nc.num2date(time_2[:],time_2.units, time_2.calendar)
#dtime_2 = np.array([nc_time_axis.CalendarDateTime(item, "360_day") for item in cf_dtime_2])

#wetland ems
data_wet_2 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_wet_ems_{dates}.nc')
#extract variables to arrays
ch4_wet_2 = data_wet_2.variables[stash_wet_ems][:]


# ### u-ca723

# In[12]:


suite_id = 'u-ca723'
suite_lab = 'uca723'
dates = '2015_2050'

data_6 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_tot_ems_{dates}.nc')
#extract variables to arrays
ch4_anthro_6 = data_6.variables[stash_ch4_ems][:]
lat_6 = data_6.variables['latitude'][:]
lon_6 = data_6.variables['longitude'][:]

time_6 = data_6.variables['time']
dtime_6 = nc.num2date(time_6[:],time_6.units)
#cf_dtime_6 = nc.num2date(time_6[:],time_6.units, time_6.calendar)
#dtime_6 = np.array([nc_time_axis.CalendarDateTime(item, "360_day") for item in cf_dtime_6])

#wetland ems
data_wet_6 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_wet_ems_{dates}.nc')
#extract variables to arrays
ch4_wet_6 = data_wet_6.variables[stash_wet_ems][:]


# ### u-cb039

# In[13]:


suite_id = 'u-cb039'
suite_lab = 'ucb039'
dates = '2015_2050'

data_5 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_tot_ems_{dates}.nc')
#extract variables to arrays
ch4_anthro_5 = data_5.variables[stash_ch4_ems][:]
lat_5 = data_5.variables['latitude'][:]
lon_5 = data_5.variables['longitude'][:]

time_5 = data_5.variables['time']
dtime_5 = nc.num2date(time_5[:],time_5.units)
#cf_dtime_5 = nc.num2date(time_5[:],time_5.units, time_5.calendar)
#dtime_5 = np.array([nc_time_axis.CalendarDateTime(item, "360_day") for item in cf_dtime_5])

#wetland ems
data_wet_5 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_wet_ems_{dates}.nc')
#extract variables to arrays
ch4_wet_5 = data_wet_5.variables[stash_wet_ems][:]


# In[14]:


hist_biomass_data = nc.Dataset(f'/gws/nopw/j04/htap2/znjs2/cmip6_ancils/hist/hist_ch4_biomass_ems_input.nc')
hist_biomass_ems = hist_biomass_data.variables['emissions_CH4_biomass_low'][:]


# In[17]:


area = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/jmw240/covid_crunch/area/areacella_fx_UKESM1-0-LL_piControl_r1i1p1f2_gn.nc')
box_area = area.variables['areacella'][:]


# ## CH4 emissions

# In[15]:


#function to convert emissions from kg m-2 s-1 to tg per year globally
def ems_convert_tg_yr(ems, area):#emissions in kg m-2 s-1
    per_sec_to_per_yr = 60*60*24*360 #define conversion factor
    ems_kg_s = np.multiply(ems, area) #emissions per box
    ems_kg_yr = ems_kg_s*per_sec_to_per_yr
    ems_tot_tg_yr = np.sum(ems_kg_yr, axis=(1,2))/1e9
    return ems_tot_tg_yr, ems_kg_yr


# ### SSP370

# In[18]:


#ssp370
ch4_ems_tot_tg_yr_bo797, ch4_ems_tot_kg_yr_bo797 = ems_convert_tg_yr(ch4_anthro_2, box_area)
ch4_wet_ems_tot_tg_yr_370, ch4_wet_ems_kg_yr_370 = ems_convert_tg_yr(ch4_wet_2, box_area)

# ca723
ch4_ems_tot_tg_yr_ca723, ch4_ems_tot_kg_yr_ca723 = ems_convert_tg_yr(ch4_anthro_6, box_area)
ch4_wet_ems_tot_tg_yr_723, ch4_wet_ems_kg_yr_473 = ems_convert_tg_yr(ch4_wet_6, box_area)

# cb039
ch4_ems_tot_tg_yr_cb039, ch4_ems_tot_kg_yr_cb039 = ems_convert_tg_yr(ch4_anthro_5, box_area)
ch4_wet_ems_tot_tg_yr_039, ch4_wet_ems_kg_yr_039 = ems_convert_tg_yr(ch4_wet_5, box_area)

#calculate anthro emissions from total
ch4_ems_tot_tg_yr_370 = np.mean([ch4_ems_tot_tg_yr_cb039,ch4_ems_tot_tg_yr_ca723,ch4_ems_tot_tg_yr_bo797],axis=0)
ch4_ems_anthro_tg_yr_370 = ch4_ems_tot_tg_yr_370 - ch4_wet_ems_tot_tg_yr_370


# ### NZAME

# In[19]:


#by186
ch4_ems_tot_tg_yr_186, ch4_ems_tot_kg_yr_186 = ems_convert_tg_yr(ch4_anthro_1, box_area)
ch4_wet_ems_tot_tg_yr_186, ch4_wet_ems_kg_yr_186 = ems_convert_tg_yr(ch4_wet_1, box_area)

# bz146
ch4_ems_tot_tg_yr_146, ch4_ems_tot_kg_yr_146 = ems_convert_tg_yr(ch4_anthro_4, box_area)
ch4_wet_ems_tot_tg_yr_146, ch4_wet_ems_kg_yr_146 = ems_convert_tg_yr(ch4_wet_4, box_area)

# bz473
ch4_ems_tot_tg_yr_473, ch4_ems_tot_kg_yr_473 = ems_convert_tg_yr(ch4_anthro_3, box_area)
ch4_wet_ems_tot_tg_yr_473, ch4_wet_ems_kg_yr_473 = ems_convert_tg_yr(ch4_wet_3, box_area)

#ens mean
ch4_wet_ems_tot_tg_yr = np.mean([ch4_wet_ems_tot_tg_yr_146,ch4_wet_ems_tot_tg_yr_473,ch4_wet_ems_tot_tg_yr_186], axis=0)

ens_mean_tot_ems = np.mean([ch4_ems_tot_tg_yr_186,ch4_ems_tot_tg_yr_146,ch4_ems_tot_tg_yr_473],axis=0)


# In[20]:


#hist emissions
ch4_hist_biomass_tot_tg_yr, ch4_hist_biomass_kg_yr = ems_convert_tg_yr(hist_biomass_ems[1:,0,:,:], box_area)#select from 1850 and only surface


# ## Historical emissions

# In[21]:


bl593_tot_ch4_ems = np.load(f'{emissions_loc}ch4_anthro_ems_1850_2014.npy')
bl593_wet_ch4_ems = np.load(f'{emissions_loc}ch4_wetland_ems_1850_2014.npy')
bl593_dates = np.load(f'{emissions_loc}ems_dates_1850_2014.npy')
bl593_anthro_ch4_ems = bl593_tot_ch4_ems - bl593_wet_ch4_ems


# In[22]:


#total emissions
total_no_anthro_ems = ens_mean_tot_ems
total_ssp_ems = ch4_ems_tot_tg_yr_370 #ch4_wet_ems_tot_tg_yr_370 + ch4_ems_anthro_tg_yr_370
ems_diff = total_ssp_ems - total_no_anthro_ems

#biogenic emissions, constant at 50 Tg per year
biogenic_ems_ssp = np.full(ch4_wet_ems_tot_tg_yr.shape,50)
biogenic_ems_hist = np.full(bl593_anthro_ch4_ems.shape,50)

#biomass emissions
ssp_biomass_ems = (ens_mean_tot_ems - ch4_wet_ems_tot_tg_yr) - biogenic_ems_ssp
hist_biomass_ems = ch4_hist_biomass_tot_tg_yr[:-2]


# In[23]:


#np_output = '/home/users/zosiast/scripts/no_anthro_ensemble/emissions_numpy/'
#np.save(f'{np_output}nzame_wet_ems_ens_mean.npy',ch4_wet_ems_tot_tg_yr)
#np.save(f'{np_output}nzame_total_ems_ens_mean.npy',total_no_anthro_ems)
#np.save(f'{np_output}nzame_biogenic_ems.npy', biogenic_ems_ssp)
#np.save(f'{np_output}hist_biogenic_ems.npy', biogenic_ems_hist)
#np.save(f'{np_output}nzame_biomass_ems.npy', ssp_biomass_ems)
#np.save(f'{np_output}hist_biomass_ems.npy', np.array(hist_biomass_ems))
#np.save(f'{np_output}ems_diff_ssp370_nzame.npy', ems_diff)
#np.save(f'{np_output}hist_anthro_ems.npy', bl593_anthro_ch4_ems)
#np.save(f'{np_output}hist_wet_ems.npy', bl593_wet_ch4_ems)
#np.save(f'{np_output}years_ssp.npy', years_1)
#np.save(f'{np_output}years_hist.npy', years_hist)


# In[24]:


rcParams['font.size'] = 16


# In[26]:


fig = plt.figure(dpi=200)#usually 200)
ax = plt.axes()

years_1 = pd.DatetimeIndex(dtime_1).year
years_hist = pd.DatetimeIndex(bl593_dates).year

plt.bar(years_1, ch4_wet_ems_tot_tg_yr, width=0.6, color = 'orange', label='Wetland')
ax.bar(years_1, biogenic_ems_ssp, width=0.6, label='Non-wetland natural', color = 'teal',bottom=ch4_wet_ems_tot_tg_yr)
ax.bar(years_1, ssp_biomass_ems, width=0.6, label='Biomass burning', color = 'chocolate',bottom=ch4_wet_ems_tot_tg_yr+50)


ax.bar(years_hist, (bl593_anthro_ch4_ems-biogenic_ems_hist-hist_biomass_ems), width=0.6, color='plum', 
       bottom=bl593_wet_ch4_ems+biogenic_ems_hist+hist_biomass_ems,
      label = 'Anthropogenic')
ax.bar(years_hist, hist_biomass_ems, width=0.6, color='chocolate', bottom=bl593_wet_ch4_ems+ biogenic_ems_hist)
ax.bar(years_hist, biogenic_ems_hist, width=0.6, color='teal', bottom=bl593_wet_ch4_ems)
ax.bar(years_hist, bl593_wet_ch4_ems, width=0.6, color = 'orange')
ax.vlines((years_hist[-1]+years_1[0])/2, 0, 860, color='k', linestyle = 'dashed')

ax.bar(years_1, ems_diff, width=0.6, label='Removed anthropogenic', 
       color = 'lightgrey',bottom=total_no_anthro_ems)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_ylim(0,860)
ax.set_xlim(1985,2050)
plt.legend(fontsize=12,bbox_to_anchor=(0.45,0.35))
plt.ylabel(" CH$_4$ emissions / Tg")
plt.xlabel("Year")
#plt.title('CH4 emissions 1850-2050: net zero anthro CH4')
plt.savefig(f'{output_write}ch4_ems_no_anthro_1850_2050.png',bbox_inches='tight')


# In[ ]:




