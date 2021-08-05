#!/usr/bin/env python
# coding: utf-8

import numpy as np
import netCDF4 as nc
import pylab as plt
import matplotlib
import numpy.ma as ma
from matplotlib import rcParams
import datetime
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import matplotlib.dates as mdates
import pandas as pd
from scipy.optimize import curve_fit
import cftime
import nc_time_axis
import xarray as xr


rcParams['font.size'] = 16

# In[2]:
var_lab = 'CH$_4$'
#output for plots
output_write = '/home/users/zosiast/jasmin_plots/no_anthro_ensemble_ssp370/output_plots_aug/'

#constants
mr_ch4 = 16.
mr_co = 28.
mr_o3 = 48.
per_sec_to_per_yr = 60*60*24*360
g_to_Tg = 1e12
n_a = 6.022e23

#stash codes
stash_ch4 = 'mass_fraction_of_methane_in_air'
stash_co = 'mass_fraction_of_carbon_monoxide_in_air'
stash_ch4_oh = 'm01s50i041'
stash_oh = 'mass_fraction_of_hydroxyl_radical_in_air'
stash_o3 = 'mass_fraction_of_ozone_in_air'
stash_trop = 'm01s50i062'
stash_mass = 'm01s50i063'
stash_temp = 'surface_temperature'


# # 2015-2050: net zero anthro

# ### u-by186

suite_id = 'u-by186'
suite_lab = 'uby186'
dates = '2015_2050'

data_1 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_{dates}.nc')
#extract variables to arrays
ch4_1 = data_1.variables[stash_ch4][:]*28/mr_ch4*1e9
lat_1 = data_1.variables['latitude'][:]
lon_1 = data_1.variables['longitude'][:]

time_1 = data_1.variables['time']
cf_dtime_1 = cftime.num2date(time_1[:],time_1.units, time_1.calendar)
dtime_1 = np.array([nc_time_axis.CalendarDateTime(item, "360_day") for item in cf_dtime_1])

#tropospheric mask
data_trop_1 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_trop_mask_{dates}.nc')
trop_1 = data_trop_1.variables[stash_trop][:]

#air mass
data_mass_1 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_airmass_{dates}.nc')
mass_1 = data_mass_1.variables[stash_mass][:]

#temperature starts in 2014
data_temp_1 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_surf_temp_2015_2050.nc')
temp_1 = data_temp_1.variables[stash_temp][:]

#ch4_oh data
data_ch4_oh = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_oh_flux_{dates}.nc')
ch4_oh_1 = data_ch4_oh.variables[stash_ch4_oh][:]

#o3_conc
data_o3 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_o3_{dates}.nc')
o3_1 = data_o3.variables[stash_o3][:]

#oh_conc
data_oh = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_oh_{dates}.nc')
oh_1 = data_oh.variables[stash_oh][:]


# ### u-bz146

suite_id = 'u-bz146'
suite_lab = 'ubz146'
dates = '2015_2050'

data_146 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_{dates}.nc')
#extract variables to arrays
ch4_146 = data_146.variables[stash_ch4][:]*28/mr_ch4*1e9
lat_146 = data_146.variables['latitude'][:]
lon_146 = data_146.variables['longitude'][:]

time_146 = data_146.variables['time']
cf_dtime_146 = cftime.num2date(time_146[:],time_146.units, time_146.calendar)
dtime_146 = np.array([nc_time_axis.CalendarDateTime(item, "360_day") for item in cf_dtime_146])

#tropospheric mask
data_trop_146 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_trop_mask_{dates}.nc')
trop_146 = data_trop_146.variables[stash_trop][:]

#air mass
data_mass_146 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_airmass_{dates}.nc')
mass_146 = data_mass_146.variables[stash_mass][:]

#temperature starts in 2014
data_temp_146 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_surf_temp_{dates}.nc')
temp_146 = data_temp_146.variables[stash_temp][:]

#ch4_oh
data_ch4_oh_146 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_oh_flux_{dates}.nc')
ch4_oh_146 = data_ch4_oh_146.variables[stash_ch4_oh][:]

#o3 conc
data_o3_146 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_o3_{dates}.nc')
o3_146 = data_o3_146.variables[stash_o3][:]

#oh_conc
data_oh_146 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_oh_{dates}.nc')
oh_146 = data_oh_146.variables[stash_oh][:]

#### u-bz473

suite_id = 'u-bz473'
suite_lab = 'ubz473'
dates = '2015_2050'

data_473 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_{dates}.nc')
#extract variables to arrays
ch4_473 = data_473.variables[stash_ch4][:]*28/mr_ch4*1e9
lat_473 = data_473.variables['latitude'][:]
lon_473 = data_473.variables['longitude'][:]

time_473 = data_473.variables['time']
cf_dtime_473 = cftime.num2date(time_473[:],time_473.units, time_473.calendar)
dtime_473 = np.array([nc_time_axis.CalendarDateTime(item, "360_day") for item in cf_dtime_473])

#tropospheric mask
data_trop_473 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_trop_mask_{dates}.nc')
trop_473 = data_trop_473.variables[stash_trop][:]

#air mass
data_mass_473 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_airmass_{dates}.nc')
mass_473 = data_mass_473.variables[stash_mass][:]

#temperature starts in 2014
data_temp_473 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_surf_temp_{dates}.nc')
temp_473 = data_temp_473.variables[stash_temp][:]

#ch4_oh
data_ch4_oh_473 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_oh_flux_{dates}.nc')
ch4_oh_473 = data_ch4_oh_473.variables[stash_ch4_oh][:]

#o3_conc
data_o3_473 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_o3_{dates}.nc')
o3_473 = data_o3_473.variables[stash_o3][:]

#oh_conc
data_oh_473 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_oh_{dates}.nc')
oh_473 = data_oh_473.variables[stash_oh][:]


# ## SSP370 counterfactual: u-bo797

suite_id = 'u-bo797'
suite_lab = 'ubo797'
dates = '2015_2050'

#ssp370 u-bo797
data_3 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_{dates}.nc')
#extract variables to arrays
ch4_3 = data_3.variables['mass_fraction_of_methane_in_air'][:-1,:,:,:]*28/mr_ch4*1e9
lat_3 = data_3.variables['latitude'][:]
lon_3 = data_3.variables['longitude'][:]

time_3 = data_3.variables['time']
cf_dtime_3 = cftime.num2date(time_3[:-1],time_3.units, time_3.calendar)
dtime_3 = np.array([nc_time_axis.CalendarDateTime(item, "360_day") for item in cf_dtime_3])

#surface temperature
data_temp_3 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ann_temp_2015_2050.nc')
temp_3 = data_temp_3.variables[stash_temp][:]

#tropospheric mask
data_trop_3 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_trop_mask_{dates}.nc')
trop_3 = data_trop_3.variables[stash_trop][:-1,:,:,:]

#air mass
data_mass_3 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_airmass_{dates}.nc')
mass_3 = data_mass_3.variables[stash_mass][:-1,:,:,:]

#ch4_oh
data_ch4_oh_3 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_oh_flux_{dates}.nc')
ch4_oh_3 = data_ch4_oh_3.variables[stash_ch4_oh][:-1,:,:,:]

#o3_conc
data_o3_3 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_o3_{dates}.nc')
o3_3 = data_o3_3.variables[stash_o3][:-1,:,:,:]

#oh_conc
data_oh_3 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_oh_{dates}.nc')
oh_3 = data_oh_3.variables[stash_oh][:-1,:,:,:]

# for some reason some of them are up to 2051 so the last year is removed


# ### u-ca723 SSP370

suite_id = 'u-ca723'
suite_lab = 'uca723'
dates = '2015_2050'

#ssp370 u-bo797
data_4 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_{dates}.nc')
#extract variables to arrays
ch4_4 = data_4.variables['mass_fraction_of_methane_in_air'][:,:,:,:]*28/mr_ch4*1e9
lat_4 = data_4.variables['latitude'][:]
lon_4 = data_4.variables['longitude'][:]

time_4 = data_4.variables['time']
cf_dtime_4 = cftime.num2date(time_4[:],time_4.units, time_4.calendar)
dtime_4 = np.array([nc_time_axis.CalendarDateTime(item, "360_day") for item in cf_dtime_4])

#surface temperature
data_temp_4 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_surf_temp_2015_2050.nc')
temp_4 = data_temp_4.variables[stash_temp][:]

#tropospheric mask
data_trop_4 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_trop_mask_{dates}.nc')
trop_4 = data_trop_4.variables[stash_trop][:,:,:,:]

#air mass
data_mass_4 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_airmass_{dates}.nc')
mass_4 = data_mass_4.variables[stash_mass][:,:,:,:]

#ch4_oh
data_ch4_oh_4 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_oh_flux_{dates}.nc')
ch4_oh_4 = data_ch4_oh_4.variables[stash_ch4_oh][:,:,:,:]

#o3_conc
data_o3_4 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_o3_{dates}.nc')
o3_4 = data_o3_4.variables[stash_o3][:]

#oh_conc
data_oh_4 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_oh_{dates}.nc')
oh_4 = data_oh_4.variables[stash_oh][:]

# ### u-cb039 SSP370

suite_id = 'u-cb039'
suite_lab = 'ucb039'
dates = '2015_2050'

#ssp370 
data_5 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_{dates}.nc')
#extract variables to arrays
ch4_5 = data_5.variables['mass_fraction_of_methane_in_air'][:,:,:,:]*28/mr_ch4*1e9
lat_5 = data_5.variables['latitude'][:]
lon_5 = data_5.variables['longitude'][:]

time_5 = data_5.variables['time']
cf_dtime_5 = cftime.num2date(time_5[:],time_5.units, time_5.calendar)
dtime_5 = np.array([nc_time_axis.CalendarDateTime(item, "360_day") for item in cf_dtime_5])

#surface temperature
data_temp_5 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_surf_temp_2015_2050.nc')
temp_5 = data_temp_5.variables[stash_temp][:]

#tropospheric mask
data_trop_5 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_trop_mask_{dates}.nc')
trop_5 = data_trop_5.variables[stash_trop][:,:,:,:]

#air mass
data_mass_5 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_airmass_{dates}.nc')
mass_5 = data_mass_5.variables[stash_mass][:,:,:,:]

#ch4_oh
data_ch4_oh_5 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_oh_flux_{dates}.nc')
ch4_oh_5 = data_ch4_oh_5.variables[stash_ch4_oh][:,:,:,:]

#o3_conc
data_o3_5 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_o3_{dates}.nc')
o3_5 = data_o3_5.variables[stash_o3][:]

#oh_conc
data_oh_5 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_oh_{dates}.nc')
oh_5 = data_oh_5.variables[stash_oh][:]


#SSP126: u-bo812

suite_id = 'u-bo812'
suite_lab = 'ubo812'
dates = '2015_2050'

data_6 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_{dates}.nc')
#extract variables to arrays
ch4_6 = data_6.variables['mass_fraction_of_methane_in_air'][:-1,:,:,:]*28/mr_ch4*1e9
lat_6 = data_6.variables['latitude'][:]
lon_6 = data_6.variables['longitude'][:]

time_6 = data_6.variables['time']
cf_dtime_6 = cftime.num2date(time_6[:-1],time_6.units, time_6.calendar)
dtime_6 = np.array([nc_time_axis.CalendarDateTime(item, "360_day") for item in cf_dtime_6])

#surface temperature
data_temp_6 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_surf_temp_{dates}.nc')
temp_6 = data_temp_6.variables[stash_temp][:-1,:,:]

#tropospheric mask
data_trop_6 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_trop_mask_{dates}.nc')
trop_6 = data_trop_6.variables[stash_trop][:-1,:,:,:]

#air mass
data_mass_6 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_airmass_{dates}.nc')
mass_6 = data_mass_6.variables[stash_mass][:-1,:,:,:]

#ch4_oh
data_ch4_oh_6 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_ch4_oh_flux_{dates}.nc')
ch4_oh_6 = data_ch4_oh_6.variables[stash_ch4_oh][:-1,:,:,:]

#o3_conc
data_o3_6 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_o3_{dates}.nc')
o3_6 = data_o3_6.variables[stash_o3][:-1,:,:,:]

#oh_conc
data_oh_6 = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/znjs2/{suite_id}/netcdf/{suite_lab}_oh_{dates}.nc')
oh_6 = data_oh_6.variables[stash_oh][:-1,:,:,:]

# ## Area and volume datasets

area = nc.Dataset(f'/gws/nopw/j04/ukca_vol1/jmw240/covid_crunch/area/areacella_fx_UKESM1-0-LL_piControl_r1i1p1f2_gn.nc')
box_area = area.variables['areacella'][:]

total_area = np.sum(box_area)
area_scaled = box_area/total_area
lat_area_scaled = area_scaled[:,0]#1D array of latitude scaling values

# Altitude data

alt_data = nc.Dataset(f'/home/users/zosiast/vol_n96.nc')
alt = alt_data.variables['level_height'][:]
vol = alt_data.variables['grid_cell_volume'][:]

# Population data
pop_data = nc.Dataset(f'/home/users/zosiast/ssp_population/pop_ssp3_n96_2006_2100.nc')
pop_2006_2100 = pop_data.variables['population'][:]

pop_xr = xr.open_dataset(f'/home/users/zosiast/ssp_population/pop_ssp3_n96_2006_2100.nc', decode_times=False)

pop_xr_time = pop_xr.variables['time']
year = np.array(pop_xr_time + 1661,dtype='int')

#select relevant years for population data
pop_data_2015_2050 = pop_2006_2100[9:45,:,:]

#ssp126 population data
pop_data_ssp1 = nc.Dataset(f'/home/users/zosiast/ssp_population/pop_ssp1_n96_2006_2100.nc')
pop_2006_2100_ssp1 = pop_data_ssp1.variables['population'][:]

pop_xr_1 = xr.open_dataset(f'/home/users/zosiast/ssp_population/pop_ssp1_n96_2006_2100.nc', decode_times=False)
#test = xr.decode_cf(pop_xr)
pop_xr_time_1 = pop_xr.variables['time']
year_1 = np.array(pop_xr_time + 1661,dtype='int')
pop_data_2015_2050_ssp1 = pop_2006_2100_ssp1[9:45,:,:]

print('All data read in')

#define functions
def molec_cm3(conc_kg_kg, mass, vol, mr):
    molec_box = conc_kg_kg*mass/mr*1000*n_a #molecules per box
    molec_cm3 = molec_box/(vol*1e6) #molec per cm3
    return molec_cm3


# ## CH4 burden

#sum of all CH4 including stratosphere
#ens 1
ch4_bl593_kg = np.multiply(mass_1,ch4_1/28*mr_ch4/1e9) #ch4 in kg
ch4_bur_bl593 = ch4_bl593_kg.sum(axis=(1,2,3))/1e9 # sum over lat, lon, alt, in Tg

#ens 2
ch4_bz146_kg = np.multiply(mass_146,ch4_146/28*mr_ch4/1e9) #ch4 in kg
ch4_bur_bz146 = ch4_bz146_kg.sum(axis=(1,2,3))/1e9 # sum over lat, lon, alt, in Tg

#ens 3
ch4_bz473_kg = np.multiply(mass_473,ch4_473/28*mr_ch4/1e9) #ch4 in kg
ch4_bur_bz473 = ch4_bz473_kg.sum(axis=(1,2,3))/1e9 # sum over lat, lon, alt, in Tg

ch4_bur_ens_mean = np.mean([ch4_bur_bz473,ch4_bur_bz146,ch4_bur_bl593],axis = 0)

#ssp370
ch4_bo797_kg = np.multiply(mass_3,ch4_3/28*mr_ch4/1e9) #ch4 in kg
ch4_bur_bo797 = ch4_bo797_kg.sum(axis=(1,2,3))/1e9 # sum over lat, lon, alt, in Tg

ch4_ca723_kg = np.multiply(mass_4,ch4_4/28*mr_ch4/1e9) #ch4 in kg
ch4_bur_ca723 = ch4_ca723_kg.sum(axis=(1,2,3))/1e9 # sum over lat, lon, alt, in Tg

ch4_cb039_kg = np.multiply(mass_5,ch4_5/28*mr_ch4/1e9) #ch4 in kg
ch4_bur_cb039 = ch4_cb039_kg.sum(axis=(1,2,3))/1e9 # sum over lat, lon, alt, in Tg

ch4_bur_ssp_mean = np.mean([ch4_bur_bo797,ch4_bur_ca723,ch4_bur_cb039],axis = 0)

ch4_bo812_kg = np.multiply(mass_6,ch4_6/28*mr_ch4/1e9) #ch4 in kg
ch4_bur_bo812 = ch4_bo812_kg.sum(axis=(1,2,3))/1e9 # sum over lat, lon, alt, in Tg


#methane lifetime calc

def ch4_lifetime_calc(ch4_bur, oh_ch4_flux, trop_nc):#ch4 in kg #oh in ppb
    
    #make trop mask
    trop_mask = ma.masked_where(trop_nc < 0.9999999, trop_nc)
    
    
    #flux calcs
    ch4_oh_tg_yr = oh_ch4_flux*mr_ch4*per_sec_to_per_yr/g_to_Tg
    ch4_oh_trop = ma.masked_where(trop_mask.mask,ch4_oh_tg_yr)
    #sum over lat lon alt
    flux_sum = np.sum(ch4_oh_trop, axis=(1,2,3))
    #flux_sum_all = np.sum(ch4_oh_tg_yr, axis=(1,2,3))

    #sum of only tropospheric methane
    #ch4_kg_trop = ma.masked_where(trop_mask.mask, ch4_kg)
    #bur_trop = ch4_kg_trop.sum(axis=(1,2,3))/1e9 # sum over lat, lon, alt, in Tg
    
    #lifetime calc
    t_ch4_strat_trop = ch4_bur/flux_sum
    
    return t_ch4_strat_trop, flux_sum

# calculate methane lifetime
t_ch4_by186, flux_by186 = ch4_lifetime_calc(ch4_bur_bl593,ch4_oh_1, trop_1)
t_ch4_bz146, flux_bz146 = ch4_lifetime_calc(ch4_bur_bz146,ch4_oh_146, trop_146)
t_ch4_bz473, flux_bz473 = ch4_lifetime_calc(ch4_bur_bz473,ch4_oh_473, trop_473)

t_ch4_bo797, flux_bo797 = ch4_lifetime_calc(ch4_bur_bo797,ch4_oh_3, trop_3)
t_ch4_ca723, flux_ca723 = ch4_lifetime_calc(ch4_bur_ca723,ch4_oh_4, trop_4)
t_ch4_cb039, flux_cb039 = ch4_lifetime_calc(ch4_bur_cb039,ch4_oh_5, trop_5)

t_ch4_bo812, flux_bo812 = ch4_lifetime_calc(ch4_bur_bo812,ch4_oh_6, trop_6)


# methane lifetime ens mean
t_ch4_ens_mean = np.mean([t_ch4_by186,t_ch4_bz146,t_ch4_bz473],axis=0)
t_ch4_ssp_mean = np.mean([t_ch4_bo797,t_ch4_ca723,t_ch4_cb039],axis=0)

#Fig: methane lifetime
fig = plt.figure(dpi=200)
ax = plt.axes()

ax.plot(dtime_1, t_ch4_by186, c='skyblue',linewidth=1)
ax.plot(dtime_146, t_ch4_bz146, c='skyblue',linewidth=1)
ax.plot(dtime_473, t_ch4_bz473, c='skyblue',linewidth=1)

ax.plot(dtime_473, t_ch4_ens_mean, label = 'NZAME',c='C0')

ax.plot(dtime_6, t_ch4_bo812, c='#1d3354',linewidth=1, label = 'SSP1-2.6')

ax.plot(dtime_3, t_ch4_bo797, c='pink')
ax.plot(dtime_4, t_ch4_ca723, c='pink')
ax.plot(dtime_5, t_ch4_cb039, c='pink')
ax.plot(dtime_3, t_ch4_ssp_mean, label = 'SSP370', c='#f11111')
ax.plot([dtime_1[0],dtime_1[-1]], [9.823, 9.823], 'grey',linestyle=':', label = 'PI level')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend()
plt.xlabel('Time')
plt.ylabel('CH$_4$ lifetime / years')
#plt.title('CH$_4$ lifetime 2015-2050')
plt.savefig(f'{output_write}ch4_lifetime_2015_2050.png')

print('Lifetime calcs and plots done')

# ## NOAA obs data

#1984-2019 obs data
noaa_ch4 = pd.read_csv(f'/home/users/zosiast/scripts/noaa_annual_average_methane.csv',header=0,skiprows=1,usecols=[1,2,3],index_col=0)
noaa_years = np.array(noaa_ch4.index, dtype='str')
noaa_mean_conc = np.array(noaa_ch4['mean'])

# ## 2014 data points, from historical runs
# in ppb
ch4_2014_bl593 = 1633.55
ch4_2014_bl998 = 1617.53
ch4_2014_bn213 = 1620.09
ch4_2014_ens_mean = np.mean([ch4_2014_bl593,ch4_2014_bl998,ch4_2014_bn213])

# In[29]:
d_time_2014 = cftime.datetime(year=2014, month=6, day=1)
nc_dtime_2014 = nc_time_axis.CalendarDateTime(d_time_2014, "360_day")
dtime_2014_2050 = np.insert(dtime_1,0, nc_dtime_2014)

# ## CH4 surface conc

# mean surface conc
#area weighted by latitude and mean over lon
ch4_lat_weighted_by186 = np.mean(np.average(ch4_1[:,0,:,:],axis=(1),weights = lat_area_scaled), axis = 1)
ch4_lat_weighted_bz146 = np.mean(np.average(ch4_146[:,0,:,:],axis=(1),weights = lat_area_scaled), axis = 1)
ch4_lat_weighted_bz473 = np.mean(np.average(ch4_473[:,0,:,:],axis=(1),weights = lat_area_scaled), axis = 1)

ch4_lat_weighted_ens = np.mean([ch4_lat_weighted_by186,ch4_lat_weighted_bz146,ch4_lat_weighted_bz473],axis=0)

ch4_lat_weighted_bo797 = np.mean(np.average(ch4_3[:,0,:,:],axis=(1),weights = lat_area_scaled), axis = 1)
ch4_lat_weighted_ca723 = np.mean(np.average(ch4_4[:,0,:,:],axis=(1),weights = lat_area_scaled), axis = 1)
ch4_lat_weighted_cb039 = np.mean(np.average(ch4_5[:,0,:,:],axis=(1),weights = lat_area_scaled), axis = 1)

ch4_lat_weighted_ssp = np.mean([ch4_lat_weighted_bo797,ch4_lat_weighted_ca723,ch4_lat_weighted_cb039],axis=0)

ch4_lat_weighted_bo812 = np.mean(np.average(ch4_6[:,0,:,:],axis=(1),weights = lat_area_scaled), axis = 1)


# 2014-2050 area weighted CH4 surface conc with obs


fig = plt.figure(dpi=200)
ax = plt.axes()

ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_by186,0, ch4_2014_bl593),  c='skyblue',linewidth=1)
ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_bz146,0,ch4_2014_bn213),  c='skyblue',linewidth=1)
ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_bz473,0, ch4_2014_bl998),  c='skyblue',linewidth=1)

ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_ens,0,ch4_2014_ens_mean), label = 'NZAME', c='C0',linewidth=1)

ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_bo797,0, ch4_2014_bl593), c='pink')
ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_ca723,0, ch4_2014_bl998), c='pink')
ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_cb039,0,ch4_2014_bn213), c='pink')
ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_ssp,0,ch4_2014_ens_mean), label = 'SSP3-7.0', c='#f11111')

ax.scatter(dtime_2014_2050[:6], noaa_mean_conc[-6:], c='darkslategray', label = 'NOAA obs', marker='+')

ax.plot([dtime_2014_2050[0],dtime_2014_2050[-1]], [776, 776], 'grey',linestyle=':', label = 'PI level')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend()
plt.xlabel('Time')
plt.ylabel(f'Mean surface [{var_lab}] / ppb')
#plt.title(f'Mean surface {var_lab} 2015-2050')
#plt.savefig(f'{output_write}ch4_surface_conc_obs_2015_2050.png')


# Normalise to year 2000 values

# year 2000 conc from bl593 hist run
ch4_2000_bl593 = 1537.89
ch4_2000_noaa = noaa_mean_conc[-20]
pi_ratio_2000 = 776./ch4_2000_bl593


# Normalised surface conc

fig = plt.figure(dpi=200)
ax = plt.axes()

ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_by186,0, ch4_2014_bl593)/ch4_2000_bl593,  c='skyblue',linewidth=1)
ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_bz146,0,ch4_2014_bn213)/ch4_2000_bl593,  c='skyblue',linewidth=1)
ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_bz473,0, ch4_2014_bl998)/ch4_2000_bl593,  c='skyblue',linewidth=1)

ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_bo812,0, ch4_2014_bl998)/ch4_2000_bl593,  label='SSP1-2.6', c='#1d3354',linewidth=1)

ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_ens,0,ch4_2014_ens_mean)/ch4_2000_bl593, label = 'NZAME', c='C0',linewidth=1)

ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_bo797,0, ch4_2014_bl593)/ch4_2000_bl593, c='pink')
ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_ca723,0, ch4_2014_bl998)/ch4_2000_bl593, c='pink')
ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_cb039,0,ch4_2014_bn213)/ch4_2000_bl593, c='pink')
ax.plot(dtime_2014_2050, np.insert(ch4_lat_weighted_ssp,0,ch4_2014_ens_mean)/ch4_2000_bl593, label = 'SSP3-7.0', c='#f11111')

ax.plot([dtime_2014_2050[0],dtime_2014_2050[-1]], [pi_ratio_2000, pi_ratio_2000], 'grey',linestyle=':', label = 'PI level')

ax.scatter(dtime_2014_2050[:6], noaa_mean_conc[-6:]/ch4_2000_noaa, c='darkslategray', label = 'NOAA obs', marker='+')


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend()
plt.xlabel('Time')
plt.ylabel(f'Mean surface [{var_lab}] / year 2000 [CH4]')
#plt.title(f'Mean surface {var_lab} 2015-2050 relative to year 2000')
plt.savefig(f'{output_write}ch4_relative_surface_conc_2015_2050.png')



# GMST plots


# ## GMST calculation

#by186 temp meaning
temp_1_lon_mean = np.mean(temp_1, axis = 2)
mean_sur_temp_186 = np.average(temp_1_lon_mean,weights = lat_area_scaled, axis = (1))

temp_146_lon_mean = np.mean(temp_146, axis = 2)
mean_sur_temp_146 = np.average(temp_146_lon_mean,weights = lat_area_scaled, axis = (1))

temp_473_lon_mean = np.mean(temp_473, axis = 2)
mean_sur_temp_473 = np.average(temp_473_lon_mean,weights = lat_area_scaled, axis = (1))

gmst_ens_mean = np.mean([mean_sur_temp_473,mean_sur_temp_146,mean_sur_temp_186],axis = 0)


# In[13]:

#bo797 global mean, area weighted
temp_3_lon_mean = np.mean(temp_3, axis = 2)
mean_sur_temp_bo797 = np.average(temp_3_lon_mean,weights = lat_area_scaled, axis = (1))

#ca723 global mean, area weighted
temp_4_lon_mean = np.mean(temp_4, axis = 2)
mean_sur_temp_ca723 = np.average(temp_4_lon_mean,weights = lat_area_scaled, axis = (1))

temp_5_lon_mean = np.mean(temp_5, axis = 2)
mean_sur_temp_cb039 = np.average(temp_5_lon_mean,weights = lat_area_scaled, axis = (1))

gmst_ens_mean_ssp = np.mean([mean_sur_temp_bo797,mean_sur_temp_ca723,mean_sur_temp_cb039],axis = 0)


# GMST vs time

fig = plt.figure(dpi=200)
ax = plt.axes()

ax.plot(dtime_1, mean_sur_temp_186, c='skyblue',linewidth=1) #temp for by186 is from 2014(?)
ax.plot(dtime_146, mean_sur_temp_146,  c='skyblue',linewidth=1)
ax.plot(dtime_473, mean_sur_temp_473, c='skyblue',linewidth=1)

ax.plot(dtime_473, gmst_ens_mean, label = 'NZAME',c='C0')

ax.plot(dtime_3, mean_sur_temp_bo797, c='pink')
ax.plot(dtime_4, mean_sur_temp_ca723, c='pink')
ax.plot(dtime_5, mean_sur_temp_cb039, c='pink')
ax.plot(dtime_3, gmst_ens_mean_ssp, label = 'SSP3-7.0',c='#f11111')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.legend()
plt.xlabel('Time')
plt.ylabel(f'GMST / K')
#plt.title('GMST 2015-2050')
plt.savefig(f'{output_write}gmst_2015_2050.png')


# GMST anomaly wrt 2015


fig = plt.figure(dpi=200)
ax = plt.axes()

ax.plot(dtime_1, mean_sur_temp_186 - mean_sur_temp_186[0], c='skyblue',linewidth=1) #temp for by186 is from 2014(?)
ax.plot(dtime_146, mean_sur_temp_146 - mean_sur_temp_146[0],  c='skyblue',linewidth=1)
ax.plot(dtime_473, mean_sur_temp_473 - mean_sur_temp_473[0], c='skyblue',linewidth=1)

ax.plot(dtime_473, gmst_ens_mean - gmst_ens_mean[0], label = 'NZAME',c='C0')

ax.plot(dtime_3, mean_sur_temp_bo797 - mean_sur_temp_bo797[0], c='pink')
ax.plot(dtime_4, mean_sur_temp_ca723 - mean_sur_temp_ca723[0], c='pink')
ax.plot(dtime_5, mean_sur_temp_cb039 - mean_sur_temp_cb039[0], c='pink')
ax.plot(dtime_3, gmst_ens_mean_ssp - gmst_ens_mean_ssp[0], label = 'SSP3-7.0',c='#f11111')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.legend()
plt.xlabel('Time')
plt.ylabel(f'GMST anomaly wrt 2015 / K')
#plt.title('GMST anomaly 2015-2050')
plt.savefig(f'{output_write}gmst_anomaly_2015_2050.png')

#gmst map plot
gmst_mean_map_nzame = np.mean([temp_1,temp_146,temp_473],axis = 0)
gmst_mean_map_ssp = np.mean([temp_3,temp_4,temp_5],axis = 0)

from matplotlib.colors import LinearSegmentedColormap,ListedColormap
from matplotlib import cm

top = cm.get_cmap('Blues_r', 128)
bottom = cm.get_cmap('Reds', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 90)),
                       bottom(np.linspace(0, 0.1, 10))))
newcmp = ListedColormap(newcolors, name='Red_blue')


fig = plt.figure(dpi=200)
ax = plt.axes()

ax = plt.axes(projection=ccrs.Robinson(central_longitude=0, globe=None))
ax.set_global()
ax.coastlines(linewidth=0.5)

var_name = 'Surface Temperature$_{NZAME - SSP37.0}$'

ch4_diff_cyclic, lon_plot = add_cyclic_point(np.mean(gmst_mean_map_nzame[-10:,:,:] - gmst_mean_map_ssp[-10:,:,:],
                                                     axis=0), coord=lon_473)

plt.pcolormesh(lon_plot,lat_473,ch4_diff_cyclic, cmap=newcmp,transform=ccrs.PlateCarree(central_longitude=0), vmin=-9, vmax=1)


plt.colorbar(label = f'{var_name} / K',orientation='horizontal',pad=0.05)
#plt.title(f'Surface temperature 2040-2050: NZAME - SSP3-7.0')
plt.savefig(f'{output_write}gmst_surf_map_2040_2050.png')

#Trop mask based on trop mask variable

#tropospheric mask
trop_mask_186 = ma.masked_where(trop_1 < 1, trop_1)
trop_mask_146 = ma.masked_where(trop_146 < 1, trop_146)
trop_mask_473 = ma.masked_where(trop_473 < 1, trop_473)

trop_mask_3 = ma.masked_where(trop_3 < 1, trop_3)


#ozone in ppb
o3_186_ppb = o3_1*28/mr_o3*1e9
o3_146_ppb = o3_146*28/mr_o3*1e9
o3_473_ppb = o3_473*28/mr_o3*1e9

o3_bo797 = o3_3*28/mr_o3*1e9
o3_ca723 = o3_4*28/mr_o3*1e9
o3_cb039 = o3_5*28/mr_o3*1e9

o3_bo812 = o3_6*28/mr_o3*1e9

#surface selected
surf_o3_186 = o3_186_ppb[:,0,:,:]
surf_o3_146 = o3_146_ppb[:,0,:,:]
surf_o3_473 = o3_473_ppb[:,0,:,:]

surf_o3_ens_mean = np.mean([surf_o3_186,surf_o3_146,surf_o3_473],axis=0)

surf_o3_bo797 = o3_bo797[:,0,:,:]
surf_o3_ca723 = o3_ca723[:,0,:,:]
surf_o3_cb039 = o3_cb039[:,0,:,:]

surf_o3_ens_mean_ssp = np.mean([surf_o3_bo797[:36,:,:],surf_o3_ca723,surf_o3_cb039],axis=0)

surf_o3_ssp126 = o3_bo812[:,0,:,:]

#area weighted by latitude and mean over lon
o3_lat_weighted_186 = np.mean(np.average(surf_o3_186,axis=(1),weights = lat_area_scaled), axis = 1)
o3_lat_weighted_146 = np.mean(np.average(surf_o3_146,axis=(1),weights = lat_area_scaled), axis = 1)
o3_lat_weighted_473 = np.mean(np.average(surf_o3_473,axis=(1),weights = lat_area_scaled), axis = 1)

o3_lat_weighted_ens_mean = np.mean([o3_lat_weighted_186,o3_lat_weighted_146,o3_lat_weighted_473],axis=0)

o3_lat_weighted_bo797 = np.mean(np.average(surf_o3_bo797,axis=(1),weights = lat_area_scaled), axis = 1)
o3_lat_weighted_ca723 = np.mean(np.average(surf_o3_ca723,axis=(1),weights = lat_area_scaled), axis = 1)
o3_lat_weighted_cb039 = np.mean(np.average(surf_o3_ca723,axis=(1),weights = lat_area_scaled), axis = 1)

o3_lat_weighted_ens_mean_ssp = np.mean([o3_lat_weighted_bo797[:36],o3_lat_weighted_ca723,o3_lat_weighted_cb039],axis=0)


# o3 surface conc
fig = plt.figure(dpi=200)
ax = plt.axes()

ax.plot(dtime_1, o3_lat_weighted_186, c='skyblue',linewidth=1)
ax.plot(dtime_146, o3_lat_weighted_146,  c='skyblue',linewidth=1)
ax.plot(dtime_473, o3_lat_weighted_473,  c='skyblue',linewidth=1)

ax.plot(dtime_473, o3_lat_weighted_ens_mean, label = 'NZAME', c='C0')

ax.plot(dtime_3,  o3_lat_weighted_bo797[:36], c='pink')
ax.plot(dtime_4, o3_lat_weighted_ca723, c='pink')
ax.plot(dtime_5, o3_lat_weighted_cb039, c='pink')

ax.plot(dtime_4,  o3_lat_weighted_ens_mean_ssp, label = 'SSP370', c='#f11111')
ax.plot([dtime_1[0],dtime_1[-1]], [20, 20], 'grey',linestyle=':', label = 'PI level')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend()
plt.xlabel('Time')
plt.ylabel(f'O$_3$ / ppb')
#plt.title('O$_3$ mean surface conc 2015-2050')
plt.savefig(f'{output_write}o3_surface_area_weighted_2015_2050.png')

#population weighting
#area weighted by population
o3_pop_weighted_186 = np.average(surf_o3_186,axis=(1,2),weights = pop_data_2015_2050)
o3_pop_weighted_146 = np.average(surf_o3_146,axis=(1,2),weights = pop_data_2015_2050)
o3_pop_weighted_473 = np.average(surf_o3_473,axis=(1,2),weights = pop_data_2015_2050)

o3_pop_weighted_ens_mean = np.mean([o3_pop_weighted_186,o3_pop_weighted_146,o3_pop_weighted_473],axis=0)

o3_pop_weighted_bo797 = np.average(surf_o3_bo797[:36,:,:],axis=(1,2),weights = pop_data_2015_2050)
o3_pop_weighted_ca723 = np.average(surf_o3_ca723,axis=(1,2),weights = pop_data_2015_2050)
o3_pop_weighted_cb039 = np.average(surf_o3_cb039,axis=(1,2),weights = pop_data_2015_2050)

o3_pop_weighted_ens_mean_ssp = np.mean([o3_pop_weighted_bo797,o3_pop_weighted_ca723,o3_pop_weighted_cb039],axis=0)

o3_pop_weighted_bo812 = np.average(surf_o3_ssp126,axis=(1,2),weights = pop_data_2015_2050_ssp1)

#ozone surface population weighted
fig = plt.figure(dpi=200)
ax = plt.axes()

ax.plot(dtime_1, o3_pop_weighted_186, c='skyblue',linewidth=1)
ax.plot(dtime_146, o3_pop_weighted_146, c='skyblue',linewidth=1)
ax.plot(dtime_473, o3_pop_weighted_473, c='skyblue',linewidth=1)

ax.plot(dtime_473, o3_pop_weighted_ens_mean, label = 'NZAME', c='C0')

ax.plot(dtime_3,  o3_pop_weighted_bo797, c='pink')
ax.plot(dtime_3,  o3_pop_weighted_ca723, c='pink')
ax.plot(dtime_3,  o3_pop_weighted_cb039, c='pink')
ax.plot(dtime_3,  o3_pop_weighted_bo812, label = 'SSP1-2.6', c='#1d3354')


ax.plot(dtime_3,  o3_pop_weighted_ens_mean_ssp, label = 'SSP3-7.0', c='#f11111')

#ax.plot([dtime_1[0],dtime_1[-1]], [20, 20], 'grey',linestyle=':', label = 'PI level')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend()
plt.xlabel('Time')
plt.ylabel(f'O$_3$ / ppb')
#plt.title('O$_3$ mean surface conc 2015-2050: population weighted')
plt.savefig(f'{output_write}o3_surface_pop_weighted_2015_2050.png')

# ozone surface concentration plot

fig = plt.figure(dpi=100)
ax = plt.axes()

ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.0))
ax.coastlines(linewidth=0.5)

var_name = 'Surface [O$_3$]$_{NZAME - SSP37.0}$'

ch4_diff_cyclic, lon_plot = add_cyclic_point(np.mean(surf_o3_ens_mean[-10:,:,:] - surf_o3_ens_mean_ssp[-10:,:,:],axis=0),
                                             coord=lon_473)

plt.pcolormesh(lon_plot,lat_473,ch4_diff_cyclic, cmap='Oranges_r',vmin = -18, vmax = 0, 
               transform=ccrs.PlateCarree(central_longitude=0))#, cmap='bwr', vmin = -8, vmax = 8)

plt.colorbar(label = f'{var_name} / ppb',orientation='horizontal',pad=0.05)
#plt.title(f'Surface ozone 2040-2050: NZAME - SSP3-7.0')
plt.savefig(f'{output_write}o3_surf_map_2040_2050.png')



#OH concentration

# convert to molec cm -3
oh_molec_cm3_186 = molec_cm3(oh_1, mass_1, vol, 17)
oh_molec_cm3_146 = molec_cm3(oh_146, mass_146, vol, 17)
oh_molec_cm3_473 = molec_cm3(oh_473, mass_473, vol, 17)

oh_molec_cm3_bo797 = molec_cm3(oh_3[:36,:,:,:], mass_3, vol, 17)
oh_molec_cm3_ca723 = molec_cm3(oh_4, mass_4, vol, 17)
oh_molec_cm3_cb039 = molec_cm3(oh_5, mass_5, vol, 17)

oh_molec_cm3_bo812 = molec_cm3(oh_6, mass_6, vol, 17)

#by186 oh
oh_trop_186 = ma.masked_where(o3_186_ppb > 125,oh_molec_cm3_186)
oh_trop_mean_186 = np.average(oh_trop_186, weights=mass_1, axis = (1,2,3))

#by146 oh
oh_trop_146 = ma.masked_where(o3_146_ppb > 125,oh_molec_cm3_146)
oh_trop_mean_146 = np.average(oh_trop_146, weights=mass_146, axis = (1,2,3))

#by473 oh
oh_trop_473 = ma.masked_where(o3_473_ppb > 125,oh_molec_cm3_473)
oh_trop_mean_473 = np.average(oh_trop_473, weights=mass_473, axis = (1,2,3))

#bo797 oh
oh_trop_bo797 = ma.masked_where(o3_bo797 > 125,oh_molec_cm3_bo797)
oh_trop_mean_bo797 = np.average(oh_trop_bo797, weights=mass_3, axis = (1,2,3))

#ca723 oh
oh_trop_ca723 = ma.masked_where(o3_ca723 > 125,oh_molec_cm3_ca723)
oh_trop_mean_ca723 = np.average(oh_trop_ca723, weights=mass_4, axis = (1,2,3))

#cb039 oh
oh_trop_cb039 = ma.masked_where(o3_cb039 > 125,oh_molec_cm3_cb039)
oh_trop_mean_cb039 = np.average(oh_trop_cb039, weights=mass_5, axis = (1,2,3))

#bo812 oh
oh_trop_bo812 = ma.masked_where(o3_bo812 > 125,oh_molec_cm3_bo812)
oh_trop_mean_bo812 = np.average(oh_trop_bo812, weights=mass_5, axis = (1,2,3))

#ensemble mean
oh_trop_ens_mean = np.mean([oh_trop_mean_186,oh_trop_mean_146,oh_trop_mean_473], axis = 0)
oh_trop_ens_mean_ssp = np.mean([oh_trop_mean_bo797,oh_trop_mean_ca723,oh_trop_mean_cb039], axis = 0)


# OH trop mean conc

fig = plt.figure(dpi=200)
ax = plt.axes()

ax.plot(dtime_1, oh_trop_mean_186/1e6, c='skyblue',linewidth=1)
ax.plot(dtime_146, oh_trop_mean_146/1e6,  c='skyblue',linewidth=1)
ax.plot(dtime_473, oh_trop_mean_473/1e6,  c='skyblue',linewidth=1)

ax.plot(dtime_6, oh_trop_mean_bo812/1e6, label='SSP1-2.6', c='#1d3354',linewidth=1)

ax.plot(dtime_473, oh_trop_ens_mean/1e6, label = 'NZAME', c='C0')

ax.plot(dtime_3, oh_trop_mean_bo797/1e6, c='pink')
ax.plot(dtime_4, oh_trop_mean_ca723/1e6, c='pink')
ax.plot(dtime_5, oh_trop_mean_cb039/1e6, c='pink')

ax.plot(dtime_4, oh_trop_ens_mean_ssp/1e6, label = 'SSP3-7.0', c='#f11111')
ax.plot([dtime_1[0],dtime_1[-1]], [0.825, 0.825], 'grey',linestyle=':', label = 'PI level')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.set_xticks(labels)

plt.legend()
plt.xlabel('Time')
plt.ylabel('OH / $10^6$ molec cm$^{-3}$')
#plt.title('OH trop mean conc 2015-2025')
plt.savefig(f'{output_write}oh_trop_mean_2015_2050.png')


## Quantitative print data for 2040-2050

def mean_std_2040_2050(a,b,c,a_ssp,b_ssp,c_ssp,var,units):
    mean_2040_2050_nzame = np.mean([a[-10],b[-10],c[-10]])
    std_2040_2050_nzame = np.mean(np.std([a[-10],b[-10],b[-10]],axis=0))
    
    mean_2040_2050_ssp = np.mean([a_ssp[-10],b_ssp[-10],c_ssp[-10]])
    std_2040_2050_ssp = np.mean(np.std([a_ssp[-10],b_ssp[-10],c_ssp[-10]],axis=0))
    
    diff_2040_2050 = mean_2040_2050_nzame - mean_2040_2050_ssp
    diff_2040_2050_err = np.sqrt(std_2040_2050_nzame**2 + std_2040_2050_ssp**2)
    
    print(f'{var} 2040-2050 NZAME: {mean_2040_2050_nzame:.2f} ± {std_2040_2050_nzame:.2f} {units}')
    print(f'{var} 2040-2050 SSP: {mean_2040_2050_ssp:.2f} ± {std_2040_2050_ssp:.2f} {units}')
    print('SSP increases over 2040-2050')
    print(f'{var} difference 2040-50: {diff_2040_2050:.2f} ± {diff_2040_2050_err:.2f} {units}')
    
    
def mean_std_2050(a,b,c,a_ssp,b_ssp,c_ssp,var,units):
    mean_2050_nzame = np.mean([a[-1],b[-1],c[-1]])
    std_2050_nzame = np.mean(np.std([a[-1],b[-1],b[-1]],axis=0))
    
    mean_2050_ssp = np.mean([a_ssp[-1],b_ssp[-1],c_ssp[-1]])
    std_2050_ssp = np.mean(np.std([a_ssp[-1],b_ssp[-1],c_ssp[-1]],axis=0))
    
    diff_2050 = mean_2050_nzame - mean_2050_ssp
    diff_2050_err = np.sqrt(std_2050_nzame**2 + std_2050_ssp**2)
    
    print(f'{var} 2050 NZAME: {mean_2050_nzame:.3f} ± {std_2050_nzame:.3f} {units}')
    print(f'{var} 2050 SSP: {mean_2050_ssp:.3f} ± {std_2050_ssp:.3f} {units}')
    print(f'{var} difference 2050: {diff_2050:.3f} ± {diff_2050_err:.3f} {units}')

# prints output file with 2040-2050 data
import sys 
stdoutOrigin=sys.stdout 
sys.stdout = open("conc_data_nzame_final.txt", "w")
    
print('Conc data for 2040-2050:')
print()
# OH conc
mean_std_2050(oh_trop_mean_186/1e6,oh_trop_mean_146/1e6,oh_trop_mean_473/1e6,
              oh_trop_mean_bo797/1e6, oh_trop_mean_ca723/1e6,oh_trop_mean_cb039/1e6,'OH surface conc','x10^6 molec cm-3')
print()                         
mean_std_2040_2050(oh_trop_mean_186/1e6,oh_trop_mean_146/1e6,oh_trop_mean_473/1e6,
              oh_trop_mean_bo797/1e6, oh_trop_mean_ca723/1e6,oh_trop_mean_cb039/1e6,'OH surface conc','x10^6 molec cm-3')    
               
#pop weighted ozone
mean_std_2050(o3_pop_weighted_186,o3_pop_weighted_146,o3_pop_weighted_473,o3_pop_weighted_bo797,
              o3_pop_weighted_ca723,o3_pop_weighted_cb039,'O3 (pop)','ppb')
print()
mean_std_2040_2050(o3_pop_weighted_186,o3_pop_weighted_146,o3_pop_weighted_473,o3_pop_weighted_bo797,
              o3_pop_weighted_ca723,o3_pop_weighted_cb039,'O3 (pop)','ppb')
#area weighted ozone
print()
mean_std_2050(o3_lat_weighted_186,o3_lat_weighted_146,o3_lat_weighted_473,o3_lat_weighted_bo797,
              o3_lat_weighted_ca723,o3_lat_weighted_cb039,'O3 (area)','ppb')
print()
mean_std_2040_2050(o3_lat_weighted_186,o3_lat_weighted_146,o3_lat_weighted_473,o3_lat_weighted_bo797,
              o3_lat_weighted_ca723,o3_lat_weighted_cb039,'O3 (area)','ppb')
# CH4 burden
print()
mean_std_2040_2050(ch4_bur_bl593,ch4_bur_bz146,ch4_bur_bz473,ch4_bur_bo797,
              ch4_bur_ca723,ch4_bur_cb039,'CH4 burden','Tg')
print()
mean_std_2050(ch4_bur_bl593,ch4_bur_bz146,ch4_bur_bz473,ch4_bur_bo797,
              ch4_bur_ca723,ch4_bur_cb039,'CH4 burden','Tg')
# CH4 surface conc
print()
mean_std_2040_2050(ch4_lat_weighted_by186,ch4_lat_weighted_bz146,ch4_lat_weighted_bz473,ch4_lat_weighted_bo797,
              ch4_lat_weighted_ca723,ch4_lat_weighted_cb039,'CH4 (area)','ppb')
print()
mean_std_2050(ch4_lat_weighted_by186,ch4_lat_weighted_bz146,ch4_lat_weighted_bz473,ch4_lat_weighted_bo797,
              ch4_lat_weighted_ca723,ch4_lat_weighted_cb039,'CH4 (area)','ppb')
print()
mean_std_2050(mean_sur_temp_473,mean_sur_temp_146,mean_sur_temp_186,mean_sur_temp_bo797,
              mean_sur_temp_ca723,mean_sur_temp_cb039,'GMST','K')

sys.stdout.close()
sys.stdout=stdoutOrigin
print('end')