import sys
import numpy as np
import scipy.io
from astropy.io import ascii
from astropy.table import Table, Column
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import pickle
import time
from scipy.stats import beta

# First read the convolved photometry
dir_conv = '/n/regal/iacs/rafael/' # Directory where convolved fluxes are located
idl_dict1=scipy.io.readsav(dir_conv+'all_conv.save')
del idl_dict1['model_str']
del idl_dict1['dir_conv']
nombres = np.arange(20071)
#angulos = idl_dict1[filter]['angle'][0][0:10]

# Read model parameters and order them
idl_dict = scipy.io.readsav('/n/regal/iacs/rafael/model_params.save')
age1 = idl_dict['age'][idl_dict['incli']==87.1300]
mass1 = idl_dict['massc'][idl_dict['incli']==87.1300]
ltot1 = idl_dict['ltot'][idl_dict['incli']==87.1300]
#id1 = idl_dict['id'][idl_dict['incli']==87.1300]
it1 = np.argsort(age1)
it2 = np.argsort(mass1)
age2 = np.sort(age1)
mass2 = np.sort(mass1)
ltot2 = np.sort(ltot1)
age3 = age1[it1]
mass3 = mass1[it1]
ltot3 = ltot1[it1]
id3 = nombres[it1]


def ext_law(wavelengths_filters, Av, wavita, opacita):
    A_lambda = Av * (opacita/211.4)
    A_lambda_waves = np.interp(wavelengths_filters,wavita,A_lambda)
    corr_factor = 10.0**(-0.4*A_lambda_waves)   # This is the array of correction factors to be applied to each band
    return corr_factor

# Read extinction law
ext_law = ascii.read("/n/regal/iacs/rafael/extinction_law.ascii")    # This is the extinction law (Fischera et al. 2005)
wavita = ext_law['col1']
opacita = ext_law['col2']

# Read observed clusters (from Esteban Morales)
hdulist = fits.open('/n/regal/iacs/rafael/r08-objects.fits')
tbdata = hdulist[1].data

id = sys.argv[1]  # ID of the source
id = int(id)
print id
print np.shape(np.array(np.where(tbdata['recno'] == id)))
id_ind = np.array(np.where(tbdata['recno'] == id))[0][0]

mags_spitzer = np.array([tbdata[id_ind]['__3_6_G'],tbdata[id_ind]['__4_5_'],tbdata[id_ind]['__5_8_G'],tbdata[id_ind]['__8_0_'],tbdata[id_ind]['__24_']])
zero_points = np.array([280.9, 179.7,115.0,64.13,7.14])
flux_spitzer =  zero_points*10.0**(-mags_spitzer/2.5)
print flux_spitzer

hdulist = fits.open('/n/regal/iacs/rafael/ukidss-sources.fits')
tbdata1 = hdulist[1].data

ind_id1 = np.where(tbdata1['r08_id'] == id)

mags_ukidss = np.array([hdulist[1].data[ind_id1]['mag_j'],hdulist[1].data[ind_id1]['mag_h'],hdulist[1].data[ind_id1]['mag_k']])
err_ukidss = np.array([hdulist[1].data[ind_id1]['emag_j'],hdulist[1].data[ind_id1]['emag_h'],hdulist[1].data[ind_id1]['emag_k']])

q_flags = np.empty_like(mags_ukidss)

q_flags[0,:] = tbdata1['qflag_j'][ind_id1]
q_flags[1,:] = tbdata1['qflag_h'][ind_id1]
q_flags[2,:] = tbdata1['qflag_k'][ind_id1]

up_flags = np.empty_like(mags_ukidss)
up_flags[0,:] = tbdata1['up_j'][ind_id1]
up_flags[1,:] = tbdata1['up_h'][ind_id1]
up_flags[2,:] = tbdata1['up_k'][ind_id1]

upper = np.where(up_flags == 1)
bad = np.where(q_flags != 0)

flux_ukidss = np.empty_like(mags_ukidss)
error_ukidss = np.empty_like(err_ukidss)
zero_points = np.array([1594.,1024.,666.7])
flux_ukidss[0,:] = zero_points[0]*10**(-mags_ukidss[0,:]/2.5)
flux_ukidss[1,:] = zero_points[1]*10**(-mags_ukidss[1,:]/2.5)
flux_ukidss[2,:] = zero_points[2]*10**(-mags_ukidss[2,:]/2.5)

flux_ukidss[bad] = np.nan  # Get rid of bad measurements

error_ukidss[0,:] = zero_points[0]*10**(-(mags_ukidss[0,:]-err_ukidss[0,:])/2.5)
error_ukidss[1,:] = zero_points[1]*10**(-(mags_ukidss[1,:]-err_ukidss[1,:])/2.5)
error_ukidss[2,:] = zero_points[2]*10**(-(mags_ukidss[2,:]-err_ukidss[2,:])/2.5)

print flux_ukidss
print flux_ukidss[0,0]

n_sources = len(flux_ukidss.T)
print 'n_sources = ',n_sources

filter_names = ['model_str_2j','model_str_2h','model_str_2k','model_str_i1',
                'model_str_i2','model_str_i3','model_str_i4','model_str_m1']
filter_wavelengths = np.array([1.235,1.651,2.159,3.550,4.493,5.731,7.872,23.68])

# Fluxes of real cluster (from Morales) (in mJy)
obs_res = 1000.0*flux_ukidss
obs_unres = 1000.0*flux_spitzer

obs_unres_err = 0.15*obs_unres
obs_res_err = 0.15*obs_res #1000.0*transpose(error_ukidss)

obs_res_err[upper] = -1  # Set measurements that are upper limits

obs_res = np.transpose(obs_res)
obs_res_err = np.transpose(obs_res_err)

pkl_file = open('/n/regal/iacs/rafael/output/standard_prior_small_step/output_chain_'+str(id)+'.pickle', 'rb')
xsample, chi2s = pickle.load(pkl_file)
N = 10000000
 
# Burn-in 20%
xout=xsample[:,:,N/5:] 
#likeli_out=likelihoods[N/5:]
chi2out=chi2s[N/5:]

print min(chi2out)
    
# Get best fit (the one that minimizes chi^2)
alo = np.where(chi2out == min(chi2out))
#ola = np.where(likeli_out == max(likeli_out))

new_params = []
for i in np.arange(n_sources):
    np_source = [xout[0][i][alo][0].astype(int),xout[1][i][alo][0].astype(int),xout[2][i][alo][0],xout[3][i][alo][0]]
    new_params.append(np_source)
    
print new_params    
with open('best_params_'+str(id)+'.pickle', 'w') as f:
    pickle.dump(new_params, f)

# PLOT HISTOGRAMS

ind_min = np.where(chi2out == min(chi2out))

print np.shape(idl_dict['age'][np.where(idl_dict['id']==id)])


plt.figure()
av_peak = []
for j in np.arange(n_sources):
    c = plt.hist(xout[2,j,:])
    av_peak.append(c[1][np.argmax(c[0])])
plt.close

# We asign parameters for the elements of the chain
plt.figure()
#fig, ax = plt.subplots(n_sources,2,figsize=(8,4*n_sources))
fig, ax = plt.subplots(figsize=(8,4*n_sources))
fig.text(0.05,0.5,'Probability',va='center',rotation='vertical',size=15)

ages_best = []
masses_best = []
ages_peak = []
masses_peak = []

for j in np.arange(n_sources):
    ages_out = []   # Array for ages
    masses_out = [] # Array for masses
    for id1 in xout[0,j]:
        ages_out.append(age1[id1])
        masses_out.append(mass1[id1])
        
    ages_best.append(age3[it1==new_params[j][0]])
    masses_best.append(mass3[it1==new_params[j][0]])
        
    ax1 = plt.subplot(n_sources,2,2*j+1)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax1.set_rasterization_zorder(1)
    a = plt.hist(np.log10(ages_out),bins=5,normed=True,alpha=0.5,zorder=1.0)
    ages_peak.append(a[1][np.argmax(a[0])])
    ##vlines(np.log10(idl_dict['age'][idl_dict['id']==ids[j]][0]),0,np.rint(1.0+max(a[0])),linewidth=2,color='red',linestyle='--',label='True value')
    plt.vlines(np.log10(ages_best[j]),0,0.3*+max(a[0])+max(a[0]),linewidth=2,color='red',linestyle='--',label='Best fit')
    plt.xlabel('log Age [yr]',size=15)
    plt.axis('tight')
    #plt.ylabel('Probability',size=15)
    ax2 = plt.subplot(n_sources,2,2*j+2)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax2.set_rasterization_zorder(1)
    b = plt.hist(np.log10(masses_out),bins=5,normed=True,alpha=0.5,zorder=1.0)
    masses_peak.append(b[1][np.argmax(b[0])])
    plt.vlines(np.log10(masses_best[j]),0,0.3*max(b[0])+max(b[0]),linewidth=2,color='red',linestyle='--',label='Best fit')
    plt.xlabel('log Mass [M$_{\odot}$]',size=15)
    #plt.ylabel('Probability',size=15)
    plt.axis('tight')


plt.legend(fontsize=12,loc=2)

plt.savefig('hist_'+str(id)+'.ps')
plt.savefig('hist_'+str(id)+'.pdf')
plt.savefig('hist_'+str(id)+'.jpg')

plt.close

with open('peak_params_'+str(id)+'.pickle', 'w') as f:
    pickle.dump([ages_peak, masses_peak, av_peak], f)

def ext_law(wavelengths_filters, Av, wavita, opacita):
    A_lambda = Av * (opacita/211.4)
    A_lambda_waves = np.interp(wavelengths_filters,wavita,A_lambda)
    corr_factor = 10.0**(-0.4*A_lambda_waves)   # This is the array of correction factors to be applied to each band
    return corr_factor

# PLOTS SEDs
plt.figure()

from astropy.constants import c
import astropy.units as u

sed_total = 0.0

for i in np.arange(len(new_params)):
    
    col = ['blue','green','red','purple','orange','pink']
    plt.scatter(filter_wavelengths[0:3],obs_res[i],color='black',marker='o',s=30,zorder=2)
    upperlimits = np.array(obs_res_err[i] == -1)
    obs_res_err[i][upperlimits] = 1.5*obs_res[i][upperlimits]
    plt.errorbar(filter_wavelengths[0:3],obs_res[i], yerr=0.15*obs_res[i], uplims=upperlimits, capsize=3,linestyle='',color='black')
        
    plt.scatter(filter_wavelengths[3:],obs_unres,color='black',marker='o',s=30,zorder=2)
    plt.errorbar(filter_wavelengths[3:],obs_unres, yerr=0.15*obs_unres, capsize=3,linestyle='',color='black')
    
    
    mod_id = int(idl_dict1['model_str_i4'][0][0][10*new_params[i][0]+new_params[i][1]])
    hdulisto = fits.open('/n/regal/iacs/rafael/seds/30'+\
    str(mod_id)[2:5]+'/'+str(mod_id)+'_'+str(1+new_params[i][1])+'_sed.fits.gz')

    wave = hdulisto[1].data['WAVELENGTH'][::-1]
    flux = hdulisto[3].data['TOTAL_FLUX'][49][::-1]
    freq = u.micron.to(u.Hz, wave, equivalencies=u.spectral())

    corri = ext_law(wave,new_params[i][2],wavita, opacita)
    sed_fluxes = 1E23*corri*flux*(1.0/(new_params[i][3]**2))/freq
    
    sed_total += sed_fluxes

    plt.plot(wave,1000*sed_fluxes,color=col[i])
    

plt.plot(wave,1000*sed_total,color='black')
limit_up = 3.0*max(1000*sed_total)
plt.text(0.7,limit_up/3.0,id,size=15)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.5,500)
plt.ylim(7E-3,limit_up)
plt.xlabel('Wavelength $\lambda$ [micron]',size=12)
plt.ylabel('Flux density [mJy]',size=12)



#flux_best_integrated = 0.0
#for j in np.arange(n_sources):
#    model_fluxes_out = []
#    for filter in filter_names:
        
#        model_fluxes_out.append(idl_dict1[filter]['fluxes'][0][49][10*new_params[j][0]+new_params[j][1]])
        
#    # Extinction correction
#    corri = ext_law(filter_wavelengths,new_params[j][2],wavita, opacita)
#    model_fluxes_out = corri*model_fluxes_out*(1.0/(new_params[j][3]**2))
    
    
#    # Plot results 
#    print 'shape = ', np.shape(obs_res)
#    col = ['blue','green','red','pink','purple']
#    plt.scatter(filter_wavelengths[0:3],obs_res[j],color='black',marker='o',s=30,zorder=2)
#    #plt.errorbar(filter_wavelengths[0:3],obs_res[j],yerr=obs_res_err[j])
#    plt.scatter(filter_wavelengths[3:],obs_unres,color='black',marker='o',s=30,zorder=2)
#    #plt.errorbar(filter_wavelengths[3:],obs_unres,yerr=obs_unres_err)
    
#    plt.plot(filter_wavelengths,model_fluxes_out,color=col[j],zorder=3,linewidth=2)
#    print model_fluxes_out[4]/10.2716818573
#    flux_best_integrated += model_fluxes_out

#obs_tot = 0
#for i in np.arange(len(obs_res)):
#    obs_tot += obs_res[i]
    
#plt.scatter(filter_wavelengths[0:3],obs_tot,color='black',marker='^',s=30,zorder=2) 
#plt.errorbar(filter_wavelengths[0:3],obs_tot, yerr=0.15*(obs_res[0]+obs_res[1]))
#plt.plot(filter_wavelengths,flux_best_integrated,color='orange',zorder=3,linewidth=2)
#plt.scatter(filter_wavelengths,flux_best_integrated,color='orange')

#plt.xscale('log')
#plt.yscale('log')

#plt.xlabel('Wavelength $\lambda$ [micron]',size=20)
#plt.ylabel('Flux density [mJy]',size=20)

#plt.xlim(0.6,50)
##plt.ylim(7E-3,3E2)

plt.savefig('sed_plot_'+str(id)+'.ps')
plt.savefig('sed_plot_'+str(id)+'.pdf')
plt.savefig('sed_plot_'+str(id)+'.jpg')


plt.close
