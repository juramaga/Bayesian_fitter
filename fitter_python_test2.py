# Import libraries

import sys
import numpy as np
import scipy.io
from astropy.io import ascii
from astropy.table import Table, Column
from astropy.io import fits
import matplotlib.pyplot as plt
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


# Read data of observed clusters (from Esteban Morales)
hdulist = fits.open('/n/regal/iacs/rafael/r08-objects.fits')
tbdata = hdulist[1].data

hdulist1 = fits.open('/n/regal/iacs/rafael/ukidss-sources.fits')
tbdata1 = hdulist1[1].data


# Read extinction law
ext_law = ascii.read("extinction_law.ascii")    # This is the extinction law (Fischera et al. 2005)
wavita = ext_law['col1']
opacita = ext_law['col2']


# Construct likelihood function

#========================================================================
#This function calculates the likelihood function for a given
#model and photometry dataset. And returns the lilekihood and chi^2.

#CALLING SEQUENCE
# > likelihood(obs_fluxes_resolved, obs_errors_resolved, observed_fluxes_unresolved, 
#                   observed_errors_unresolved, model_fluxes, weight))
# obs_fluxes_resolved: An array of dimension m x n where m is the number of sources and n is
# the number of resolved bands. This array contains the resolved fluxes.
#
# obs_errors_resolved: The associated errors to resolved observations. Same dimension.
#
# observed_fluxes_unresolved: An array of dimension p where p is the number of unresolved
# bands. It contains the unresolved fluxes.
#
# model_fluxes: the fluxes of the models to comapred with data. The array must have m rows
# and n+p columns.
#
# weight: The weight to be given to each datapoint (uniform, etc.)
#========================================================================

def likelihood(obs_fluxes_resolved, obs_errors_resolved, obs_fluxes_unresolved, 
               obs_errors_unresolved, model_fluxes, weight):
    
    # Get number of sources
    n_sources = len(obs_fluxes_resolved)
    # Get number of bands in which sources are resolved
    n_bands_resolved = len(obs_fluxes_resolved[0])
    n_data = len(model_fluxes[0])
        
    # Remove no-data points
    zeros_index_unres = np.array(~np.isnan(obs_unres))
    zeros_index_res = []
    for source in np.arange(n_sources):
        zeros_index_res.append(np.array(~np.isnan(obs_res[source])))

    
    chi_squared = 0
    
    # Calculate chi squared for resolved bands
    
    for source in np.arange(n_sources):
        
        # Get upper limits
        ind_upper_res = np.where(obs_errors_resolved[source] == -1)
        
        for i in np.arange(len(model_fluxes[source][ind_upper_res])):
            if (model_fluxes[source][ind_upper_res][i] > obs_fluxes_resolved[source][ind_upper_res][i]):
                chi_squared += float('Inf')
                continue            
        
        # Updated number of resolved datapoints
        n_data_res = len(obs_res[source][zeros_index_res[source]])
    
        # Unbiased flux in log space (since we are fitting in logarithmic space, see Robitaille et al. 2007)
        log_obs_flux = np.log10(obs_res[source][zeros_index_res[source]]) \
        -0.5*(1.0/np.log(10.))*(1.0/(obs_res[source][zeros_index_res[source]]**2.0)) \
        *obs_res_err[source][zeros_index_res[source]]*obs_res_err[source][zeros_index_res[source]]
        
        # Unbiased variances in log space (since we are fitting in logarithmic space, see Robitaille et al. 2007)
        log_obs_var2 = (((1.0/np.log(10.))*(1.0/obs_res[source][zeros_index_res[source]]))**2.0) \
        *obs_res_err[source][zeros_index_res[source]]*obs_res_err[source][zeros_index_res[source]] 
    
        # Model flux in log space
        log_mod_flux = np.log10(model_fluxes[source][0:n_bands_resolved][zeros_index_res[source]]) 
    
        # Reduced chi^2
        #print source, n_data_res, (1.0/n_data_res)*sum(((log_obs_flux - log_mod_flux)*(log_obs_flux - log_mod_flux)) \
                                            #/(2.0*weight*(log_obs_var2)))
        chi_squared += (1.0/n_data_res)*sum(((log_obs_flux - log_mod_flux)*(log_obs_flux - log_mod_flux)) \
                                            /(2.0*weight*(log_obs_var2)))
    
    # Calculate chi squared for unresolved bands
    # Get upper limits
    ind_upper_unres = np.where(obs_errors_unresolved == -1)
    
    # Likelihood is 0 if model flux at upper limit band is larger than the measured one.
    mod_flux_upper = 0.0
    for source in np.arange(n_sources): 
        mod_flux_upper += model_fluxes[source][n_bands_resolved:][zeros_index_unres][ind_upper_unres]
    if (mod_flux_upper > obs_unres_err[ind_upper_unres]):
        print 'WARNING: Upper limit in unresolved band'
        chi_squared += float('Inf')
    
    # Updated number of unresolved datapoints
    n_data_unres = len(obs_unres[zeros_index_unres])
    
    # Unbiased flux in log space (since we are fitting in logarithmic space, see Robitaille et al. 2007)
    log_obs_flux = np.log10(obs_unres[zeros_index_unres])-0.5*(1.0/np.log(10.)) \
    *(1.0/(obs_unres[zeros_index_unres]**2.0))*obs_unres_err[zeros_index_unres]*obs_unres_err[zeros_index_unres]
        
    # Unbiased variances in log space (since we are fitting in logarithmic space, see Robitaille et al. 2007)
    log_obs_var2 = (((1.0/np.log(10.))*(1.0/obs_unres[zeros_index_unres]))**2.0) \
    *obs_unres_err[zeros_index_unres]*obs_unres_err[zeros_index_unres] 
    
    # Model flux in log space
    mod_flux = np.zeros(n_data_unres)
    for source in np.arange(n_sources): 
        mod_flux += model_fluxes[source][n_bands_resolved:][zeros_index_unres]
    log_mod_flux = np.log10(mod_flux)
        
        
    # Reduced chi^2
    #print 'unres', n_data_unres, (1.0/n_data_unres)*sum(((log_obs_flux - log_mod_flux)*(log_obs_flux - log_mod_flux)) \
                                        #/(2.0*weight*(log_obs_var2)))
    chi_squared += (1.0/n_data_unres)*sum(((log_obs_flux - log_mod_flux)*(log_obs_flux - log_mod_flux)) \
                                        /(2.0*weight*(log_obs_var2)))
            
    # Likelihood probability (assuming Gaussian errors). Note that the chi squared value is the sum
    # of both resolved and unresolved bands
    if (chi_squared <= 745):
        likelihood = np.exp(-chi_squared)
    else:
        likelihood = np.exp(-745)
    
    likelihood_dict = {'chi2':chi_squared, 'likelihood':likelihood}
    
    #print 'chisq_tot', chi_squared
    #print 'prob_tot', likelihood
    
    return likelihood_dict


# Prior for the inclination angle
# This prior is based on a randon orientation of the disks. In that case, the 
# angle with respect to the line of sight is distributed as the cosine
# of the angle.

def prior_angle(index):
    prior = np.array([  6.50851226e+00,   2.97655246e+00,   1.66647382e+00,   9.58111152e-01,\
             5.37746335e-01,   2.82934901e-01,   1.33018844e-01,   5.08983303e-02, \
             1.26255358e-02,   7.12009421e-04])
    #prior = np.ones(10) # Uniform prior
    #prior = np.array([7.12009421e-04,5.08983303e-02,2.82934901e-01,9.58111152e-01,2.97655246e+00,\
    #                 6.50851226e+00,1.66647382e+00,5.37746335e-01,1.33018844e-01,1.26255358e-02])
    prior_norm = prior/sum(prior)
    return prior_norm[index]


# Prior for age
# Samples age in log space. Normal distribution.
def prior_age(log_age,mean,sigma):
    prior = scipy.stats.norm.pdf(log_age,loc=mean,scale=sigma)
    return prior


# Prior for Av
# Normal distribution in log space
def prior_Av(log_Av,mean,sigma):
    prior = scipy.stats.norm.pdf(log_Av,loc=mean,scale=sigma)
    return prior


# Define distance function (on the mass-age plane)
# This is useful for the random walk
def distance(age_old,mass_old,age_new,mass_new):
    return np.sqrt((np.log10(age_new)-np.log10(age_old))**2+(np.log10(mass_new)-np.log10(mass_old))**2)


# Construct proposal function for Metropolis
#========================================================================
#Given a current position in the parameter space (age, mass, extinction
#and inclination), this function provides a new candidate position. The
#function is symmetric.

#CALLING SEQUENCE
# > g(model_id,incl,Av,d,d_ref)
# model_id: The current model ID, that sets current mass and age.
# incl: Current inclination
# Av: Current Av
# d: current distance
# d: ref: same as current distance, used as reference to set step size.

# The function returns new position in the parameter space.
#========================================================================
def g(model_index,incl,Av,d,d_ref):

    age_old = idl_dict['age'][10*model_index+incl]
    mass_old = idl_dict['massc'][10*model_index+incl]
    ot = np.where(age3 == age_old)[0][0]
    
    dist =  1.0
    
    while (dist > 0.03):  # Here the radius of the function is set in the mass-age plane
        new_index = ot + np.random.random_integers(-200,200)   
        incl_new = incl + np.random.random_integers(-1,1)   # Radius in inclination is 1
        log_Av_new = np.log10(Av) + np.random.uniform(-0.03,0.03) # Radius in AV is 0.3 in log space
        log_d_new = np.log10(d) + np.random.uniform(-0.002,0.002) # Radius of distance is 0.02 kpc in log space
        
        if ((new_index > 0) and (new_index < 20070) and (incl_new >= 0) and (incl_new < 10) and (abs(np.log10(d_ref)-log_d_new) < 0.02)):
            index_new = it1[new_index]
            age_new = age3[new_index]
            mass_new = mass3[new_index]
            dist = distance(age_old,mass_old,age_new,mass_new)
        else:
            continue
        
    return index_new,incl_new,10**log_Av_new,10.0**log_d_new,d_ref


# Function to apply extinction correction
def ext_law(wavelengths_filters, Av, wavita, opacita):
    A_lambda = Av * (opacita/211.4)
    A_lambda_waves = np.interp(wavelengths_filters,wavita,A_lambda)
    corr_factor = 10.0**(-0.4*A_lambda_waves)   # This is the array of correction factors to be applied to each band
    return corr_factor


# Main MCMC function. Here we define a function to perform the Metropolis algorithm. The function returns
# the trace, the set of likelihoods and chi^2 values, as well as the new parameters.

def mh(Ns, x0, filter_names, obs_fluxes_resolved, obs_errors_resolved, obs_fluxes_unresolved, obs_errors_unresolved):
    
    filter_names_all = np.array(['model_str_bu','model_str_bb','model_str_bv','model_str_2j','model_str_2h','model_str_2k','model_str_i1',
                'model_str_i2','model_str_i3','model_str_i4','model_str_m1','model_str_pacs1','model_str_pacs2',
                'model_str_pacs3','model_str_spir1','model_str_spir2'])
    filter_wavelengths_all = np.array([0.36,0.44,0.55,1.235,1.651,2.159,3.550,4.493,5.731,7.872,23.68,71.889,102.457,166.186,252.051,362.088])
    filters = [filter_names_all,filter_wavelengths_all]
    
    # Assign wavelengths to desired filters
    filter_wavelengths = []
    for name in filter_names:
        filter_wavelengths.append(filter_wavelengths_all[filter_names_all == name][0])
        
    n_sources = len(obs_fluxes_resolved)              # Total number of sources
    n_bands_resolved = len(obs_fluxes_resolved[0])    # Number of resolved bands
    n_bands_unresolved = len(obs_fluxes_unresolved)   # Number of unresolved bands
    
    xsample=np.zeros((5,n_sources,Ns))              # Array for the samples. 5 parameters for each source, n_sources in total
    likelihoods = np.zeros(Ns)
    chi2s = np.zeros(Ns)
    x0 = np.transpose(x0)
    xsample[:,:,0]=x0
    acceptcnt=0
    cnt=0 # Initialize counts

    list_filters = filter_names

    # Assign fluxes to the model SEDs
    model_fluxes_all = []   # Array that will be filled with the model fluxes for all sources at all wavelengths
    prior_incl_0 = 1.0
    prior_t_0 = 1.0
    prior_obs_0 = 1.0
    for i in np.arange(n_sources):
        model_fluxes_prop = []
        for filter in list_filters:
            model_fluxes_prop.append(idl_dict1[filter]['fluxes'][0][49][10*xsample[0,i,0]+xsample[1,i,0]])

        corr_ext = ext_law(filter_wavelengths,xsample[2,i,0],wavita,opacita)   # Extinction correction
        model_fluxes_prop = corr_ext*model_fluxes_prop/(xsample[3,i,0]**2)
        model_fluxes_all.append(model_fluxes_prop)
        prior_incl_0 *= prior_angle(xsample[1,i,0].astype(int)) # Prior on inclination
        age_0 = idl_dict['age'][10.0*xsample[0,i,0].astype(int)+xsample[1,i,0].astype(int)]
        prior_t_0 *= prior_age(np.log10(age_0),5.5,1.0)
        #prior_obs_0 *= prior_Av(np.log10(xsample[2,i,0]),float(sys.argv[3]),float(sys.argv[4]))
        prior_obs_0 *= prior_Av(np.log10(xsample[2,i,0]),1.3,0.8)

    # Initial log likelihood (i.e. log likelihood function evaluated at the initial guess values) times the priors.
    lprop_old = np.log(prior_incl_0*prior_t_0*prior_obs_0*likelihood(obs_fluxes_resolved,obs_errors_resolved,obs_fluxes_unresolved,obs_errors_unresolved,model_fluxes_all,1.0)['likelihood']) 
    
    # Initial chi squared value
    chi2_old = likelihood(obs_fluxes_resolved,obs_errors_resolved,obs_fluxes_unresolved,obs_errors_unresolved,model_fluxes_all,1.0)['chi2']-(np.log(prior_t_0)+np.log(prior_incl_0)+np.log(prior_obs_0))
    # We start filling the arrays for likelihood and chi square distributions
    likelihoods[0] = lprop_old
    chi2s[0] = chi2_old
    
    # Main loop
    while cnt+1 < Ns:
        if (cnt%1000 == 0): print cnt, 'out of', Ns
        # Get new proposed parameters for each source
        model_fluxes_star_all = []
        xstar_all = []
        prior_incl = 1.0
        prior_t = 1.0
        prior_obs = 1.0
        for i in np.arange(n_sources):
            #print xsample[0,i,cnt].astype(int),xsample[1,i,cnt].astype(int),xsample[2,i,cnt],xsample[3,i,cnt],xsample[4,i,cnt]
            xstar = g(xsample[0,i,cnt].astype(int),xsample[1,i,cnt].astype(int),xsample[2,i,cnt],xsample[3,i,cnt],xsample[4,i,cnt]) # The next step in the chain is sampled randomly from a Gaussian
            #print i,xstar
            ages_star = idl_dict['age'][10.0*xstar[0].astype(int)+xstar[1].astype(int)]  # Corresponding age for proposed step
            #print i,ages_star
            corr_ext = ext_law(filter_wavelengths,xstar[2],wavita,opacita)
            
            # Assign fluxes for proposed step
            model_fluxes_star = []
            for filter in list_filters:
                #nombres = idl_dict1[filter]['name'][0]
                #angulos = idl_dict1[filter]['angle'][0]

                model_fluxes_star.append(idl_dict1[filter]['fluxes'][0][49][10*xstar[0]+xstar[1]])                
                #model_fluxes_star.append(idl_dict1[filter]['fluxes'][0][49][(nombres == xstar[0]) & (angulos == idl_dict1[filter]['angle'][0][xstar[1]])][0])
                
            model_fluxes_star = corr_ext*model_fluxes_star/(xstar[3]**2)
            model_fluxes_star_all.append(model_fluxes_star)
            xstar_all.append(xstar)
            prior_incl *= prior_angle(xstar[1].astype(int))  # Prior on inclination
            prior_t *= prior_age(np.log10(ages_star),5.5,1.0)
            #prior_obs *= prior_Av(np.log10(xstar[2]),float(sys.argv[3]),float(sys.argv[4]))
            prior_obs *= prior_Av(np.log10(xstar[2]),1.3,0.8)

        # The following calculates the posterior(likelihood times priors)
        #print obs_fluxes_resolved,obs_fluxes_unresolved,model_fluxes_star_all
        if (likelihood(obs_fluxes_resolved,obs_errors_resolved,obs_fluxes_unresolved,
                                    obs_errors_unresolved,model_fluxes_star_all,1.0)['likelihood'] < np.exp(-745.)):
            lprop_new = -745.
        else:
            lprop_new = np.log(prior_incl*prior_t*prior_obs*likelihood(obs_fluxes_resolved,obs_errors_resolved,obs_fluxes_unresolved,
                                    obs_errors_unresolved,model_fluxes_star_all,1.0)['likelihood'])
        chi2_new = likelihood(obs_fluxes_resolved,obs_errors_resolved,obs_fluxes_unresolved,
                                    obs_errors_unresolved,model_fluxes_star_all,1.0)['chi2']-(np.log(prior_t)+np.log(prior_incl)+np.log(prior_obs))
    
        if (lprop_new - lprop_old) > np.log(np.random.uniform(0,1,1)):  # Acceptance ratio. Notice that here only the ratio of the evaluated functions matters.
            x0=xstar_all
            x0 = np.transpose(x0)
            lprop_old=lprop_new
            chi2_old=chi2_new
            acceptcnt += 1
        
        cnt += 1
        xsample[:,:,cnt]=x0
        likelihoods[cnt]=lprop_old
        chi2s[cnt]=chi2_old
        
    print 'The acceptance rate is:', acceptcnt/float(Ns)

    # Save the chain
    with open('/n/regal/iacs/rafael/output/standard_prior_small_step/output_chain_'+str(id)+'.pickle', 'w') as f:
        pickle.dump([xsample,chi2s], f)
        
    # Burn-in 20%
    xout=xsample[:,:,Ns/5:] 
    likeli_out=likelihoods[Ns/5:]
    chi2out=chi2s[Ns/5:]
    
    # Get best fit (the one that minimizes chi^2)
    alo = chi2out == min(chi2out)
    ola = likeli_out == max(likeli_out)
    
    new_params = []
    for i in np.arange(n_sources):
        np_source = [xout[0][i][ola][0].astype(int),xout[1][i][ola][0].astype(int),xout[2][i][ola][0]]
        new_params.append(np_source)
        
    return [xout,likeli_out,chi2out,new_params]


# Fit for one cluster

id = sys.argv[1]  # ID of the source
id = int(id)

print np.shape(np.array(np.where(tbdata['recno'] == id)))
id_ind = np.array(np.where(tbdata['recno'] == id))[0][0]

mags_spitzer = np.array([tbdata[id_ind]['__3_6_G'],tbdata[id_ind]['__4_5_'],tbdata[id_ind]['__5_8_G'],tbdata[id_ind]['__8_0_'],tbdata[id_ind]['__24_']])

zero_points = np.array([280.9, 179.7,115.0,64.13,7.14])
flux_spitzer =  zero_points*10.0**(-mags_spitzer/2.5)

ind_id1 = np.where(tbdata1['r08_id'] == id)

mags_ukidss = np.array([tbdata1[ind_id1]['mag_j'],tbdata1[ind_id1]['mag_h'],tbdata1[ind_id1]['mag_k']])
err_ukidss = np.array([tbdata1[ind_id1]['emag_j'],tbdata1[ind_id1]['emag_h'],tbdata1[ind_id1]['emag_k']])

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


filter_names = ['model_str_2j','model_str_2h','model_str_2k','model_str_i1',
                'model_str_i2','model_str_i3','model_str_i4','model_str_m1']
filter_wavelengths = np.array([1.235,1.651,2.159,3.550,4.493,5.731,7.872,23.68])


# Set errors and initial values for paramters
# Fluxes of real cluster (from Morales) (in mJy)
obs_res = 1000.0*flux_ukidss
obs_unres = 1000.0*flux_spitzer

obs_res_err = 0.15*obs_res #1000.0*np.transpose(error_ukidss)
obs_unres_err = 0.15*obs_unres
    
obs_res_err[upper] = -1  # Set measurements that are upper limits
    
obs_res = np.transpose(obs_res)
obs_res_err = np.transpose(obs_res_err)

print obs_res
print obs_unres

# Set distance to source (in kpc)
d_src = float(sys.argv[2])

# Initial parameters
all_file = open('/n/regal/iacs/rafael/latest/newer/GA_results_'+str(id)+'_new.pickle','rb')
optimal_sol,data = pickle.load(all_file)
print optimal_sol,optimal_sol[0]
init = []
for i in np.arange(len(optimal_sol)):
    init.append(np.append(optimal_sol[i],d_src))
    #for j in np.arange(len(optimal_sol[i])):
    #    init.append(optimal_sol[i][j])
    #init.append(d_src)

print 'INITIAL = ',init

# Number of iterations
N = 10000000

# Run the MCMC
start_time = time.clock()
print init
xout,likeli_out,chi2out,new_params = mh(N,init,filter_names,obs_res, obs_res_err, obs_unres, obs_unres_err)
end_time = time.clock()

print "Total duration: ",(end_time-start_time)/60., "minutes"
