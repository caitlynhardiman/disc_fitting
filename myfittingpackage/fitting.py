import numpy as np
import matplotlib.pyplot as plt
import pymcfost as mcfost
from pymcfost.utils import FWHM_to_sigma
import casa_cube as casa
from scipy.optimize import shgo
import os
import time
import subprocess
#from multiprocess.pool import Pool
import multiprocessing
import importlib
from myfittingpackage.bilby_likelihood import mcfost_likelihood
from myfittingpackage.generate_model import make_model
from scipy import stats
from astropy.convolution import Gaussian2DKernel, convolve_fft, convolve



class Disc:

    def __init__(self,
                 datacube: None,
                 uncertainty: float = None,
                 **kwargs):


        if isinstance(datacube, str):
            print("Reading cube ...")
            cube = casa.Cube(datacube)

        if cube.nx > 256:
            rescale = 256/cube.nx
            print('Need to resize cube...')
            cube = casa.Cube(datacube, zoom=rescale)


        self.cube = cube
        self.beam_area = self.cube.bmin * self.cube.bmaj * np.pi / (4.0 * np.log(2.0))
        self.pix_area = self.cube.pixelscale**2
        self.offset = v_offset
        self.uncertainty = uncertainty * self.pix_area/self.beam_area

        # finding 9 channels for comparison - assuming cube is centered on systemic velocity

        flux_vals = np.array(self.cube.image)
        flux_vals[np.isnan(flux_vals)] = 0
        flux_vals = flux_vals*self.pix_area/self.beam_area
        lp_fluxes = np.sum(flux_vals[:,:,:], axis=(1, 2))

        systemic_index = int(self.cube.nv/2)
        left_peak = np.argmax(lp_fluxes[systemic_index:])
        right_peak = np.argmax(lp_fluxes[:systemic_index])+systemic_index
        spacing = np.max(np.abs(left_peak-systemic_index), np.abs(right_peak-systemic_index))
        chans = []
        for i in range(-4, 5):
            chans.append(systemic_index+i*spacing)
        vel_chans = []
        fluxes = []
        for chan in chans:
            vel_chans.append(self.cube.velocity[chan])
            fluxes.append(self.cube.image[chan])

        self.channels = np.array(vel_chans)
        self.fluxes = np.array(fluxes)


    def _convolve_model(self, mcfost_model):
        model = []
        for i in range(len(mcfost_model.lines)):
            mcfost_model.plot_map(iv=i, bmaj=self.cube.bmaj, bmin=self.cube.bmin, bpa=self.cube.bpa)
            model.append(mcfost_model.last_im)
            plt.close()
        return model

    def _init_tracking(self, filename, ranges):
        # don't need self??
        results_file = open(filename, "a")
        num_params = len(ranges.params)
        size = 20*num_params + 40
        dash = '-' * size + '\n'
        first_columns = ['Iteration Number', 'Reduced Chi Squared']
        results_file.write(dash)
        results_file.write(dash)
        results_file.write('{:<20s}{:<25s}'.format(first_columns[0], first_columns[1]))
        for param in ranges.params:
            results_file.write('{:<20s}'.format(param))
        results_file.write('\n')
        results_file.write(dash)
        results_file.write(dash)
        results_file.close()

    def record_iterations(filename, parameters, counter, chi_value):
        results_file = open(filename, "a")
        results_file.write('{:<20s}{:<25s}'.format(str(counter[0]), str(round(chi_value, 5))))
        for param_value in parameters:
            results_file.write('{:<20s}'.format(str(param_value)))
        results_file.write('\n')
        results_file.close()

    def _best_result(self, filename, ranges, parameters, chi_value):
        results_file = open(filename, "a")
        num_params = len(ranges.params)
        size = 20*num_params + 40
        dash = '-' * size + '\n'
        results_file.write(dash)
        results_file.write('{:<20s}{:<25s}'.format('BEST FIT', str(round(chi_value, 5))))
        for param_value in ranges.params:
            results_file.write('{:<20s}'.format(str(param_value)))
        results_file.write('\n')
        results_file.write(dash)
        results_file.close()

    def write_run_mcfost(inclination, stellar_mass, scale_height, r_c, r_in, flaring_exp, PA, dust_param, v_turb):
        # Rewrite mcfost para file
        updating = mcfost.Params('dmtau.para')
        updating.map.RT_imin = inclination+180
        updating.map.RT_imax = inclination+180
        updating.stars[0].M = stellar_mass
        updating.zones[0].h0 = scale_height
        updating.zones[0].Rc = r_c
        updating.zones[0].Rin = r_in
        updating.zones[0].flaring_exp = flaring_exp
        updating.map.PA = PA
        updating.simu.viscosity = dust_param
        updating.mol.v_turb = v_turb
        updating.writeto('dmtau.para')    # Run mcfost
        mcfost.run('dmtau.para', options="-mol -casa -photodissociation", delete_previous=True)
        # Obtain flux from line profile
        model = mcfost.Line('data_CO/')
        return model

    def red_chi_squared_cube(self, theta, x, y, yerr, counter, channel, filename, convolve):
        inclination, stellar_mass, scale_height, r_c, r_in, flaring_exp, PA, dust_param = theta
        mcfost_model = write_run_mcfost(self, inclination, stellar_mass, scale_height, r_c, r_in, flaring_exp, PA, dust_param, counter)
        sigma2 = yerr**2
        if convolve:
            model = self._convolve(mcfost_model)
        else:
            model = mcfost_model
        red_chisq = np.sum((np.array(y)-np.array(model))**2/yerr**2)/(len(model)*len(model[0])*len(model[0][0])-1)
        record_iterations(filename, theta, counter, red_chisq)

    def shgo_cube_fit(self, ranges, convolve: bool = True):

        # Initialise results file
        filename = 'shgo_fit_to_cube.txt'
        self._init_tracking(filename, ranges)

        # Run shgo optimization - still a bit hard coded here for DM Tau cube
        counter = [0, 50]
        channel = 4
        chi_args = lambda theta, x=self.channels, y=self.fluxes, yerr=self.uncertainty, counter=counter, channel=channel, filename=filename, convolve=convolve: red_chi_squared_cube(self, theta, x, y, yerr, counter, channel, filename, convolve)
        bounds = ranges.bounds
        soln = shgo(chi_args, bounds, iters=3, n=100, sampling_method='simplicial') # not enough iterations here - need to up this when you figure out how many are required
        best_fit = soln.x
        best_chi = soln.fun

        # Conclude results file
        self._best_result(filename, ranges, best_fit, best_chi)



    def bilby_mcmc_fit(self, ranges, method, convolve: bool = True):

        import bilby
        from bilby_likelihood import standard_likelihood
        import multiprocess

        # Labels for bilby directories
        label = method
        outdir = 'bilby'+method

        # Using only uniform priors for now
        parameters = ['inc', 'stellar_mass', 'scale_height', 'r_c', 'r_in', 'psi', 'PA', 'dust_alpha', 'vturb', 'dust_mass', 'gasdust_ratio']
        priors = dict()

        for param in parameters:
            if ranges.param is not None:
                if param == 'dust_alpha':
                    priors[param] = bilby.core.prior.LogUniform(ranges.param)
                # gaussian
                if param == 'inc' or param == 'PA':
                    priors[param] = bilby.core.prior.Gaussian(ranges.param)
                # truncated gaussian
                if param == 'psi':
                    priors[param] = bilby.core.prior.TruncatedGaussian(ranges.param)
                else:
                    priors[param] = bilby.core.prior.Uniform(ranges.param)

        # need to differentiate between methods here
        if method=='cube':
            likelihood = myLikelihood(self.channels, self.fluxes, self.uncertainty, cube_flux)
        elif method=='line':
            lp_fluxes = np.sum(self.fluxes[:,:,:], axis=(1, 2))
            uncertainty = np.empty(len(lp_fluxes)); uncertainty.fill(2.0)
            likelihood = myLikelihood(self.channels, lp_fluxes, uncertainty, line_profile_flux)
        else:
            # error - needs to be cube or line
            return 0

        # And run sampler
        if __name__ == "__main__":
            result = bilby.run_sampler(
            likelihood=likelihood, priors=priors, sampler='emcee',
            nwalkers = 64,
            nsteps=20,
            npool=4)

        result.plot_corner()

##################################################################################


class image_plane_fit:

    def __init__(self,
                 datacube: None,
                 model_params=None,
                 distance=None,
                 uncertainty: float = None,
                 npix = 256,
                 vismode = False,
                 vel_range = None,
                 casa_sim = False,
                 **kwargs):
        
        if distance is None:
            print('Need to provide a distance!')
            return
        else:
            self.distance = distance

        if isinstance(datacube, str):
            print("Reading data cube ...")
            cube = casa.Cube(datacube)
        else:
            print('Need a valid cube name!')
            return
        
        if uncertainty is None:
            cube.get_std()
            uncertainty = cube.std

        if cube.nx > npix:
            rescale = npix/cube.nx
            print('Need to resize cube...')
            cube = casa.Cube(datacube, zoom=rescale)

        self.bmin = cube.bmin
        self.bmaj = cube.bmaj
        self.bpa = cube.bpa
        self.pixelscale = cube.pixelscale
        self.beam_area = cube.bmin * cube.bmaj * np.pi / (4.0 * np.log(2.0))
        self.pix_area = cube.pixelscale**2
        self.uncertainty = uncertainty * self.pix_area/self.beam_area
        systemic_index = int(cube.nv/2)
        self.vsyst = cube.velocity[systemic_index]

        flux_vals = np.array(cube.image)
        flux_vals[np.isnan(flux_vals)] = 0
        flux_vals = flux_vals*self.pix_area/self.beam_area

        # for visibilities comparison - need to match channels 
        if vismode:
            if vel_range is not None:
                if casa_sim:
                    offset = 6.008
                    cube.velocity = cube.velocity/1e3
                    cube.velocity+= offset
                lower = vel_range[0]/1e3
                upper = vel_range[1]/1e3
                i_l = np.argmin(np.abs(cube.velocity-lower))
                i_h = np.argmin(np.abs(cube.velocity-upper))
                if i_l > i_h:
                    new_i_l = i_h
                    new_i_h = i_l
                    i_l = new_i_l
                    i_h = new_i_h
                print(i_l, i_h)
                self.channels = cube.velocity[i_l:i_h+1]
                self.fluxes = flux_vals[i_l:i_h+1]
            else:
                print('Provide velocity range for comparing to visibilities')
                return
        else:
            # finding 9 channels for comparison - assuming cube is centered on systemic velocity
            lp_fluxes = np.sum(flux_vals[:,:,:], axis=(1, 2))

            left_peak = np.argmax(lp_fluxes[:systemic_index])
            right_peak = np.argmax(lp_fluxes[systemic_index:])+systemic_index
            spacing = np.max([np.abs(left_peak-systemic_index), np.abs(right_peak-systemic_index)])
            chans = []
            for i in range(-4, 5):
                chans.append(systemic_index+i*spacing)
            vel_chans = []
            fluxes = []
            for chan in chans:
                vel_chans.append(cube.velocity[chan])
                fluxes.append(cube.image[chan])

            self.channels = np.array(vel_chans)
            self.fluxes = np.array(fluxes)
            self.data_line_profile = np.sum(self.fluxes[:, :, :], axis=(1, 2))
        

        self.update_parafile()


        if model_params is not None:
            test_model = make_model(model_params, self.vsyst)
            model = test_model.model          # this is model.line

            # convolve model
            image = model.lines[:, :, :]
            sigma_x = cube.bmin / model.pixelscale * FWHM_to_sigma  # in pixels
            sigma_y = cube.bmaj / model.pixelscale * FWHM_to_sigma  # in pixels
            beam = Gaussian2DKernel(sigma_x, sigma_y, cube.bpa * np.pi / 180)
            for iv in range(image.shape[0]):
                image[iv,:,:] = convolve_fft(image[iv,:,:], beam)
            rms = np.nanstd(cube.image[[0,-1],:,:])
            noise = np.random.randn(image.size).reshape(image.shape)
            for iv in range(image.shape[0]):
                noise[iv,:,:] = convolve_fft(noise[iv,:,:], beam)
            noise *= rms / np.std(noise)
            image += noise
            self.fluxes = np.array(image)



    def update_parafile(self):

        import pymcfost as mcfost

        updating = mcfost.Params('model.para')

        updating.map.distance = self.distance
        updating.mol.molecule[0].nv = len(self.channels)
        updating.mol.molecule[0].v_min = self.channels[0]
        updating.mol.molecule[0].v_max = self.channels[-1]

        updating.writeto('model.para')



        
    def bilby_mcmc_fit(self, method='cube', nwalkers=64, nsteps=100, npool=16, ozstar=False, outfile=None):

        import bilby
        import emcee

        # Labels for bilby directories
        label = method
        outdir = 'bilby'+method

        if method=='cube':
            likelihood = mcfost_likelihood(self.channels, self.fluxes, self.uncertainty, method, self.vsyst, ozstar)
        else:
            print('only cube fitting works for now!')
            return 0
        
        priors = importlib.import_module('priors')
        priors_dict = priors.priors
        bilbypriors = {}

        for parameter in priors_dict:
            pri_type = priors_dict[parameter][0]
            if pri_type == 'Gaussian':
                mu = priors_dict[parameter][1][0]
                sigma = priors_dict[parameter][1][1]
                bilbypriors[parameter]=bilby.core.prior.Gaussian(mu=mu, sigma=sigma, name=parameter)
            elif pri_type == 'Uniform':
                minimum = priors_dict[parameter][1][0]
                maximum = priors_dict[parameter][1][1]
                bilbypriors[parameter]=bilby.core.prior.Uniform(minimum=minimum, maximum=maximum, name=parameter)
            elif pri_type == 'LogUniform':
                minimum = priors_dict[parameter][1][0]
                maximum = priors_dict[parameter][1][1]
                bilbypriors[parameter]=bilby.core.prior.LogUniform(minimum=minimum, maximum=maximum, name=parameter)
            elif pri_type == 'TruncatedGaussian':
                mu = priors_dict[parameter][1][0]
                sigma = priors_dict[parameter][1][1]
                minimum = priors_dict[parameter][1][2]
                maximum = priors_dict[parameter][1][3]
                bilbypriors[parameter]=bilby.core.prior.TruncatedGaussian(mu=mu, sigma=sigma, minimum=minimum, maximum=maximum, name=parameter)
            else:
                print('Prior currently not implemented')
                return 0

        # And run sampler
        backend = emcee.backends.HDFBackend(outfile)
        backend.reset(nwalkers, len(priors_dict))
        if not os.path.isfile(outfile):
            print('Starting from scratch!')
        else:
            print("Initial size: {0}".format(backend.iteration))
            nsteps = nsteps - backend.iteration        
        result = bilby.run_sampler(likelihood=likelihood, 
                                   priors=bilbypriors, sampler='emcee',
                                   nwalkers=nwalkers, nsteps=nsteps, npool=npool, 
                                   label=label, outdir=outdir, burn_in_act=2, backend=backend)

        result.plot_corner()


    def emcee_mcmc(self, Nwalk, Ninits, Nsteps, outpost='backend.h5', append=False, Nthreads=4):
        
        import emcee

        # param??

        print('Initialising priors')
        priors = importlib.import_module('priors')
        Ndim = len(priors.pri_pars)
        p0 = self.mcfost_priors(priors, Nwalk, Ndim)

        if not append:
            # Initialize the MCMC walkers
            print('Initialising walkers')
            with Pool(processes=Nthreads) as pool:
                isamp = emcee.EnsembleSampler(Nwalk, Ndim, self.log_posterior,    
                                                pool=pool)
                isamp.run_mcmc(p0, Ninits, progress=True)
            isamples = isamp.get_chain()   # [Ninits, Nwalk, Ndim]-shaped
            lop0 = np.quantile(isamples[-1, :, :], 0.25, axis=0)
            hip0 = np.quantile(isamples[-1, :, :], 0.75, axis=0)
            p00 = [np.random.uniform(lop0, hip0, Ndim) for iw in range(Nwalk)]

            # Prepare the backend
            os.system('rm -rf '+outpost)
            backend = emcee.backends.HDFBackend(outpost)
            backend.reset(Nwalk, Ndim)

            # Sample the posterior distribution
            print('Full MCMC')
            with Pool(processes=Nthreads) as pool:
                samp = emcee.EnsembleSampler(Nwalk, Ndim, self.log_posterior,
                                                pool=pool, backend=backend)
                t0 = time.time()
                samp.run_mcmc(p00, Nsteps, progress=True)
            t1 = time.time()
            print('backend run in ', t1-t0)
        else:
            # Load the old backend
            new_backend = emcee.backends.HDFBackend(outpost)
            print("Initial size: {0}".format(new_backend.iteration))
            
            # Continue sampling the posterior distribution
            with Pool(processes=Nthreads) as pool:
                samp = emcee.EnsembleSampler(Nwalk, Ndim, self.log_posterior,
                                                pool=pool, backend=new_backend)
                t0 = time.time()
                samp.run_mcmc(None, Nsteps-new_backend.iteration, progress=True)
            t1 = time.time()

        print('\n\n    This run took %.2f hours' % ((t1 - t0) / 3600))                        



    def mcfost_priors(self, priors, nwalk, ndim):
        p0 = np.empty((nwalk, ndim))
        for ix in range(ndim):
            if priors.pri_types[ix] == "normal" or priors.pri_types[ix] == "uniform":
                _ = [str(priors.pri_pars[ix][ip])+', ' for ip in range(len(priors.pri_pars[ix]))]
                cmd = 'np.random.'+priors.pri_types[ix]+'('+"".join(_)+str(nwalk)+')'
                p0[:,ix] = eval(cmd)
            elif priors.pri_types[ix] == "truncnorm" or priors.pri_types[ix] == "loguniform":
                if priors.pri_types[ix] == "truncnorm":
                    params = priors.pri_pars[ix]
                    mod_pri_pars = [(params[2]-params[0])/params[1], (params[3]-params[0])/params[1], params[0], params[1]]
                    _ = [str(mod_pri_pars[ip])+', ' for ip in range(len(mod_pri_pars))]
                else:
                    _ = [str(priors.pri_pars[ix][ip])+', ' for ip in range(len(priors.pri_pars[ix]))]
                cmd = 'stats.'+priors.pri_types[ix]+'.rvs('+"".join(_)+'size='+str(nwalk)+')'
                p0[:,ix] = eval(cmd)
            else:
                raise NameError('Prior type unaccounted for')
        return p0
    
    def log_posterior(self, theta, param=None, fromvis=None):

        # Calculate log-prior
        priors = importlib.import_module('priors')
        lnT = np.sum(priors.logprior(theta))
        if lnT == -np.inf:
            return -np.inf, -np.inf

        # Compute log-likelihood
        lnL = self.log_likelihood(theta, param, fromvis=fromvis)

        print('likelihood = ', lnL)
        print('prior = ', lnT)

        # return the log-posterior and the log-prior
        return lnL + lnT #, lnT
    
    def log_posterior_brute(self, theta, param=None, fromvis=None):

        # Calculate log-prior
        priorfile = 'priors_MCFOST_'
        for par in param:
            priorfile+= par+'_'
        priors = importlib.import_module(priorfile[:-1])
        lnT = np.sum(priors.logprior(theta))
        if lnT == -np.inf:
            return -np.inf, -np.inf

        # Compute log-likelihood
        lnL = self.log_likelihood(theta, param, brute=True, fromvis=fromvis)

        print('likelihood = ', lnL)
        print('prior = ', lnT)

        # return the log-posterior and the log-prior
        return lnL + lnT #, lnT
    

    def log_likelihood(self, theta, params=None, brute=False, fromvis=None, line_profile=False):
        if fromvis is not None:
            mcfost_model = fromvis  # this is just model.Line
        else:
            if params is not None:
                theta_dict = {}
                for i in range(len(params)):
                    theta_dict[params[i]] = theta[i]
            if brute:
                from myfittingpackage.generate_model_movie import make_model
            mcfost_model = make_model(theta_dict, vsyst=self.vsyst, ozstar=False).model

        # convolve model
        # image = mcfost_model.lines[:, :, :]
        # sigma_x = self.bmin / mcfost_model.pixelscale * FWHM_to_sigma  # in pixels
        # sigma_y = self.bmaj / mcfost_model.pixelscale * FWHM_to_sigma  # in pixels
        # beam = Gaussian2DKernel(sigma_x, sigma_y, self.bpa * np.pi / 180)
        # convolved_model = []
        # for iv in range(image.shape[0]):
        #     convolved_model.append(convolve_fft(image[iv], beam))
        # convolved_model = np.array(convolved_model)
        model = []
        print('Convolving with mcfost')
        for i in range(len(mcfost_model.lines)):
            mcfost_model.plot_map(iv=i, bmaj=self.bmaj, bmin=self.bmin, bpa=self.bpa)
            model.append(mcfost_model.last_image)
            plt.close()
        convolved_model = np.array(model)

        ##################################
       
        n = 1
        for i in range(len(self.fluxes.shape)):
            n = n*self.fluxes.shape[i]
        res = self.fluxes - convolved_model
        logL = -0.5*(np.sum((res / self.uncertainty)**2)
                    + n*np.log(2*np.pi)
                               *self.uncertainty**2)
        logL = -0.5*np.sum(res**2)

        if line_profile:
            model_line_profile = np.sum(convolved_model[:, :, :], axis=(1, 2))
            yerr = 2 #hardcoded
            red_chisq = np.sum((self.data_line_profile-model_line_profile)**2/yerr**2)/(len(model_line_profile)-1)
            return logL, red_chisq

        return logL
    

    def brute_force(self, params):

        # param needs to be a list so that we can do 2 param grid search

        self.values = None

        for param in params:       
            values = []
            if param == 'inclination':
                for i in range(30):
                    values.append([25+i])
            elif param == 'stellar_mass':
                for i in range(30):
                    values.append([round(0.1 + 0.03*i, 4)])
            elif param == 'scale_height':
                for i in range(30):
                    values.append([5 + 0.7*i])
            elif param == 'r_c':
                for i in range(30):
                    values.append([200 + 6*i])
            elif param == 'flaring_exp':
                for i in range(30):
                    values.append([1 + i/30])
            elif param == 'PA':
                for i in range(360):
                    values.append([i])
            elif param == 'dust_param':
                for i in range(100):
                    values.append([10**(-5 + i/50)])
            elif param == 'vturb':
                for i in range(30):
                    values.append([i/100])
            elif param == 'dust_mass':
                for i in range(100):
                    values.append([10**(-4 + 0.03*i)])
            elif param == 'gasdust_ratio':
                for i in range(100):
                    values.append([10**(0.03*i)])
            else:
                print("Not a valid parameter")
                return        
            
            if self.values is None:
                self.values = values
            else:
                full_values = []
                for value in self.values:
                    for second_value in values:
                        full_values.append([value[0], second_value[0]])
                self.values = full_values

        with multiprocessing.Pool() as pool:
            posteriors = pool.starmap(self.log_posterior_brute, [(value, params) for value in self.values])
        
        print(values)
        print(posteriors)

        name = ''
        for param in params:
            name+= param+'_'
        stored_results = name + 'results_image.npz'
        np.savez(stored_results, self.values, posteriors)

        # plt.figure()
        # plt.plot(values, posteriors)
        # plt.title('Log posterior as a function of ' + param)
        # plt.xlabel(param)
        # plt.ylabel('Log posterior')
        # plt.savefig(param+'lnposterior.pdf')

        return posteriors



#################################################################################################################


class model_to_model_fit:

    def __init__(self,
                 datacube: None,
                 params: float = None,
                 uncertainty: float = None,
                 vsyst: float = None,
                 nwalkers=64, 
                 nsteps=100, 
                 npool=16, 
                 ozstar=False,
                 outfile=None,
                 **kwargs):
        

        if params is None:
            print('Need to specify parameters for model comparison')
            return

        if isinstance(datacube, str):
            print("Reading cube ...")
            cube = casa.Cube(datacube)
        else:
            print('Need a valid cube name!')
            return

        self.cube = cube

        test_model = make_model(params, vsyst)
        model = test_model.model          # this is model.line

        # convolve model
        convolved_model = []
        for i in range(len(model.lines)):
            model.plot_map(iv=i, bmaj=self.cube.bmaj, bmin=self.cube.bmin, bpa=self.cube.bpa)
            convolved_model.append(model.last_image)
            plt.close()

        self.channels = model.velocity
        self.fluxes = np.array(convolved_model)
        self.uncertainty = uncertainty
        self.vsyst = vsyst

        self.outfile = outfile

        


    def run_fit(self, nwalkers, nsteps, npool, ozstar, method, outfile):

        import bilby
        import emcee

        # Labels for bilby directories
        label = method
        outdir = 'bilby'+method

        if method=='cube':
            likelihood = mcfost_likelihood(self.channels, self.fluxes, self.uncertainty, method, self.vsyst, ozstar)
        else:
            print('only cube fitting works for now!')
            return 0
        
        priors = importlib.import_module('priors')
        priors_dict = priors.priors
        bilbypriors = {}

        for parameter in priors_dict:
            pri_type = priors_dict[parameter][0]
            if pri_type == 'Gaussian':
                mu = priors_dict[parameter][1][0]
                sigma = priors_dict[parameter][1][1]
                bilbypriors[parameter]=bilby.core.prior.Gaussian(mu=mu, sigma=sigma, name=parameter)
            elif pri_type == 'Uniform':
                minimum = priors_dict[parameter][1][0]
                maximum = priors_dict[parameter][1][1]
                bilbypriors[parameter]=bilby.core.prior.Uniform(minimum=minimum, maximum=maximum, name=parameter)
            elif pri_type == 'LogUniform':
                minimum = priors_dict[parameter][1][0]
                maximum = priors_dict[parameter][1][1]
                bilbypriors[parameter]=bilby.core.prior.LogUniform(minimum=minimum, maximum=maximum, name=parameter)
            elif pri_type == 'TruncatedGaussian':
                mu = priors_dict[parameter][1][0]
                sigma = priors_dict[parameter][1][1]
                minimum = priors_dict[parameter][1][2]
                maximum = priors_dict[parameter][1][3]
                bilbypriors[parameter]=bilby.core.prior.TruncatedGaussian(mu=mu, sigma=sigma, minimum=minimum, maximum=maximum, name=parameter)
            else:
                print('Prior currently not implemented')
                return 0

        # And run sampler
        backend = emcee.backends.HDFBackend(outfile)
        print("Initial size: {0}".format(backend.iteration))
        steps = nsteps - backend.iteration
        result = bilby.run_sampler(likelihood=likelihood, 
                                   priors=bilbypriors, sampler='emcee',
                                   nwalkers=nwalkers, nsteps=steps, npool=npool, 
                                   label=label, outdir=outdir, burn_in_act=2, backend=backend)

        result.plot_corner()