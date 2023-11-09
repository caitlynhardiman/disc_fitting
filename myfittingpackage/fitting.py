import numpy as np
import matplotlib.pyplot as plt
import pymcfost as mcfost
import casa_cube as casa
from scipy.optimize import shgo
import os
import subprocess
import multiprocess
import importlib
from myfittingpackage.bilby_likelihood import mcfost_likelihood



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
        updating.mol.v_turb = vturb
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
                 distance: None,
                 uncertainty: float = None,
                 **kwargs):
        
        if distance is None:
            print('Need to provide a distance!')
            return
        else:
            self.distance = distance

        if isinstance(datacube, str):
            print("Reading cube ...")
            cube = casa.Cube(datacube)
        else:
            print('Need a valid cube name!')
            return

        if cube.nx > 256:
            rescale = 256/cube.nx
            print('Need to resize cube...')
            cube = casa.Cube(datacube, zoom=rescale)

        self.cube = cube
        self.beam_area = self.cube.bmin * self.cube.bmaj * np.pi / (4.0 * np.log(2.0))
        self.pix_area = self.cube.pixelscale**2
        self.uncertainty = uncertainty * self.pix_area/self.beam_area

        # finding 9 channels for comparison - assuming cube is centered on systemic velocity

        flux_vals = np.array(self.cube.image)
        flux_vals[np.isnan(flux_vals)] = 0
        flux_vals = flux_vals*self.pix_area/self.beam_area
        lp_fluxes = np.sum(flux_vals[:,:,:], axis=(1, 2))

        systemic_index = int(self.cube.nv/2)
        left_peak = np.argmax(lp_fluxes[:systemic_index])
        right_peak = np.argmax(lp_fluxes[systemic_index:])+systemic_index
        spacing = np.max([np.abs(left_peak-systemic_index), np.abs(right_peak-systemic_index)])
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
        self.vsyst = self.cube.velocity[systemic_index]

        self.update_parafile()



    def update_parafile(self):

        import pymcfost

        updating = mcfost.Params('model.para')

        updating.map.distance = self.distance
        updating.mol.molecule[0].nv = len(self.channels)
        updating.mol.molecule[0].v_min = self.channels[0]
        updating.mol.molecule[0].v_max = self.channels[-1]

        updating.writeto('model.para')



        
    def bilby_mcmc_fit(self, method='cube'):

        import bilby

        # Labels for bilby directories
        label = method
        outdir = 'bilby'+method

        if method=='cube':
            likelihood = mcfost_likelihood(self.channels, self.fluxes, self.uncertainty, method, self.vsyst)
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
        result = bilby.run_sampler(likelihood=likelihood, 
                                   priors=bilbypriors, sampler='emcee',
                                   nwalkers = 64, nsteps=25, npool=64, 
                                   nburn=0, label=label, outdir=outdir)

        result.plot_corner()
