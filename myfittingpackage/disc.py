import numpy as np
import matplotlib.pyplot as plt
import pymcfost as mcfost
import casa_cube as casa
from scipy.optimize import shgo
import os
import subprocess


class Disc:

    def __init__(self,
                 cube: None,
                 zoom: float = None,
                 v_syst: float = None,
                 v_offset: float = None,
                 uncertainty: float = None,
                 interpolate: bool = True,
                 **kwargs):


        if isinstance(cube,str):
            print("Reading cube ...")
            cube = casa.Cube(cube, zoom=zoom)

        self.cube = cube
        self.beam_area = self.cube.bmin * self.cube.bmaj * np.pi / (4.0 * np.log(2.0))
        self.pix_area = self.cube.pixelscale**2
        self.offset = v_offset
        self.uncertainty = uncertainty * self.pix_area/self.beam_area

        if interpolate:
            if v_offset is not None:
                x, y = self._interpolate_cube()
            else:
                print("Need offset value to interpolate!")
        else:
            # fix it otherwise
            x = np.array([self.cube.velocity[137], self.cube.velocity[140], self.cube.velocity[143],
                      self.cube.velocity[146], self.cube.velocity[149], self.cube.velocity[152],
                      self.cube.velocity[155], self.cube.velocity[158], self.cube.velocity[161]])
            flux_vals = np.array([self.cube.image[137], self.cube.image[140], self.cube.image[143],
                              self.cube.image[146], self.cube.image[149], self.cube.image[152],
                              self.cube.image[155], self.cube.image[158], self.cube.image[161]])
            flux_vals[np.isnan(flux_vals)] = 0
            flux_vals = flux_vals*self.pix_area/self.beam_area
            y = flux_vals

        self.channels = x
        self.fluxes = y


    def _interpolate_cube(self):
        # this needs to be consistent for any cube!!! atm my 9 channels
        original_velocity = np.array([self.cube.velocity[137], self.cube.velocity[140], self.cube.velocity[143],
                                      self.cube.velocity[146], self.cube.velocity[149], self.cube.velocity[152],
                                      self.cube.velocity[155], self.cube.velocity[158], self.cube.velocity[161]])
        x = []
        for vel in original_velocity:
            x.append(vel+self.offset)
        y = []
        for vel in x:
            flux = self._get_channel(vel)
            flux_val = flux * self.pix_area/self.beam_area
            y.append(flux_val)

        return x, y

    # Interpolation for the data cube

    def _get_channel(self, v):
        """
           get channel corresponding to specified velocity, by interpolating from neighbouring channels
        """
        iv = np.abs(self.cube.velocity - v).argmin()

        # do not interpolate past bounds of the array, ends just return end channel
        ivp1 = iv+1
        ivm1 = iv-1
        if (ivp1 > len(self.cube.velocity)-1 or ivm1 < 0):
           return self.cube.image[iv,:,:],iv,iv

        # deal with channels in either increasing or decreasing order
        if ((self.cube.velocity[iv] < v and self.cube.velocity[ivp1] >= v) or (self.cube.velocity[ivp1] <= v and self.cube.velocity[iv] > v)):
           iv1 = ivp1
        else:
           iv1 = ivm1

        c1 = np.nan_to_num(self.cube.image[iv,:,:])
        c2 = np.nan_to_num(self.cube.image[iv1,:,:])
        dv = v - self.cube.velocity[iv]
        deltav = self.cube.velocity[iv1] - self.cube.velocity[iv]
        x = dv/deltav
        #print("retrieving channel at v=",v," between ",exocube.velocity[iv]," and ",exocube.velocity[iv1]," pos = ",x)
        return c1*(1.-x) + x*c2


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


    #def shgo_line_profile_fit(self, bounds):

        # do stuff



    def bilby_mcmc_fit(self, ranges, convolve: bool = True, method):

        import bilby
        from bilby_likelihood import standard_likelihood
        import multiprocess

        # Labels for bilby directories
        label = method
        outdir = 'bilby'+method

        # Using only uniform priors for now
        parameters = ['inc', 'stellar_mass', 'scale_height', 'r_c', 'r_in', 'psi', 'PA', 'dust_alpha', 'vturb']
        priors = dict()

        for param in parameters:
            if ranges.param is not None:
                priors[param] = bilby.core.prior.Uniform(ranges.param)

        # need to differentiate between methods here
        if method='cube':
            likelihood = myLikelihood(self.channels, self.fluxes, self.uncertainty, cube_flux)
        elif method='line':
            lp_fluxes = np.sum(self.fluxes[:,:,:], axis=(1, 2))
            uncertainty = np.empty(len(lp_fluxes)); uncertainty.fill(2.0)
            likelihood = myLikelihood(self.channels, lp_fluxes, uncertainty, line_profile_flux)
        else:
            # error - needs to be cube or line
            return

        # And run sampler
        if __name__ == "__main__":
            result = bilby.run_sampler(
            likelihood=likelihood, priors=priors, sampler='emcee',
            nwalkers = 90,
            nsteps=100,
            npool=8)

        result.plot_corner()

##################################################################################
