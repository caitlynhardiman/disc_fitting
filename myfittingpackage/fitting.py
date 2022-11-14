import numpy as np
import matplotlib.pyplot as plt
import pymcfost as mcfost
import casa_cube as casa
from scipy.optimize import shgo



class Param_Ranges:

    def __init__(self,
                 inc: tuple = None,
                 stellar_mass: tuple = None,
                 scale_height: tuple = None,
                 r_c: tuple = None,
                 r_in: tuple = None,
                 psi: tuple = None,
                 PA: tuple = None,
                 dust_alpha: tuple = None,
                 vturb: tuple = None):

        params = []
        bounds = []

        self.inc = inc
        self.mass = stellar_mass
        self.h = scale_height
        self.rc = r_c
        self.rin = r_in
        self.psi = psi
        self.PA = PA
        self.dust_alpha = dust_alpha
        self.vturb = vturb

        if inc is not None:
            params.append('Inclination')
            bounds.append(inc)
        if stellar_mass is not None:
            params.append('Stellar Mass')
            bounds.append(stellar_mass)
        if scale_height is not None:
            params.append('Scale Height')
            bounds.append(scale_height)
        if r_c is not None:
            params.append('R_c')
            bounds.append(r_c)
        if r_in is not None:
            params.append('R_in')
            bounds.append(r_in)
        if psi is not None:
            params.append('Flaring Exponent')
            bounds.append(psi)
        if PA is not None:
            params.append('PA')
            bounds.append(PA)
        if dust_alpha is not None:
            params.append('Dust α')
            bounds.append(dust_alpha)
        if vturb is not None:
            params.append('Vturb')
            bounds.append(vturb)

        self.params = params
        self.bounds = bounds


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
        self.offset = offset
        self.uncertainty = uncertainty

        if interpolate:
            x, y = self._interpolate_cube()
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
            flux = get_channel(vel)
            flux_val = flux * self.pix_area/self.beam_area
            y.append(flux_val)

        return x, y

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

    def shgo_cube_fit(self, ranges, convolve: bool = True):

        # Initialise results file
        filename = 'shgo_fit_to_cube.txt'
        self._init_tracking(filename, ranges)

        # Run shgo optimization - still a bit hard coded here for DM Tau cube
        counter = [0, 50]
        channel = 4
        chi_args = lambda theta, x=self.channels, y=self.fluxes, yerr=self.uncertainty, counter=counter,
                channel=channel, filename=filename, convolve=convolve: red_chi_squared_cube(theta, x, y, yerr, counter, channel, filename, convolve)
        bounds = ranges.bounds
        soln = shgo(chi_args, bounds, iters=3, n=100, sampling_method='simplicial')
        best_fit = soln.x
        best_chi = soln.fun

        # Conclude results file
        self._best_result(filename, ranges, best_fit, best_chi)

    def red_chi_squared_cube(theta, x, y, yerr, counter, channel, filename, convolve):
        inclination, stellar_mass, scale_height, r_c, r_in, flaring_exp, PA, dust_param = theta
        mcfost_model = write_run_mcfost(inclination, stellar_mass, scale_height, r_c, r_in, flaring_exp, PA, dust_param, counter)
        sigma2 = yerr**2
        if convolve:
            model = convolve(mcfost_model)
        else:
            model = mcfost_model
        red_chisq = np.sum((np.array(y)-np.array(model))**2/yerr**2)/(len(model)*len(model[0])*len(model[0][0])-1)


    def shgo_line_profile_fit(self, bounds):

        # do stuff
