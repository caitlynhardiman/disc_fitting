import pymcfost as mcfost
import scipy.constants as sc
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import subprocess
from datetime import datetime
import casa_cube as casa
import multiprocess


class make_model:

    def __init__(self, params=None, vsyst=None, ozstar=False):

        self.directory = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        if params is None:
            print('No parameters provided - running for default configuration')
            model = self.write_run_mcfost()
        elif isinstance(params, dict):
            dir_name = ''
            for param in params:
                dir_name += param+'_'+str(params[param])+'_'
            self.directory = dir_name[:-1]
            model = self.write_run_mcfost(vsyst=vsyst, ozstar=ozstar, **params)
        else:
            inclination, stellar_mass, scale_height, r_c, flaring_exp, PA, dust_param, vturb, dust_mass, gasdust_ratio = params
            model = self.write_run_mcfost(inclination, stellar_mass, scale_height, r_c, flaring_exp, PA, dust_param, vturb, dust_mass, gasdust_ratio, vsyst, ozstar)

        self.model = model


    def write_run_mcfost(self, inclination=None, stellar_mass=None, scale_height=None,
                     r_c=None, flaring_exp=None, PA=None, dust_param=None,
                     vturb=None, dust_mass=None, gasdust_ratio=None, vsyst=None,
                     ozstar=False):
        # Rewrite mcfost para file

        if ozstar:
            jobfs = os.getenv("JOBFS")
            directory = jobfs+"/"+self.directory
        else:
            directory = self.directory

        if os.path.isdir(directory) == False:
            subprocess.call("mkdir "+directory, shell = True)
        updating = mcfost.Params('model.para')

        if inclination is not None:
            updating.map.RT_imin = 180-inclination
            updating.map.RT_imax = 180-inclination
        if stellar_mass is not None:
            updating.stars[0].M = stellar_mass
        if scale_height is not None:
            updating.zones[0].h0 = scale_height
        if r_c is not None:
            updating.zones[0].Rc = r_c
        if flaring_exp is not None:
            updating.zones[0].flaring_exp = flaring_exp
        if PA is not None:
            updating.map.PA = PA+180
        if dust_param is not None:
            updating.simu.viscosity = dust_param
        if vturb is not None:
            updating.mol.v_turb = vturb
        if dust_mass is not None:
            updating.zones[0].dust_mass = dust_mass
        if gasdust_ratio is not None:
            updating.zones[0].gas_to_dust_ratio = gasdust_ratio

        updating.mol.molecule[0].nv = 9

        para = directory+'/model.para'
        updating.writeto(para)
        origin = os.getcwd()
        os.chdir(directory)
        if vsyst is not None:
            options = "-mol -casa -photodissociation -v_syst " + str(vsyst)
        else:
            options = "-mol -casa -photodissociation"
        mcfost.run('model.para', options=options, delete_previous=True, logfile='mcfost.log')
        os.chdir(origin)
        model = mcfost.Line(directory+'/data_CO/')
        return model
    

    def compare_with_data(self, cube, pix_area, beam_area, logL=None):


        model = mcfost.Line(self.directory+'/data_CO')
        residuals = mcfost.Line(self.directory+'/data_CO')

        velocities = model.velocity

        exocubelines = cube.image * pix_area/beam_area
        exocubelines[np.isnan(exocubelines)] = 0
        model_chans = []
        exocube_chans = []

        for vel in velocities:
            iv = np.abs(cube.velocity - vel).argmin()
            exocube_chans.append(exocubelines[iv])

        for vel in velocities:
            iv = np.abs(model.velocity - vel).argmin()
            model_chans.append(model.lines[iv])

        model_chans = np.array(model_chans)
        exocube_chans = np.array(exocube_chans)
        residuals.lines = exocube_chans - model_chans

        # Plotting arguments
        fmax = 0.05
        cmap = 'Blues'
        fmin = 0
        colorbar = False
        vlabel_color = 'black'
        lim = 6.99
        limits = [lim, -lim, -lim, lim]
        no_ylabel = False

        fig, axs = plt.subplots(3, 9, figsize=(18, 6), sharex='all', sharey='all')
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        for i in range(9):
            if i != 0:
                no_ylabel = True
            if i == 8:
                colorbar = True
            if i != 4:
                no_xlabel = True
            else:
                no_xlabel = False
            cube.plot(ax=axs[0, i], v=velocities[i], fmin=fmin, fmax=fmax, cmap=cmap, colorbar=colorbar, no_vlabel=False, vlabel_color='black', limits=limits, no_xlabel=True, no_ylabel=True)
            axs[0, i].get_xaxis().set_visible(False)
            print('Per beam')
            model.plot_map(ax=axs[1, i], v=velocities[i],  bmaj=cube.bmaj, bmin=cube.bmin, bpa=cube.bpa, fmin=fmin, fmax=fmax, cmap=cmap, colorbar=colorbar, per_beam=True, limits=limits, no_xlabel=True, no_ylabel=no_ylabel, no_vlabel=False, no_xticks=True)
            residuals.plot_map(ax=axs[2, i], v=velocities[i],  bmaj=cube.bmaj, bmin=cube.bmin, bpa=cube.bpa, fmin=-fmax, fmax=fmax, cmap='RdBu', colorbar=colorbar, per_beam=True, limits=limits, no_ylabel=True, no_vlabel=False, no_xlabel=no_xlabel)            

        plt.savefig(self.directory+"/comparison.png", bbox_inches="tight", pad_inches=0.1, dpi=300, transparent=False)
        plt.show()