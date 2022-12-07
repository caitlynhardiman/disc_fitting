import numpy as np
import matplotlib.pyplot as plt
import pymcfost as mcfost
import casa_cube as casa
from scipy.optimize import shgo
import os
import subprocess


class vis_disc:

    def __init__(self,
                 ms_file: None,
                 # potetially observing properties? Depends if they can be read from the ms file
                 **kwargs):

        # read in ms file
        # convert to csalt format if required?
        # obtain observing properties from file or define them here to be passed on later to csalt in the likelihood



    def visibility_mcmc_fit(self, ranges):

        import bilby
        from bilby_likelihood import csalt_likelihood
        import multiprocess

        # Labels for bilby directories
        label = 'visibility'
        outdir = 'bilby_'+label

        # Set up priors?

        # Dummy values
        nu_rest = 5
        FOV = 5
        Npix = 5
        dist = 5
        cfg_dict = {}

        fixed = nu_rest, FOV, Npix, dist, cfg_dict
        formatted_data = data.fitdata(datafile, vra=vra, vcensor=vcensor, nu_rest=fixed[0], chbin=chbin)

        # Need to pass disc parameters to csalt_likelihood - make them into an attribute? could set fixed straight away

        likelihood = csalt_likelihood(params, formatted_data, fixed)
        #
        # # And run sampler
        # if __name__ == "__main__":
        #     result = bilby.run_sampler(
        #     likelihood=likelihood, priors=priors, sampler='emcee',
        #     nwalkers = 90,
        #     nsteps=100,
        #     npool=8)
        #
        # result.plot_corner()

