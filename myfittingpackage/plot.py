import os
import sys
import numpy as np
import multiprocess
from functools import partial
#from casatasks import exportfits
import emcee

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Ellipse
from astropy.io import fits
import scipy.constants as sc
import cmasher as cmr
from myfittingpackage.generate_model import make_model

def make_plot(params, data, likelihood=None, iteration=None):

    model = make_model(params, vsyst=6.04)
    model.compare_with_data(data)

    return 



def best_model(h5file, data):

    filename = h5file
    reader = emcee.backends.HDFBackend(filename)
    all_samples = reader.get_chain(discard=0, flat=False)
    logpost_samples = reader.get_log_prob(discard=0, flat=False)
    best_model_logprob = -np.inf

    for iteration in range(len(all_samples)):
        for walker in logpost_samples[iteration]:
            if walker > best_model_logprob:
                best_model_logprob = walker
    best_index = np.argwhere(logpost_samples==best_model_logprob)
    if len(best_index) > 1:
        best_index = best_index[-1]
    i, j = best_index

    best_params = all_samples[i][j]
    best_likelihood = logpost_samples[i][j]
    print(best_params)

    # check that I can retrieve it

    # make plot
    make_plot(best_params, data, likelihood=best_likelihood, iteration=i)
    return