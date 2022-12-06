import numpy as np
import matplotlib.pyplot as plt
import pymcfost as mcfost
import casa_cube as casa
from scipy.optimize import shgo




# exocube = casa.Cube('~/Desktop/exoALMA discs/DM Tau/DM_Tau_12CO_robust0.5_width0.1kms_threshold4.0sigma.clean.JvMcorr.fits')
# offset = 0.045766
# beam_area = exocube.bmin * exocube.bmaj * np.pi / (4.0 * np.log(2.0))
# pix_area = exocube.pixelscale**2
# uncertainty = 8.412e-4 * pix_area/beam_area
# interpolation = True
# convolution = True
