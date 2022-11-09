import numpy as np
import matplotlib.pyplot as plt
import pymcfost as mcfost
import casa_cube as casa
from scipy.optimize import shgo




exocube = casa.Cube('~/Desktop/exoALMA discs/DM Tau/DM_Tau_12CO_robust0.5_width0.1kms_threshold4.0sigma.clean.JvMcorr.fits')
offset = 0.045766
beam_area = exocube.bmin * exocube.bmaj * np.pi / (4.0 * np.log(2.0))
pix_area = exocube.pixelscale**2
uncertainty = 8.412e-4 * pix_area/beam_area
interpolation = True
convolution = True
comparison = 'line'


if comparison == 'cube':
    # Interpolate cube?
    if interpolation == True:
        x, y = interpolate(exocube, offset, pix_area, beam_area, comparison)
    else:
        x = np.array([exocube.velocity[137], exocube.velocity[140], exocube.velocity[143],
                      exocube.velocity[146], exocube.velocity[149], exocube.velocity[152],
                      exocube.velocity[155], exocube.velocity[158], exocube.velocity[161]])
        flux_vals = np.array([exocube.image[137], exocube.image[140], exocube.image[143],
                              exocube.image[146], exocube.image[149], exocube.image[152],
                              exocube.image[155], exocube.image[158], exocube.image[161]])
        flux_vals[np.isnan(flux_vals)] = 0
        flux_vals = flux_vals*pix_area/beam_area
        y = flux_vals

    yerr = np.full_like(y, uncertainty)
else:
    # line profiles
    if interpolation == True:
        x, y = interpolate(exocube, offset, pix_area, beam_area, comparison)
    else:
        x = np.array([exocube.velocity[137], exocube.velocity[140], exocube.velocity[143],
                      exocube.velocity[146], exocube.velocity[149], exocube.velocity[152],
                      exocube.velocity[155], exocube.velocity[158], exocube.velocity[161]])
        flux_vals = np.array([exocube.image[137], exocube.image[140], exocube.image[143],
                              exocube.image[146], exocube.image[149], exocube.image[152],
                              exocube.image[155], exocube.image[158], exocube.image[161]])
        y = np.nansum(flux_vals[:,:,:], axis=(1,2)) * pix_area/beam_area

    yerr = np.empty(len(y)); yerr.fill(2.0)
