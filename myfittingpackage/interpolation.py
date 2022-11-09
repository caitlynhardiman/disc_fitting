import numpy as np

def get_channel(cubename, v):
    """
       get channel corresponding to specified velocity, by interpolating from neighbouring channels
    """
    iv = np.abs(cubename.velocity - v).argmin()

    # do not interpolate past bounds of the array, ends just return end channel
    ivp1 = iv+1
    ivm1 = iv-1
    if (ivp1 > len(cubename.velocity)-1 or ivm1 < 0):
       return cubename.image[iv,:,:],iv,iv

    # deal with channels in either increasing or decreasing order
    if ((cubename.velocity[iv] < v and cubename.velocity[ivp1] >= v) or (cubename.velocity[ivp1] <= v and cubename.velocity[iv] > v)):
       iv1 = ivp1
    else:
       iv1 = ivm1

    c1 = np.nan_to_num(cubename.image[iv,:,:])
    c2 = np.nan_to_num(cubename.image[iv1,:,:])
    dv = v - cubename.velocity[iv]
    deltav = cubename.velocity[iv1] - cubename.velocity[iv]
    x = dv/deltav
    #print("retrieving channel at v=",v," between ",cubename.velocity[iv]," and ",cubename.velocity[iv1]," pos = ",x)
    return c1*(1.-x) + x*c2

def interpolate(cubename, offset, pix_area, beam_area, method):
    original_velocity = np.array([cubename.velocity[137], cubename.velocity[140], cubename.velocity[143],
                                  cubename.velocity[146], cubename.velocity[149], cubename.velocity[152],
                                  cubename.velocity[155], cubename.velocity[158], cubename.velocity[161]])
    x = []
    for vel in original_velocity:
        x.append(vel+offset)
    y = []
    for vel in x:
        flux = get_channel(vel)
        if method == 'cube':
            flux_val = flux[165:880, 180:855] * pix_area/beam_area
        else:
            flux_val = np.sum(flux[165:880, 180:855]) * pix_area/beam_area
        y.append(flux_val)

    return x, y
