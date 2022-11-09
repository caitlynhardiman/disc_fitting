import pymcfost as mcfost


# Run mcfost to generate model with new parameters
def write_run_mcfost(inclination, stellar_mass, scale_height, r_c, r_in, flaring_exp, PA, dust_param, counter):
    counter[0]+=1
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
    #updating.mol.v_turb = vturb
    # Update para file
    updating.writeto('dmtau.para')
    # Run mcfost
    mcfost.run('dmtau.para', options="-mol -casa -photodissociation", delete_previous=True)
    model = mcfost.Line('data_CO/')
    return model

# Add in case for when we are running in parallel - need to name directory and then delete
