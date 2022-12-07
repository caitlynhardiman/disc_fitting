class Priors:

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
