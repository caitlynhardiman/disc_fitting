import numpy as np



# Bilby Likelihood Function - this can stay the same regardless of the comparison method (will just change the function)

class myLikelihood(bilby.Likelihood):
    def __init__(self, x, y, sigma, function):

        self.x = x
        self.y = y
        self.sigma = sigma
        self.N = len(x)
        self.function = function

        super().__init__(parameters={'inclination': None,
                                     'stellar_mass': None,
                                     'scale_height': None,
                                     'r_c': None,
                                     'r_in': None,
                                     'flaring_exp': None,
                                     'PA': None,
                                     'dust_param': None,
                                     'vturb': None})


    def log_likelihood(self):

        inc = self.parameters['inclination']
        mass = self.parameters['stellar_mass']
        h = self.parameters['scale_height']
        rc = self.parameters['r_c']
        rin = self.parameters['r_in']
        psi = self.parameters['flaring_exp']
        pa = self.parameters['PA']
        dust_param = self.parameters['dust_param']
        vturb = self.parameters['vturb']

        mcfost_model = self.function(self.x, inc, mass, h, rc, rin, psi, pa, dust_param, vturb)
        res = self.y - mcfost_model

        return -0.5 * (np.sum((res / self.sigma)**2)
                       + self.N*np.log(2*np.pi*self.sigma**2))


###############################################################################
