import numpy as np

class standard_likelihood(bilby.Likelihood):
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



class csalt_likelihood(bilby.Likelihood):

    from csalt import parametric_disk_mcfost, fit, models

    def __init__(self, params, data, fixed):



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

        # call csalt here with the parameters and the mcfost option
        # might need to hardcode in some DM Tau visibility parameters for now
        # need to generate the csalt model previously so it doesn't need to be done every time


        # Christophe's part - make mcfost model in csalt and return
        mcfost_model = parametric_disk_mcfost.parametric_disk(params)

        global data_
        data_ = data
        global fixed_
        fixed_ = fixed

        # calculate likelihood
        
        likelihood = fit.lnprob(params)

        return likelihood



###############################################################################
