import numpy as np
import bilby
import pymcfost as mcfost
import os
import subprocess
import multiprocess

class mcfost_likelihood(bilby.Likelihood):
    def __init__(self, x, y, sigma, method, vsyst, ozstar):

        self.x = x
        self.y = y
        self.sigma = sigma
        n = 1
        for i in range(len(y.shape)):
            n = n*y.shape[i]
        self.N = n
        self.method = method
        self.vsyst = vsyst
        self.ozstar = ozstar

        super().__init__(parameters={'inclination': None,
                                     'stellar_mass': None,
                                     'scale_height': None,
                                     'r_c': None,
                                     'r_in': None,
                                     'flaring_exp': None,
                                     'PA': None,
                                     'dust_param': None,
                                     'vturb': None,
                                     'dust_mass': None, 
                                     'gasdust_ratio': None})


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
        dust_mass = self.parameters['dust_mass']
        gasdust_ratio = self.parameters['gasdust_ratio']

        if self.method=='cube':
            mcfost_model = self.cube_flux(self.x, inc, mass, h, rc, rin, psi, pa, dust_param, vturb, dust_mass, gasdust_ratio, self.vsyst)
        else:
            print('only cube fitting implemented for now')
            return 0
        res = self.y - mcfost_model

        return -0.5 * (np.sum((res / self.sigma)**2)
                       + self.N*np.log(2*np.pi*self.sigma**2))



    def cube_flux(self, velax, inc, mass, h, rc, rin, psi, pa, dust_param, vturb, dust_mass, gasdust_ratio, vsyst):
        # Rewrite mcfost para file
        pool_id = multiprocess.current_process()
        pool_id = pool_id.pid
        if self.ozstar:
            jobfs = os.getenv("JOBFS")
            directory = jobfs+"/"+str(pool_id)
        else:
            directory = str(pool_id)
        if os.path.isdir(directory) == False:
            subprocess.call("mkdir "+directory, shell = True)
       
        updating = mcfost.Params('model.para')
        if inc is not None:
            updating.map.RT_imin = inc+180
            updating.map.RT_imax = inc+180
        if mass is not None:
            updating.stars[0].M = mass
        if h is not None:
            updating.zones[0].h0 = h
        if rc is not None:
            updating.zones[0].Rc = rc
        if rin is not None:
            updating.zones[0].Rin = rin
        if psi is not None:
            updating.zones[0].flaring_exp = psi
        if pa is not None:
            updating.map.PA = pa
        if dust_param is not None:
            updating.simu.viscosity = dust_param
        if vturb is not None:
            updating.mol.v_turb = vturb
        if dust_mass is not None:
            updating.zones[0].dust_mass = dust_mass
        if gasdust_ratio is not None:
            updating.zones[0].gas_to_dust_ratio = gasdust_ratio

        para = directory+'/csalt_'+str(pool_id)+'.para'
        updating.writeto(para)
        origin = os.getcwd()
        os.chdir(directory)
        if vsyst is not None:
            options = "-mol -casa -photodissociation -v_syst " + str(vsyst)
        else:
            options = "-mol -casa -photodissociation"
        mcfost.run('csalt_'+str(pool_id)+'.para', options=options, delete_previous=True, logfile='mcfost.log')
        os.chdir(origin)
        model = mcfost.Line(directory+'/data_CO/')
        return model.lines