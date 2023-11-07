priors = {'inclination': ['Gaussian', (36, 1)],
          'stellar_mass': ['Uniform', (0.2, 0.6)],
          'scale_height': ['Uniform', (7, 25)],
          'r_c': ['Uniform', (100, 300)],
          'r_in': ['Uniform', (0.1, 10)],
          'flaring_exp': ['TruncatedGaussian', (1.3, 0.1, 1, 2)],
          'PA': ['Gaussian', (156, 2)],
          'dust_param': ['LogUniform', (1e-5, 1e-3)],
          'vturb': ['Uniform', (0.0, 0.2)],
          'dust_mass': ['Uniform', (5e-4, 1e-3)],
          'gasdust_ratio': ['Uniform', (10, 1000)]
          }