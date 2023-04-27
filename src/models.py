import os
import numpy as np
import jax
import types

from functools import partial
from prospect.models import ProspectorParams, priors
from prospect.models.sedmodel import SedModel
from astropy.cosmology import WMAP9
from scipy.special import erf, erfinv

#
# EMULATOR
#
class JAXGELUEmulator(ProspectorParams):
    def __init__(self, model_params, fp=None, obs=None, param_order=None):
        super(JAXGELUEmulator, self).__init__(model_params, param_order=param_order)

        # Load weights, biases, normalization data, and filter names.
        redir, normlayer, nnlayers, denormlayer, resid, emulbounds = np.load(fp, allow_pickle=True)
        # This redir dictionary is from Julia, which is 1-indexed. Hence the -1 here.
        self.sorter = [redir[f.name]-1 for f in obs['filters']]
        
        # If zred is not a free parameter, we'll need to know to add it before calling
        # the emulator. This should probably be modified for situations where *any* of
        # the 18 emulated parameters aren't free.
        if not model_params["zred"]["isfree"]:
            self.zred_in_theta = False
        else:
            self.zred_in_theta = True
        self.zred_index = 14

        # TODO: Check if prior bounds are with emulbounds.
        self.emulbounds = emulbounds

        # Extract test set residuals.
        # Currently, rows are filters and columns are:
        #  -4σ, -3σ, -2σ, -1σ, Median, +1σ, +2σ, +3σ, +4σ, Mean, StDev
        self.resid = resid

        # Extract NN data.
        self.norm_mu, self.norm_sig = normlayer
        self.nn_W, self.nn_b = nnlayers
        self.denorm_mu, self.denorm_sig = denormlayer

        # In cases where *any* of the emulator's parameters are set to
        # fixed values in a fit (such as zred getting set to z_best),
        # they aren't provided to the emulator in the input theta vector.
        # However, the emulator *needs* these values in order to predict
        # set of filter magnitudes, so we need to supply them to the
        # emulator somehow.
        #
        # This is how I've been doing it. I'm open to any improvements here!
        self.zred     = obs['zred']
        self.zred_pos = 14

        # Convert things to JAX arrays.
        self.norm_mu = jax.numpy.array(self.norm_mu)
        self.norm_sig = jax.numpy.array(self.norm_sig)

        # Select *only* the filters provided by the obs dictionary.
        self.nn_W[-1] = self.nn_W[-1][self.sorter, :]
        self.nn_b[-1] = self.nn_b[-1][self.sorter]

        # Convert more things to JAX arrays.
        self.nn_W = [jax.numpy.array(W) for W in self.nn_W]
        self.nn_b = [jax.numpy.array(b) for b in self.nn_b]

        # Again remove unneeded filters and convert to JAX arrays.
        self.denorm_mu = self.denorm_mu[np.array(self.sorter, dtype=int)]
        self.denorm_sig = self.denorm_sig[np.array(self.sorter, dtype=int)]
        self.denorm_mu = jax.numpy.array(self.denorm_mu)
        self.denorm_sig = jax.numpy.array(self.denorm_sig)

    # Add zred to theta if zred is fixed
    def modify_theta(self, theta):
        if self.zred_in_theta:
            return theta
        else:
            return np.insert(theta, self.zred_index, self.config_dict["zred"]["init"])
    
    @partial(jax.jit, static_argnums=(0,))
    def nonlinear_layers(self, norm_theta):
        result = norm_theta
        for i in range(len(self.nn_W) - 1):
            result = jax.nn.gelu(jax.numpy.dot(self.nn_W[i], result) + self.nn_b[i])
        return result
    
    @partial(jax.jit, static_argnums=(0,))
    def norm_layer(self, theta):
        return (theta - self.norm_mu) / self.norm_sig
    
    @partial(jax.jit, static_argnums=(0,))
    def linear_layer(self, in_vec):
        return jax.numpy.dot(self.nn_W[-1], in_vec) + self.nn_b[-1]
    
    @partial(jax.jit, static_argnums=(0,))
    def denorm_layer(self, norm_phot):
        return (self.denorm_sig * norm_phot) + self.denorm_mu
    
    @partial(jax.jit, static_argnums=(0,))
    def abmag_to_maggie(self, abmag):
        result = jax.numpy.multiply(abmag, -0.4)
        result = jax.numpy.divide(result, jax.numpy.log10(jax.numpy.e))
        result = jax.numpy.exp(result)
        return result
    
    def predict_phot_jax(self, theta_in, **extras):
        # Add redshift to the parameter vector - see the long comment above.
        # Note if you're adding more fixed parameters here, order matters!
        # zred is being appended to the end because it just so happens to be
        # the last element in the emulator's input.
        theta = self.modify_theta(theta_in)

        # Normalize layer.
        result = self.norm_layer(theta)

        # Nonlinear layers with GELU activation.
        result = self.nonlinear_layers(result)

        # Linear layer.
        result = self.linear_layer(result)

        # Denormalize layer.
        result = self.denorm_layer(result)

        # Convert back to maggies
        result = self.abmag_to_maggie(result)

        return result

    def predict_phot(self, theta_in, **extras):
        result = self.predict_phot_jax(theta_in)
        result = np.array(result)
        return result

    # We don't predict spectra, both because we don't have an emulator available
    # and because it would take too long anyway. Thus, let's just generate a vector
    # of zeros and return that in case Prospector complains about not having a
    # spectrum predicted.
    def predict_spec(self, theta, obs=None, **extras):
        return np.zeros(5994)

    # Same thing with mfrac, although we *do* actually have an mfrac emulator.
    def predict_mfrac(self, theta, obs=None, **extras):
        return -1.0

    # General predict function, returns useless mfrac and spec.
    def predict(self, theta, obs=None, **extras):
        return (self.predict_spec(theta, obs=obs, **extras),
                self.predict_phot(theta, obs=obs, **extras),
                self.predict_mfrac(theta, obs=obs, **extras))

def massmet_to_logmass(massmet=None, **extras):
    return massmet[0]

def massmet_to_logzsol(massmet=None, **extras):
    return massmet[1]

# Emulator's build_model function
def load_model_emulator(obs=None, emulfp=None, **extras):
    fit_order = ['massmet',
                 'logsfr_ratios',
                 'dust2',
                 'dust_index',
                 'dust1_fraction',
                 'log_fagn',
                 'log_agn_tau',
                 'gas_logz',
                 'zred',
                 'duste_qpah',
                 'duste_umin',
                 'log_duste_gamma']

    # -------------
    # MODEL_PARAMS
    model_params = {}

    # --- BASIC PARAMETERS ---
    model_params['zred'] = {'N': 1, 'isfree': False,
                            'init': obs["zred"],
                            'prior': FastUniform(a=1e-3, b=12.0)}

    model_params['logmass'] = {'N': 1, 'isfree': False,
                              'depends_on': massmet_to_logmass,
                              'init': 10.0,
                              'units': 'Msun',
                              'prior': FastUniform(a=7.0, b=12.5)}

    model_params['logzsol'] = {'N': 1, 'isfree': False,
                               'init': -0.5,
                               'depends_on': massmet_to_logzsol,
                               'units': r'$\log (Z/Z_\odot)$',
                               'prior': FastUniform(a=-1.98, b=0.19)}

    model_params['massmet'] = {'N': 2, 'isfree': True,
                               'init': np.array([10, -0.5]),
                               'prior': FastMassMet(a=6.0, b=12.5)}

    # --- SFH ---
    model_params['logsfr_ratios'] = {'N': 6, 'isfree': True,
                                     'init': 0.0,
                                     'prior': FastTruncatedEvenStudentTFreeDeg2(hw=np.ones(6)*5.0,
                                                                                sig=np.ones(6)*0.3)}

    # --- Dust Absorption ---
    model_params['dust1_fraction'] = {'N': 1, 'isfree': True,
                                      'init': 1.0,
                                      'prior': FastTruncatedNormal(a=0.0, b=2.0, mu=1.0, sig=0.3)}

    model_params['dust2'] = {'N': 1, 'isfree': True,
                             'init': 0.0,
                             'units': '',
                             'prior': FastTruncatedNormal(a=0.0, b=4.0, mu=0.3, sig=1.0)}

    model_params['dust_index'] = {'N': 1, 'isfree': True,
                                  'init': 0.7,
                                  'units': '',
                                  'prior': FastUniform(a=-1.2, b=0.4)}

    # --- Nebular Emission ---
    model_params['gas_logz'] = {'N': 1, 'isfree': True,
                                'init': -0.5,
                                'units': r'log Z/Z_\odot',
                                'prior': FastUniform(a=-2.0, b=0.5)}

    # --- AGN dust ---
    model_params['log_fagn'] = {'N': 1, 'isfree': True,
                                'init': -7.0e-5,
                                'prior': FastUniform(a=-5.0, b=np.log10(3.0))}

    model_params['log_agn_tau'] = {'N': 1, 'isfree': True,
                                   'init': np.log10(20.0),
                                   'prior': FastUniform(a=np.log10(5.0), b=np.log10(150.0))}

    # --- Dust Emission ---
    model_params['duste_qpah'] = {'N':1, 'isfree':True,
                                  'init': 2.0,
                                  'prior': FastTruncatedNormal(a=0.0, b=7.0, mu=2.0, sig=2.0)}

    model_params['duste_umin'] = {'N':1, 'isfree':True,
                                  'init': 1.0,
                                  'prior': FastTruncatedNormal(a=0.1, b=25.0, mu=1.0, sig=10.0)}

    model_params['log_duste_gamma'] = {'N':1, 'isfree':True,
                                       'init': -2.0,
                                       'prior': FastTruncatedNormal(a=-4.0, b=0.0, mu=-2.0, sig=1.0)}

    #---- Units ----
    model_params['peraa'] = {'N': 1, 'isfree': False, 'init': False}

    model_params['mass_units'] = {'N': 1, 'isfree': False, 'init': 'mformed'}

    extra = [k for k in model_params.keys() if k not in fit_order]
    fit_order = fit_order + list(extra)

    return JAXGELUEmulator(
        model_params,
        fp = emulfp,
        obs = obs,
        param_order = fit_order
    )



#
# FSPS
#
def xm(m, am=2.5*np.log10(np.exp(1.0)), m0=0.0):
    return np.exp(-1.0 * (m-m0)/am)

def ux(x, au=2.5*np.log10(np.exp(1.0)), u0=35.0):
    return -1.0 * au * np.arcsinh(0.5*x*np.exp(u0/au)) + u0

# Modify SedModel's predict method, since we need to convert the
# predicted fluxes to arsinh mags, assume they are log10 mags, and
# then convert back to linear flux.
def _predict(self, theta, obs=None, sps=None, **extras):
    s, p, x = self.sed(theta, obs, sps=sps, **extras)
    self._speccal = self.spec_calibration(obs=obs, **extras)
    if obs.get('logify_spectrum', False):
        s = np.log(s) + np.log(self._speccal)
    else:
        s *= self._speccal
    return s, xm(ux(p)), x

def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2

def zred_to_agebins(zred=None,agebins=None,amin=7.1295,nbins_sfh=7,**extras):
    tuniv = WMAP9.age(zred).value[0]*1e9
    tbinmax = (tuniv*0.9)
    if (zred[0] <= 3.):
        agelims = np.concatenate((
            np.array([0.0,7.47712]),
            np.linspace(8.0,np.log10(tbinmax),nbins_sfh-2),
            np.array([np.log10(tuniv)]),
        ))
    else:
        agelims = np.concatenate((
            np.linspace(amin,np.log10(tbinmax),nbins_sfh),
            np.array([np.log10(tuniv)]),
        ))
        agelims[0] = 0.0
        
    agebins = np.array([agelims[:-1], agelims[1:]])
    return agebins.T

def logmass_to_masses(massmet=None, logsfr_ratios=None, zred=None, **extras): 
    agebins = zred_to_agebins(zred=zred)
    logsfr_ratios = np.clip(logsfr_ratios,-10,10) # numerical issues...
    nbins = agebins.shape[0]
    sratios = np.exp(logsfr_ratios * np.log(10.0))
    dt = (10.0**agebins[:,1]-10.0**agebins[:,0])
    coeffs = np.array([ (1./np.prod(sratios[:i])) * (np.prod(dt[1:(i+1)]) / np.prod(dt[:i])) for i in range(nbins)])
    m1 = (np.exp(massmet[0] * np.log(10.0))) / coeffs.sum()
    return m1 * coeffs

def to_duste_gamma(log_duste_gamma=None, **extras):
    return 10.0**(log_duste_gamma)

def to_fagn(log_fagn=None, **extras):
    return 10.0**(log_fagn)

def to_agn_tau(log_agn_tau=None, **extras):
    return 10.0**(log_agn_tau)

def load_model_fsps(obs=None, sps=None, df=2, sigma=0.3, nbins_sfh=7, **extras):
    fit_order = ['massmet',
                 'logsfr_ratios',
                 'dust2',
                 'dust_index',
                 'dust1_fraction',
                 'log_fagn',
                 'log_agn_tau',
                 'gas_logz',
                 'zred',
                 'duste_qpah',
                 'duste_umin',
                 'log_duste_gamma']

    # -------------
    # MODEL_PARAMS
    model_params = {}

    # --- BASIC PARAMETERS ---
    model_params['zred'] = {'N': 1, 'isfree': False,
                            'init': obs["zred"],
                            'prior': FastUniform(a=1e-3, b=12.0)}
    
    model_params['add_igm_absorption'] = {'N': 1, 'isfree': False,
                                          'init': 1}
    
    model_params['add_agb_dust_model'] = {'N': 1, 'isfree': False,
                                          'init': True}
    
    model_params['pmetals'] = {'N': 1, 'isfree': False,
                               'init': -99}

    model_params['logmass'] = {'N': 1, 'isfree': False,
                              'depends_on': massmet_to_logmass,
                              'init': 10.0,
                              'units': 'Msun',
                              'prior': FastUniform(a=7.0, b=12.5)}

    model_params['logzsol'] = {'N': 1, 'isfree': False,
                               'init': -0.5,
                               'depends_on': massmet_to_logzsol,
                               'units': r'$\log (Z/Z_\odot)$',
                               'prior': FastUniform(a=-1.98, b=0.19)}

    model_params['massmet'] = {'N': 2, 'isfree': True,
                               'init': np.array([10, -0.5]),
                               'prior': FastMassMet(a=6.0, b=12.5)}

    # --- SFH ---
    model_params['sfh'] = {'N': 1, 'isfree': False,
                           'init': 0}
    
    model_params['mass'] = {'N': 1, 'isfree': False,
                            'init': 1.0,
                            'depends_on': logmass_to_masses}
    
    model_params['agebins'] = {'N': 1, 'isfree': False,
                               'init': [],
                               'depends_on': zred_to_agebins}
    
    model_params['logsfr_ratios'] = {'N': 6, 'isfree': True,
                                     'init': np.zeros(nbins_sfh - 1),
                                     'prior': FastTruncatedEvenStudentTFreeDeg2(hw=np.ones(6)*5.0,
                                                                                sig=np.ones(6)*0.3)}
    
    # --- IMF ---
    model_params['imf_type'] = {'N': 1, 'isfree': False,
                                'init': 1}

    # --- Dust Absorption ---
    model_params['dust_type'] = {'N': 1, 'isfree': False,
                                 'init': 4}
    
    model_params['dust1'] = {'N': 1, 'isfree': False,
                             'init': 1.0,
                             'depends_on': to_dust1}
    
    model_params['dust1_fraction'] = {'N': 1, 'isfree': True,
                                      'init': 1.0,
                                      'prior': FastTruncatedNormal(a=0.0, b=2.0, mu=1.0, sig=0.3)}

    model_params['dust2'] = {'N': 1, 'isfree': True,
                             'init': 0.0,
                             'units': '',
                             'prior': FastTruncatedNormal(a=0.0, b=4.0, mu=0.3, sig=1.0)}

    model_params['dust_index'] = {'N': 1, 'isfree': True,
                                  'init': 0.7,
                                  'units': '',
                                  'prior': FastUniform(a=-1.2, b=0.4)}
    
    model_params['dust1_index'] = {'N': 1, 'isfree': False,
                                   'init': -1.0}
    
    model_params['dust_tesc'] = {'N': 1, 'isfree': False,
                                 'init': 7.0}
    
    # --- Dust Emission ---
    model_params['add_dust_emission'] = {'N': 1, 'isfree': False,
                                         'init': 1}
    
    model_params['duste_qpah'] = {'N':1, 'isfree':True,
                                  'init': 2.0,
                                  'prior': FastTruncatedNormal(a=0.0, b=7.0, mu=2.0, sig=2.0)}

    model_params['duste_umin'] = {'N':1, 'isfree':True,
                                  'init': 1.0,
                                  'prior': FastTruncatedNormal(a=0.1, b=25.0, mu=1.0, sig=10.0)}
    
    model_params['duste_gamma'] = {'N': 1, 'isfree': False,
                                   'init': 0.01,
                                   'depends_on': to_duste_gamma}

    model_params['log_duste_gamma'] = {'N':1, 'isfree':True,
                                       'init': -2.0,
                                       'prior': FastTruncatedNormal(a=-4.0, b=0.0, mu=-2.0, sig=1.0)}

    # --- Nebular Emission ---
    model_params['add_neb_emission'] = {'N': 1, 'isfree': False,
                                        'init': True}
    
    model_params['add_neb_continuum'] = {'N': 1, 'isfree': False,
                                         'init': True}
    
    model_params['nebemlineinspec'] = {'N': 1, 'isfree': False,
                                       'init': True}
    
    model_params['gas_logz'] = {'N': 1, 'isfree': True,
                                'init': -0.5,
                                'units': r'log Z/Z_\odot',
                                'prior': FastUniform(a=-2.0, b=0.5)}
    
    model_params['gas_logu'] = {'N': 1, 'isfree': False,
                                'init': -1.0}

    # --- AGN dust ---
    model_params['add_agn_dust'] = {'N': 1, 'isfree': False,
                                    'init': True}
    
    model_params['fagn'] = {'N': 1, 'isfree': False,
                            'init': 0.01,
                            'depends_on': to_fagn}
    
    model_params['agn_tau'] = {'N': 1, 'isfree': False,
                               'init': 10.0,
                               'depends_on': to_agn_tau}
    
    model_params['log_fagn'] = {'N': 1, 'isfree': True,
                                'init': -7.0e-5,
                                'prior': FastUniform(a=-5.0, b=np.log10(3.0))}

    model_params['log_agn_tau'] = {'N': 1, 'isfree': True,
                                   'init': np.log10(20.0),
                                   'prior': FastUniform(a=np.log10(5.0), b=np.log10(150.0))}
    
    # --- Calibration ---
    model_params['phot_jitter'] = {'N': 1, 'isfree': False,
                                   'init': 0.0}
    
    # --- Smoothing ---
    model_params['sigma_smooth'] = {'N': 1, 'isfree': False,
                                    'init': 0.0}

    #---- Units ----
    model_params['peraa'] = {'N': 1, 'isfree': False, 'init': False}

    model_params['mass_units'] = {'N': 1, 'isfree': False, 'init': 'mformed'}

    agebins = zred_to_agebins(
        zred = [model_params['zred']['init']],
        nbins_sfh = nbins_sfh,
    )
    
    model_params['agebins']['N'] = 7
    model_params['agebins']['init'] = agebins
    
    model_params['mass']['N'] = nbins_sfh
    
    model_params['logsfr_ratios']['N'] = nbins_sfh - 1
    model_params['logsfr_ratios']['init'] = np.zeros(nbins_sfh-1)
    
    extra = [k for k in model_params.keys() if k not in fit_order]
    fit_order = fit_order + list(extra)
    
    new_model_params = [model_params[par] for par in fit_order]
    for (i, m) in enumerate(new_model_params):
        new_model_params[i]['name'] = fit_order[i]
        
    model_params = new_model_params
    
    model = SedModel(
        model_params,
        obs=obs,
        sps=sps,
    )

    model.predict = types.MethodType(_predict, model)

    return model

#
# PRIORS
#

massmet_file = os.path.join(os.path.dirname(__file__), "..", "data", "gallazzi_05_massmet.txt")

# A faster uniform distribution. Give it a lower bound `a` and
# an upper bound `b`.
class FastUniform(priors.Prior):

    prior_params = ['a', 'b']

    def __init__(self, a=0.0, b=1.0, parnames=[], name='', ):
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)

        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name

        self.a, self.b = a, b

        if self.b <= self.a:
            raise ValueError('b must be greater than a')

        self.diffthing = b - a
        self.pdfval = 1.0 / (b - a)
        self.logpdfval = np.log(self.pdfval)

    def __len__(self):
        return 1

    def __call__(self, x):
        if not hasattr(x, "__len__"):
            if self.a <= x <= self.b:
                return self.logpdfval
            else:
                return np.NINF
        else:
            return [self.logpdfval if (self.a <= xi <= self.b) else np.NINF for xi in x]

    def scale(self):
        return 0.5 * self.diffthing

    def loc(self):
        return 0.5 * (self.a + self.b)

    def unit_transform(self, x):
        return (x * self.diffthing) + self.a

    def sample(self):
        return self.unit_transform(np.random.rand())


# A faster truncated normal distribution. Give it a lower bound `a`,
# a upper bound `b`, a mean `mu`, and a standard deviation `sig`.
class FastTruncatedNormal(priors.Prior):

    prior_params = ['a', 'b', 'mu', 'sig']

    def __init__(self, a=-1.0, b=1.0, mu=0.0, sig=1.0, parnames=[], name='', ):
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)

        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name

        self.a, self.b, self.mu, self.sig = a, b, mu, sig

        if self.b <= self.a:
            raise ValueError('b must be greater than a')

        self.alpha = (self.a - self.mu) / self.sig
        self.beta = (self.b - self.mu) / self.sig

        self.A = erf(self.alpha / np.sqrt(2.0))
        self.B = erf(self.beta / np.sqrt(2.0))

    def xi(self, x):
        return (x - self.mu) / self.sig

    def phi(self, x):
        return np.sqrt(2.0 / (self.sig**2.0 * np.pi)) * np.exp(-0.5 * self.xi(x)**2.0)

    def __len__(self):
        return 1

    def __call__(self, x):
        # if self.a <= x <= self.b:
        #     return np.log(self.phi(x) / (self.B - self.A))
        # else:
        #     return np.NINF
        if not hasattr(x, "__len__"):
            if self.a <= x <= self.b:
                return np.log(self.phi(x) / (self.B - self.A))
            else:
                np.NINF
        else:
            return [np.log(self.phi(xi) / (self.B - self.A)) if (self.a <= xi <= self.b) else np.NINF for xi in x]

    def scale(self):
        return self.sig

    def loc(self):
        return self.mu

    def unit_transform(self, x):
        return self.sig * np.sqrt(2.0) * erfinv((self.B - self.A) * x + self.A) + self.mu

    def sample(self):
        return self.unit_transform(np.random.rand())


# This is a sort of Student's t-distribution that allows
# for truncation and rescaling, but it requires nu = 2 and mu = 0
# and for the truncation limits to be equidistant from mu. Give it
# the half-width of truncation (i.e. if you want it truncated to the
# domain (-5, 5), give it `hw = 5`) and the rescaled standard
# devation `sig`. Kinda hacky but it works.
class FastTruncatedEvenStudentTFreeDeg2(priors.Prior):

    prior_params = ['hw', 'sig']

    def __init__(self, hw=0.0, sig=1.0, parnames=[], name='', ):
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)

        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name

        self.hw, self.sig = hw, sig

        if np.any(self.hw <= 0.0):
            raise ValueError('hw must be greater than 0.0')

        if np.any(self.sig <= 0.0):
            raise ValueError('sig must be greater than 0.0')

        self.const1 = np.sqrt(1.0 + 0.5*(self.hw**2.0))
        self.const2 = 2.0 * self.sig * self.hw
        self.const3 = self.const2**2.0
        self.const4 = 2.0 * (self.hw**2.0)

    def __len__(self):
        return len(self.hw)

    def __call__(self, x):
        if not hasattr(x, "__len__"):
            if np.abs(x) <= self.hw:
                return np.log(self.const1 / (self.const2 * (1 + 0.5*(x / self.sig)**2.0)**1.5))
            else:
                return np.NINF
        else:
            ret = np.log(self.const1 / (self.const2 * (1 + 0.5*(x / self.sig)**2.0)**1.5))
            bad = np.abs(x) > self.hw
            ret[bad] = np.NINF
            return ret

    def scale(self):
        return self.sig

    def loc(self):
        return 0.0

    def invcdf_numerator(self, x):
        return -1.0 * (self.const3 * x**2.0 - self.const3 * x + (self.sig * self.hw)**2.0)

    def invcdf_denominator(self, x):
        return self.const4 * x**2.0 - self.const4 * x - self.sig**2.0

    def unit_transform(self, x):
        f = (((x > 0.5) & (x <= 1.0)) * np.sqrt(self.invcdf_numerator(x) / self.invcdf_denominator(x)) -
             ((x >= 0.0) & (x <= 0.5)) * np.sqrt(self.invcdf_numerator(x) / self.invcdf_denominator(x)))
        return f

    def sample(self):
        return self.unit_transform(np.random.rand())


# A faster mass-metallicity prior, essentially combining
# the FastUniform and FastTruncatedNormal distributions above.
# Provide it the lower and upper limits for the mass prior (`a`
# and `b`) along with the filepath for the Gallazzi et al. 2005
# data table `fp`.
class FastMassMet(priors.Prior):

    prior_params = ['a', 'b']

    def __init__(self, a=6.0, b=12.5, fp=massmet_file):
        self.mass_dist = FastUniform(a=a, b=b)
        self.massmet = np.loadtxt(fp)
        self.params = {}

    def __len__(self):
        return 2

    def scale(self, mass):
        upper_84 = np.interp(mass, self.massmet[:, 0], self.massmet[:, 3])
        lower_16 = np.interp(mass, self.massmet[:, 0], self.massmet[:, 2])
        return (upper_84-lower_16)

    def loc(self, mass):
        return np.interp(mass, self.massmet[:, 0], self.massmet[:, 1])

    @property
    def range(self):
        return ((self.mass_dist.a, self.mass_dist.b), (-1.98, 0.19))

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        p = np.zeros_like(x)
        if x.shape == (2,):
            p[0] = self.mass_dist(x[0])
            met_dist = FastTruncatedNormal(a=-1.98, b=0.19, mu=self.loc(x[0]), sig=self.scale(x[0]))
            p[1] = met_dist(x[1])
        else:
            p[..., 0] = [self.mass_dist(mass_i) for mass_i in x[..., 0]]
            met_dists = [FastTruncatedNormal(a=-1.98, b=0.19, mu=self.loc(mass_i), sig=self.scale(mass_i)) for mass_i in x[..., 0]]
            p[..., 1] = [met_dists[i](met_i) for (i, met_i) in enumerate(x[..., 1])]
        return p

    def sample(self, nsample=None, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = self.mass_dist.sample()
        met_dist = FastTruncatedNormal(a=-1.98, b=0.19, mu=self.loc(mass), sig=self.scale(mass))
        met = met_dist.sample()

        return np.array([mass, met])

    def unit_transform(self, x, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = self.mass_dist.unit_transform(x[0])
        met_dist = FastTruncatedNormal(a=-1.98, b=0.19, mu=self.loc(mass), sig=self.scale(mass))
        met = met_dist.unit_transform(x[1])

        return np.array([mass, met])
