import copy
import numpy as np

# Account for emulator uncertainty in obs["maggies_unc"]. Returns a new obs dictionary.
def modify_obs(obs=None, model=None):
    new_obs = copy.copy(obs)

    # Convolve emulator errors with observational errors
    a_u = 2.5 * np.log10(np.exp(1.0))
    u_0 = 35.0
    sig_u = model.resid[model.sorter, 10]
    xsq = new_obs["maggies"]**2.0
    sig_x = np.abs((-2.0 * sig_u / a_u) * np.exp(-1.0 * u_0 / a_u) * np.sqrt(1.0 + 0.25 * xsq * np.exp(2.0 * u_0 / a_u)))
    new_obs["obs_maggies_unc"] = new_obs["maggies_unc"]
    new_obs["emul_maggies_unc"] = sig_x
    new_obs["maggies_unc"] = np.sqrt(sig_x**2.0 + new_obs["maggies_unc"]**2.0)

    return new_obs

# Account for emulator uncertainty in obs["maggies_unc"]. Returns a new obs dictionary.
def modify_obs_stitched(obs=None, model=None):
    new_obs = copy.copy(obs)

    # Extract the observational errors from the emulator model. To compute this, we take
    # the standard deviation of test set residuals for each filter and each sub-emulator,
    # and then compute the median of these across the sub-emulators.
    stat_i = model.dat["resid_quants_labels"].index("std")
    full_emul_error = model.dat["resid_quants"][model.sorter, stat_i, :]
    sig_u = np.median(full_emul_error, axis=1)

    # Convolve emulator errors with observational errors
    a_u = 2.5 * np.log10(np.exp(1.0))
    u_0 = 35.0
    xsq = new_obs["maggies"]**2.0
    sig_x = np.abs((-2.0 * sig_u / a_u) * np.exp(-1.0 * u_0 / a_u) * np.sqrt(1.0 + 0.25 * xsq * np.exp(2.0 * u_0 / a_u)))
    new_obs["obs_maggies_unc"] = new_obs["maggies_unc"]
    new_obs["emul_maggies_unc"] = sig_x
    new_obs["maggies_unc"] = np.sqrt(sig_x**2.0 + new_obs["maggies_unc"]**2.0)

    return new_obs
