import os
import numpy as np
from sedpy import observate

# Filter list
cosmos_filters = {
    "wfc3_ir_f160w",
    "cfht_megacam_us_9301",
    "subaru_suprimecam_B",
    "cfht_megacam_gs_9401",
    "subaru_suprimecam_V",
    "acs_wfc_f606w",
    "cfht_megacam_rs_9601",
    "subaru_suprimecam_rp",
    "cfht_megacam_is_9701",
    "subaru_suprimecam_ip",
    "acs_wfc_f814w",
    "cfht_megacam_zs_9801",
    "subaru_suprimecam_zp",
    "vista_vircam_Y",
    "wfc3_ir_f125w",
    "mayall_newfirm_J1",
    "mayall_newfirm_J2",
    "mayall_newfirm_J3",
    "cfht_wircam_J_8101",
    "vista_vircam_J",
    "wfc3_ir_f140w",
    "mayall_newfirm_H1",
    "mayall_newfirm_H2",
    "cfht_wircam_H_8201",
    "vista_vircam_H",
    "mayall_newfirm_K",
    "cfht_wircam_Ks_8302",
    "vista_vircam_Ks",
    "spitzer_irac_ch1",
    "spitzer_irac_ch2",
    "spitzer_irac_ch3",
    "spitzer_irac_ch4",
    "subaru_suprimecam_ia427",
    "subaru_suprimecam_ia464",
    "subaru_suprimecam_ia484",
    "subaru_suprimecam_ia505",
    "subaru_suprimecam_ia527",
    "subaru_suprimecam_ia574",
    "subaru_suprimecam_ia624",
    "subaru_suprimecam_ia679",
    "subaru_suprimecam_ia709",
    "subaru_suprimecam_ia738",
    "subaru_suprimecam_ia767",
    "subaru_suprimecam_ia827",
    "spitzer_mips_24",
}

datadir = os.path.join(os.path.dirname(__file__), "..", "data")
filter_list_fp = os.path.join(datadir, "parrot_v4_filters.txt")

# Function for loading in the mock observations
def build_obs(objid=None, emulator_filter_fp=filter_list_fp, filter_selection=cosmos_filters, datadir=datadir, **extras):
    ### load the names of all filters in V4 emulator
    with open(emulator_filter_fp) as f:
        lines = [line.strip() for line in f.readlines()]
        
    ### find the index of each filter in V4 emulator list
    ### sort by index (which is increasing effective wavelength)
    filter_idx = np.sort([np.where(np.array(lines)==filt)[0][0] for filt in filter_selection])
    
    ### reselect the filter names to resort them
    new_filters = np.array(lines)[filter_idx]
    
    ### make sure we haven't made a mistake
    for f in new_filters:
        assert f in filter_selection
        
    ### load the observations
    maggies = np.load(
        os.path.join(datadir, "obs_maggies.npy"),
    )
    maggies = maggies[filter_idx,int(objid)-1]
    maggies_unc = np.load(
        os.path.join(datadir, "obs_maggies_unc.npy"),
    )
    maggies_unc = maggies_unc[filter_idx,int(objid)-1]
    
    ### get the redshift
    zred_i = 14
    pars = np.load(os.path.join(datadir, "pars.npy"))
    zred = pars[zred_i,int(objid)-1]

    ### build output dictionary
    obs = {}
    obs['filters'] = observate.load_filters(new_filters)
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']])
    obs['phot_mask'] = np.full_like(maggies, True, dtype=bool)
    obs['maggies'] = maggies
    obs['maggies_unc'] =  maggies_unc
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['logify_spectrum'] = False
    obs['zred'] = zred
    
    # Copy truths
    mask = np.ones(pars.shape[0], dtype=bool)
    mask[zred_i] = 0
    obs['truth'] = pars[mask,int(objid)-1]

    return obs
