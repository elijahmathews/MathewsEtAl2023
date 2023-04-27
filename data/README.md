# Data Files

## Emulators

This repository contains two groups of emulators - the "single" emulators (i.e. emulators each consisting of a single neural network) and "stitched" emulators (i.e. emulators each consisting of multiple neural networks). The "single" emulators are the emulators that are trained and used in the Mathews et al. (2023) paper and cover the redshift range $0. 5 < z < 3$, while a stitched emulator is provided here for use with higher (and lower) redshift galaxies with redshift coverage $0 < z < 24$.

While both groups of emulators are saved on disk using the `.npy` format, the data contained within each file is slightly different for each group.

### "Single" Emulators

A total of six "single" emulators are included, corresponding to the six emulators trained in the paper. Each emulator follows the architecture described in the paper (e.g. [GELU](https://doi.org/10.48550/arXiv.1606.08415) activation, 5 layers with nonlinear activation, normalization input parameters, denormalization of output photometry), with six different selections for the number of nodes per layer:

- `parrot_v4_obsphot_32n_5l_05z30.npy`: 32 nodes per layer
- `parrot_v4_obsphot_64n_5l_05z30.npy`: 64 nodes per layer
- `parrot_v4_obsphot_128n_5l_05z30.npy`: 128 nodes per layer
- `parrot_v4_obsphot_256n_5l_05z30.npy`: 256 nodes per layer
- `parrot_v4_obsphot_512n_5l_05z30.npy`: 512 nodes per layer
- `parrot_v4_obsphot_1024n_5l_05z30.npy`: 1024 nodes per layer

Reading each file will result in a NumPy array of length 6:

```python
import numpy as np

fp = "/path/to/data/parrot_v4_obsphot_256n_5l_05z30.npy"

emul = np.load(
    fp,
    allow_pickle = True,
)

emul.shape # -> (6,)
```

Each of these 6 elements contain different information about the emulator. The first is dictionary listing the photometric filters being emulated and which of the emulator's outputs each filter corresponds to (in **1-based indexing** due to this coming from Julia):

```python
filter_info = emul[0]

filter_info["jwst_f200w"] # -> 96, which corresponds to 95 in 0-based indexing
```

The second element is the data for the emulator's normalization layer, structured as a tuple:

```python
norm_mu, norm_sig = emul[1]

norm_mu # -> Array of means to subtract off
norm_sig # -> Array of standard devations to divide by, set to 1 for these emulators
```

The third element is where the guts of the emulator is, as it contains the weights matrices and bias vectors for each layer, of which there are 6 each due to the emulators having 5 nonlinear layers and 1 linear layer:

```python
weights, biases = emul[2]

len(weights) # -> 6
len(biases) # -> 6

weights[0].shape # -> (256, 18)
biases[0].shape # -> (256,)
```

The fourth element is of similar nature to the second element, containing data for the emulator's denormalization layer:

```python
denorm_mu, denorm_sig = emul[3]

denorm_sig # -> Array of standard deviations to multiply by, set to 1 for these emulators
denorm_mu # -> Array of means to add on
```

The fifth element contains a set of statistics, derived from the emulator's performance on the test set, to provide the user with different metrics of how well the emulator performs in each filter. 11 statistics are included, each of which are evaluated on all of the test set residuals (i.e. `emulated - truth`) in each filter, in the following order:

1. The -4σ quantile (i.e. $\frac{1}{2}-\frac{1}{2}\text{erf}\left(\frac{4}{\sqrt{2}}\right) \approx 0.00317\\%$)
2. The -3σ quantile (i.e. $\frac{1}{2}-\frac{1}{2}\text{erf}\left(\frac{3}{\sqrt{2}}\right) \approx 0.135\\%$)
3. The -2σ quantile (i.e. $\frac{1}{2}-\frac{1}{2}\text{erf}\left(\frac{2}{\sqrt{2}}\right) \approx 2.275\\%$)
4. The -1σ quantile (i.e. $\frac{1}{2}-\frac{1}{2}\text{erf}\left(\frac{1}{\sqrt{2}}\right) \approx 15.866\\%$)
5. The median (i.e. $50\\%$)
6. The +1σ quantile (i.e. $\frac{1}{2}+\frac{1}{2}\text{erf}\left(\frac{1}{\sqrt{2}}\right) \approx 84.134\\%$)
7. The +2σ quantile (i.e. $\frac{1}{2}+\frac{1}{2}\text{erf}\left(\frac{2}{\sqrt{2}}\right) \approx 97.725\\%$)
8. The +3σ quantile (i.e. $\frac{1}{2}+\frac{1}{2}\text{erf}\left(\frac{3}{\sqrt{2}}\right) \approx 99.865\\%$)
9. The +4σ quantile (i.e. $\frac{1}{2}+\frac{1}{2}\text{erf}\left(\frac{4}{\sqrt{2}}\right) \approx 99.99683\\%$)
10. The mean
11. The standard deviation

```python
test_set_stats = emul[4]

test_set_stats.shape # -> (137, 11)

test_set_stats[4,:] # -> median residual for each filter
test_set_stats[10,:] # -> standard deviation of residuals for each filter
test_set_stats[9,95] # -> mean residual for jwst_f200w
```

Finally, the sixth element contains information about the range of values for each parameter that the emulator was trained on (e.g. to allow for an error check if the user attempts to evaluate the emulator on a parameter value that lies outside the training set):

```python
parameter_ranges = emul[5]

len(parameter_ranges) # -> 18, due to 18 parameters

parameter_ranges[0] # -> array([ 6. , 12.5]), corresponding to the lower and upper bounds on the logmass training set
```

### "Stitched" Emulators

As described in Section 4.3 of the paper, redshift is difficult parameter to emulate in photometric emulators, which makes training a "single" emulator covering a wide redshift range quite difficult. One solution that we have found to this is to instead train *multiple* emulators, each of which cover smaller redshift ranges that slightly overlap. The overlaps help mitigate discontinuities between emulators, as they allow one to call two emulators within the overlap and then compute a weighted average between them using $\sin^2{\theta(z)}$ and $\cos^2{\theta(z)}$ (see Equations 11 and 12, along with Figure 13, for more information). In total, this allows one to create a "stitched" emulator that covers a wider redshift range more accurately than any single emulator could.

One "stitched" emulator is provided in this repository, `parrot_v4_obsphot_512n_5l_24s_00z24.npy`, and is saved to disk in a different way from the "single" emulators. In this case, all of the data is saved to disk as a single dictionary with 9 entries:

```python
import numpy as np

fp = "/path/to/data/parrot_v4_obsphot_512n_5l_24s_00z24.npy"

emul = np.load(fp, allow_pickle=True).all()

type(emul) # -> <class 'dict'>
len(emul) # -> 9
```

Some of the entries are very equivalent to data stored for the "single" emulators. The `emul["filters"]` entry contains an ordered list of the filters included in the emulator (i.e. the same information as in `parrot_v4_filters.txt`), and `emul["filter_redir"]` contains a dictionary that accomplishes the same goal as the `filter_info` in the "single" emulators (except **0-based indexing is used here**). Meanwhile, the `emul["parameters"]` entry contains an ordered list of the parameters that are included in the emulator's physical model, and `emul["parameter_limits"]` serves an identical role to `parameter_ranges` in the "single" emulators. The `emul["zred_index"]` entry simply stores the index of redshift as this is the parameter that needs special treatment in the stitched emulator:

```python
emul["zred_index"] # -> 14, 0-based indexing
```

## Filters

To train the emulators included in this repository, training and test set data was generated using [FSPS](https://github.com/cconroy20/fsps) (to generate stellar population spectra) and [sedpy](https://github.com/bd-j/sedpy) (to convolve these spectra with filter transmission curves). All emulators share the same list of 137 filters, whose names are listed in the order in which they appear in the emulator's native output in `parrot_v4_filters.txt`. Of these 137 filters, 136 of them use the filter transmission curves that are in sedpy's [default filter repository](https://github.com/bd-j/sedpy/tree/main/sedpy/data/filters). The singular exception is a custom box-shaped filter centered around 1.3 mm, whose filter transmission curve (designed to be compatible with sedpy) is provided in the `1d3mm.par` file.
