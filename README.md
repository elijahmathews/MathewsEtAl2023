# Mathews et al. (2023)

This repository contains code and data relevant to the Mathews et al. (2023) article published in the Astrophysical Journal.

The `/data` directory contains the emulator used in the paper in addition and documentation for reading the emulator file. In addition, the `/src` directory contains code that allows one to run [Prospector](https://github.com/bd-j/prospector) fits using the emulator in place of [FSPS](https://github.com/cconroy20/fsps). Finally, the `/notebooks` directory contains a notebook demonstrating a sample fit to a mock galaxy using the emulator and Prospector.

The code in this repository does *not* (currently) contain code for generating emulator training sets and training new emulators. To train a new emulator, consider using:
- [Speculator](https://github.com/justinalsing/speculator): A Python library (an extension of the [Tensorflow](https://github.com/tensorflow/tensorflow) machine learning library) containing code useful for training spectral population synthesis emulators. See [Alsing et al. (2020)](https://doi.org/10.3847/1538-4365/ab917f).
- [Parrot.jl](https://github.com/elijahmathews/Parrot.jl): A Julia library (an extension of the [Flux.jl](https://github.com/FluxML/Flux.jl) machine learning library) containing code useful for training spectral population synthesis emulators, including the activation function used in Speculator. This is the library used in the paper.
