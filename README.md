<div align="center">
</div>

# pyQME

pyQME is a Package for Open Quantum System Dynamics and spectroscopies simulations in the exciton framework, written in Python 3.

## Disclaimer and Copyright

The terms for using, copying, modifying, or distributing this code are specified in the file `LICENSE`

## Contacts

Piermarco Saraceno (piermarco.saraceno@phd.unipi.it)

Lorenzo Cupellini (lorenzo.cupellini@unipi.it)


Dipartimento di Chimica e Chimica Industriale, 
Via G. Moruzzi 13, I-56124, Pisa (PI), Italy

## Installation

### System requirements

When installing pyQME all the dependencies are included, however in case you have specific requirements for the pyQME installation,
you may want to install pyQME separately, before installing pyQME.

An environment with any version of Python 3 is recommended. You can create it with `conda`, `virtualenv`, or `pyenv`.

For example, with conda run the following code:

```shell
conda create -n pyQME-env python=3.7
```
```shell
conda activate pyQME-env
```

### Installation

From the `pyQME` folder, you can install the module with pip:

```shell
pip install .
```

#### Developer mode installation

You can install pyQME locally with:

```shell
pip install -e .
```

so that possible code changes take effect immediately on the installed version.

## Usage

You may want to look at the examples. The package doesn't come with a Command Line Interface (CLI), so it must be used via Python (or any hosting platform, such as Jupyter Lab).

### Spectral density

Spectral density are created (generally) from a frequency axis and the spectral density.

For example,

```
# freq_axis is a 1D numpy array of size N
# SD is a 1D numpy array of size N

SD_obj = SpectralDensity(freq_axis,SD,temperature=298)
```

### Relaxation tensors

Relaxation tensors are created (generally) from a Hamiltonian and a spectral density object.

For example, to create a Real Redfield tensor, 

```
# H is a NxN numpy array
# SD_obj is a SpectralDensity object

redf = RedfieldTensorReal(H,SD_obj,initialize=True)
relaxation_tensor = redf.get_tensor()
```

The same procedure is valid for the Relaxation tensors in the double exciton manifold, needed for pump-probe calculations.

For example,

```
# H is a NxN numpy array
# SD_obj is a SpectralDensity object

redf_double = RedfieldTensorRealDouble(H,SD_obj)
```

### Density matrix propagation

Once the Relaxation tensor object has been created, it can be used to propagate a density matrix.

For example,

```
# time_axis_ps is a 1D Numpy array of size T (in picoseconds)
# rho_0 is the density matrix (NxN numpy array) at the beginning of the simulation in the exciton basis

rho_t = redf.propagate(rho0,units='ps',basis='exciton')

# rho_t is the propagated density matrix (TxNxN numpy array) in the exciton basis
```

### Linear spectra

The simulation of absorption and fluorescence spectra takes as input the relaxation tensor object and the transition dipoles.

For example, to simulate an absorption spectrum:

```
# dipoles is a Nx3 numpy array

lin_spec_obj = LinearSpectraCalculator(redf)
freq_axis,OD = lin_spec_obj.calc_OD(dipoles)

# freq_axis and OD are 1D numpy arrays of size W
```

### Pump-probe spectra

The simulation of pump-probe spectra takes as input the relaxation tensors object in the single and double exciton-manifold, the transition dipoles and the populations.

For example:

```
# dipoles is a Nx3 numpy array
# pop_t_exc is a TxN numpy array (diagonal of the density matrix in the exciton basis)

pump_probe_obj = PumpProbeSpectraCalculator(redf,redf_double)
pump_probe_obj.calc_components_lineshape(dipoles=dipoles)
freq_axis,GSB,SE,ESA,PP = spectrum_obj.get_pump_probe(pop_t_exc)

# freq_axis is a numpy array of size W
# GSB,SE,ESA,PP are numpy arrays of shape TxW
```

### Units

The frequency axis, the time axis of the spectral density, the exciton Hamiltonian and the EET rates must be in $cm^{-1}$.

The time axis for the density matrix propagation can be either in cm or in ps (see the "units" argument of the method `/pyQME/tensors/relaxation_tensor/RelTensor.propagate`)

The electric transition dipoles must be in Debye (x,y,z components).

The temperature must be in Kelvin.

The absorption and pump-probe spectra are returned in  in ${L}$ · ${cm}^{-1}$ · ${mol}^{-1}$ (molar extinction coefficient).

Some useful conversion factors and physical constants in $cm^{-1}$ can be found in `pyQME/utils.py`.

### Known issues

- The spectral density axis must not contain $0$ $cm^{-1}$.
- The frequency and time axes must be sorted in ascending order.
- The spectral density used as input must not be divided by the frequency axis.
- The spectral density used as input must be multiplied by $\pi$. 
- The time axis used for the lineshape functions and for the spectra calculation must be defined in the spectral density class. If you're not sure about how to set it, use the `get_timeaxis` function in `pyQME/utils.py`.
- Be sure that the Liouvillian of your system is diagonalizable before propagating the density matrix using the "eig" mode.
- The spectra returned are already multiplied by the frequency axis (raised to the appropriate power).
- When you use the same Spectral Density object for multiple calculations (for example for repeated spectra calculations along a Molecular Dynamics trajectory), you should calculate the lineshape function only one time using SDobj._calc_gt(), before passing SDobj to the Relaxation Tensor objects. 

### Indices convention

The indices convention employed in this code is the following:
- **i,j,k,l**: site basis (single excitations)
- **a,b,c,d**: single-exciton manifold
- **u,v**: site basis (double excitations)
- **q,r,s,t**: double-exciton manifold
- **Z**: spectral density
## Notes for Developers

We recommend making changes in a branch of your local version. 
Make sure that your main branch is up to date with the upstream:

```shell
git pull upstream main
```

If you feel your work is completed and want to merge it with the `main` branch of pyQME, you can
make a merge request and ask for a review of your work.

If, when contributing with some feature, you want to write some unit test for it, we are all super happy. 

## Reference Papers

Redfield theory:

https://doi.org/10.1016/B978-1-4832-3114-3.50007-6
https://doi.org/10.1063/1.1470200


Förster theory:

https://doi.org/10.1117/1.JBO.17.1.011002


Modified Redfield theory:

https://doi.org/10.1063/1.476212
https://doi.org/10.1016/j.chemphys.2014.11.026


Redfield-Förster theory:

https://doi.org/10.1016/S0006-3495(03)74461-0


Linear spectra:

https://doi.org/10.1063/1.4918343


Pump-probe:

https://doi.org/10.1016/j.bpj.2010.04.039
https://doi.org/10.1063/1.1470200

Double-exciton manifold:

https://doi.org/10.1063/5.0170295