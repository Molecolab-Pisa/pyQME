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

redf = RedfieldTensor(H,SD_obj,initialize=True)
relaxation_tensor = redf.get_tensor()
```

The same procedure is valid for the Relaxation tensors in the double exciton manifold, used for pump-probe calculations.

For example,

```
# H is a NxN numpy array
# SD_obj is a SpectralDensity object

redf_double = RedfieldTensorDouble(H,SD_obj)
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

lin_spec_obj = SecularLinearSpectraCalculator(redf)
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

The absorption and pump-probe spectra are returned in optical density units (${L}$ · ${cm}^{-1}$ · ${mol}^{-1}) (i.e. molar extinction coefficient), or lineshape (i.e, units_of_dypole^2, for example Debye^2).

Some useful conversion factors and physical constants in $cm^{-1}$ can be found in `pyQME/utils.py`.

### Known issues

- The frequency and time axes must be sorted in ascending order.
- The spectral density used as input must not be divided by the frequency axis.
- The spectral density used as input must contain the $\pi$ factor. To be sure about it, you can check that SDobj.Reorg corresponds to the expected reorganization energy.
- The time axis used for the lineshape functions and for the spectra calculation must be defined in the spectral density class. If you're not sure about how to set it, use the `get_timeaxis` function in `pyQME/utils.py`, and check that it is long enough to ensure the decay of the bath correlation function SD_obj.get_Ct(). The time step must also be small enough to ensure sufficient sampling of the bath correlation function.
- If you are propagating the density matrix using the "eig" mode of relaxation_tensor.propagate, be sure that the Liouvillian of your system (relaxation_tensor.get_Liouv()) is diagonalizable. For this, you can also compare the density matrix propagated using the "exp" mode with that propagated using the "eig", which is in general faster. Sometimes the 
- If the spectra are calculated in optical density units, the spectra returned have already been multiplied by the frequency axis (raised to the appropriate power). Otherwise, if the lineshape is calculated, you can check that, for absorption spectra, the integral of the spectra returns sum_ix (mu_ix)^2.
- When you use the same Spectral Density object for multiple calculations (for example for repeated spectra calculations along a Molecular Dynamics trajectory), you need to calculate the lineshape function only once, using SDobj._calc_gt(), before passing SDobj to the Relaxation Tensor objects. 

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

## Citing pyQME

When using pyQME, please cite the following paper.

Saraceno, P.; Sl ama, V.; Cupellini, L. The Journal of Chemical Physics 2023,159, 184112.
https://doi.org/10.1063/5.0170295

## Reference Papers

Redfield theory:

REDFIELD, A. In Advances in Magnetic Resonance, Waugh, J. S., Ed.; Advances in Magnetic and Optical Resonance, Vol. 1; Academic Press: 1965, 1–32.
https://doi.org/10.1016/B978-1-4832-3114-3.50007-6

Renger, T.; Marcus, R. A. The Journal of Chemical Physics 2002, 116, 9997–10019.
https://doi.org/10.1063/1.1470200


Förster theory:

F ̈orster, T. Journal of Biomedical Optics 2012, 17, 011002.
https://doi.org/10.1117/1.JBO.17.1.011002


Modified Redfield theory:

Zhang, W. M.; Meier, T.; Chernyak, V.; Mukamel, S. The Journal of Chemical Physics 1998, 108, 7763–7774.
https://doi.org/10.1063/1.476212

Hwang-Fu, Y.-H.; Chen, W.; Cheng, Y.-C. Chemical Physics 2015, 447, 46–53.
https://doi.org/10.1016/j.chemphys.2014.11.026


Redfield-Förster theory:

Yang, M.; Damjanovi ́c, A.; Vaswani, H. M.; Fleming, G. R. Biophysical Journal 2003, 85, 140–158.
https://doi.org/10.1016/S0006-3495(03)74461-0


Linear spectra:

Gelzinis, A.; Abramavicius, D.; Valkunas, L. The Journal of Chemical Physics 2015, 142, 154107.
https://doi.org/10.1063/1.4918343

Cupellini, L.; Lipparini, F.; Cao, J. The Journal of Physical Chemistry B 2020, 124, 8610–8617.
https://doi.org/10.1021/acs.jpcb.0c05180

Nothling, J. A.; Mancal, T.; Kruger, T. P. J. The Journal of Chemical Physics 2022, 157, 095103.
https://doi.org/10.1063/5.0100977

Ma, J., & Cao, J. (2015). Journal of Chemical Physics, 142(9). 
https://doi.org/10.1063/1.4908599

Circular Dichroism:
Jurinovich, S., Cupellini, L., Guido, C. A., & Mennucci, B. (2018). Exat: Excitonic analysis tool. Journal of Computational Chemistry, 39(5), 279–286.
https://doi.org/10.1002/jcc.25118

Pump-probe:

Novoderezhkin, V. I.; Doust, A. B.; Curutchet, C.; Scholes, G. D.; van Grondelle, R. Biophysical Journal 2010, 99, 344–352.
https://doi.org/10.1016/j.bpj.2010.04.039

Renger, T.; Marcus, R. A. The Journal of Chemical Physics 2002, 116, 9997–10019.
https://doi.org/10.1063/1.1470200

Double-exciton manifold:

Saraceno, P.; Slama, V.; Cupellini, L. The Journal of Chemical Physics 2023,159, 184112.
https://doi.org/10.1063/5.0170295