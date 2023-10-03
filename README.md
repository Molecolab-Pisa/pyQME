<div align="center">
<img src="images/logo.png" alt="logo"></img>
</div>

# PyQME

PyQME is a Package for Open Quantum System Dynamics and spectroscopies simulations in the exciton framework, written in Python 3.

## Disclaimer and Copyright

The terms for using, copying, modifying, or distributing this code are specified in the file `LICENSE`

## Contacts

Piermarco Saraceno (piermarco.saraceno@phd.unipi.it)

Lorenzo Cupellini (lorenzo.cupellini@unipi.it)


Dipartimento di Chimica e Chimica Industriale, 
Via G. Moruzzi 13, I-56124, Pisa (PI), Italy

## Installation

### System requirements

When installing PyQME all the dependencies are included, however in case you have specific requirements for the PyQME installation,
you may want to install PyQME separately, before installing PyQME.

An environment with any version of Python 3 is recommended. You can create it with `conda`, `virtualenv`, or `pyenv`.

For example, with conda run the following code:

```shell
conda create -n PyQME-env python=3.7
```
```shell
conda activate PyQME-env
```

### Installation

To clone the PyQME module, run:

```shell
git clone git@molimen1.dcci.unipi.it:p.saraceno/redfield-package.git
```

This will create the `PyQME` repository.

From the `PyQME` folder, you can install the module with pip:

```shell
pip install .
```

#### Developer mode installation

It is recommended to fork this repository and clone the "forked" one as `origin`. The `upstream` version
can be added by running the following code:

```shell
git remote add upstream git@molimen1.dcci.unipi.it:p.saraceno/redfield-package.git
```

More details on [configuring a remote repository for a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork).

You can install it locally with:

```shell
pip install -e .
```

## Testing

In the `test/` directory there is a simple test to verify the compilation. 
Follow the instructions there and compare the results with the reference ones. 
Note that there will most probably be roundoff errors, which do not impact the quality of the results.

It is recommended to run the tests before pushing your changes to the upstream repository.

## Usage

You may want to look at the examples.

### Units of measure

The frequency and time axes, the exciton Hamiltonian and the EET rates must be in $cm^{-1}$.

The electric transition dipoles must be in Debye (x,y,z components).

The temperature must be in Kelvin.

The absorption and pump-probe spectra are returned in  in ${L}$ · ${cm}^{-1}$ · ${mol}^{-1}$ (molar extinction coefficient).

Some useful conversion factors and physical constants in $cm^{-1}$ can be found in `PyQME/utils.py`.

### Other informations

- The spectral density axis must not contain $0$ $cm^{-1}$.
- The spectral density used as input must not be divided by the frequency axis.
- The spectral density used as input must be multiplied by $\pi$. 
- The time axis used for the lineshape functions and for the spectra calculation must be defined in the spectral density class. If you're not sure about how to set it, use the `get_timeaxis` function in `PyQME/utils.py`.
- Be sure that the Liouvillian of your system is diagonalizable before propagating the density matrix using the "eig" mode.
- The spectra returned are already multiplied by the frequency axis (raised to the appropriate power).

## Notes for Developers

We recommend making changes in a branch of your local version. 
Make sure that your main branch is up to date with the upstream:

```shell
git pull upstream main
```

If you feel your work is completed and want to merge it with the `main` branch of PyQME, you can
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

https://chemrxiv.org/engage/chemrxiv/article-details/6516c4f8a69febde9eeb8ad2