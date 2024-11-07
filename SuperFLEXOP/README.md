# FLEXOP Model for SHARPy

This repository contains the [SHARPy](http://github.com/imperialcollegelondon/sharpy) version of the equivalent FLEXOP model based on the European [FLEXOP project](https://flexop.eu/) with a detailed implementation description included in [1]. 
<p align="center">
<img  src="doc/source/FLEXOP_white.png" alt="Aerodynamic FLEXOP model">
 </p>

## Nonlinear Aeroservoelastic Studies using the (Super)FLEXOP as a Demonstrator Model
- [Enhanced_UVLM_nonlinear_aeroelastic](https://github.com/sduess/Enhanced_UVLM_nonlinear_aeroelastic)
## Installation

Clone the repository to your local computer. It is intended for use within SHARPy so you must ensure that SHARPy is 
properly installed in your system, including the conda environment provided.

With SHARPy and its environment installed, the only required step to use this model is to run from the terminal where
you are running your scripts

```bash
source <path-to-flexop-model>/bin/flexop_vars.sh
```
 
This will append the location of the `flexop-model` folder to the system's path, such that your code will be able to find
the modules.

## Using the model

In your SHARPy model generation script, import the flexop model:

```python
from flexop_model import aircraft
```

The first initialisation step takes the form of

```python
flexop_model = FLEXOP(case_name, case_route, output)
```
followed by for an e.g. aeroelastic simulation

```python
flexop_model.init_aeroelastic(flexop_settings)
```
where the `flexop_settings` refer to various parameter settings regarding discretisation, modelling assumptions, etc. A list of these
settings is provided in the source file.

For usual aeroelastic  simulation, the input files for SHARPy (including structure, aero, and SHARPy h5 files) are generated with

```python
flexop_model.generate()
flexop_model.create_settings(settings)
```

and SHARPy can finally be run with


```python
flexop_model.run()
```

## References

[1] Duessler, S., & Palacios. [Enhanced Unsteady Vortex Lattice Aerodynamics for Nonlinear Flexible Aircraft Dynamic Simulation](https://doi.org/10.2514/1.J063174). AIAA Journal 62(4):1-16, 2023.

## Contact

If you have any questions and want to get in touch, 
[contact us](https://www.imperial.ac.uk/aeroelastics/people/duessler/).

If you have any questions on how to use the model or find any bugs please file an issue. 
