# Pazy Wing Model for SHARPy

This repository contains the [SHARPy](http://github.com/imperialcollegelondon/sharpy) version of the equivalent beam
model of the Pazy wing described in [1]. The model
has been slightly modified to model the mass and inertia in a distributed fashion per element rather than lumped at the 
finite-element nodes. This is an updated version of the original SHARPy model from 
[HERE](https://github.com/ngoiz/pazy-model), which removes redundant code and fixes some issues, however the model 
properties should otherwise be consistent.

## Using the model

In your SHARPy model generation script, import the Pazy wing:

```python
from pazy_wing_model import PazyWing
```

The initialisation call takes the form of

```python
wing = PazyWing(case_name, case_route, pazy_settings)
```

where the `pazy_settings` refer to whether the skin should be modelled, discretisation refinement etc. A list of these
settings is provided in the source file.

For a purely structural simulation (no aerodynamic grid generated):

```python
wing.generate_structure()
wing.save_files()
```

That will generate the `.fem.h5` file that is used by SHARPy. When accompanied with a `.sharpy` file containing the 
simulation settings, you can run SHARPy. Included in the repository is the `flutter_case.py` script that runs a flutter
analysis as an example.

## References

[1] Riso, C., & Cesnik, C. E. S.. Equivalent Beam Distributions of the Pazy Wing. University of Michigan, 2020.

## Contact

If you have any questions on how to use the model or find any bugs please file an issue. 