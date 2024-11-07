import os
import numpy as np
from create_multibody_flexop import FlexopAeroelastic
import sharpy.sharpy_main

"""
Free case with hinged wingtips
Includes fuselage and tail, with wingtips at 70% span with a flare angle
Dynamic coupled simulation
"""

# Simulation inputs
case_name = 'free_hinged'
case_route = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/cases/'
case_out_folder = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/output/'

try:
    os.makedirs(case_route)
except FileExistsError:
    pass

try:
    os.makedirs(case_out_folder)
except FileExistsError:
    pass

u_inf = 45.
m = 10
m_star_fact = 1.5
physical_time = 1.
c_ref = 0.35
dt = c_ref / (m * u_inf)
n_tstep = int(physical_time / dt)

# these values were found from trim of the regular FLEXOP, will change if the model is updated
# for this case they will not be exact, which would require the 'DynamicTrim' routine to find
alpha = -0.0033
delta = -0.0471
thrust = 3.9980

settings = {
    # flow of solvers to use for simulation
    'flow': ['BeamLoader',
        'AerogridLoader',
        'DynamicCoupled',
        ],

    # true seperates the tips as seperate bodies (3 bodies total), false leaves as one body
    'use_multibody': True,

    # true includes fuselage and tail, false is wing only
    'include_tail': True,

    # true includes static elevators which can be used for trim
    'include_elevators': True,

    # elevator angle
    'elevator_angle': delta,

    # true creates a cambered wing surface to mimic aerofoil effects, false creates a flat surface
    'use_airfoil': True,

    # true includes coupled aero via the StepUvlm, false is structure only
    'use_aero': True,

    # true uses the JAX-based multibody solver 'NonLinearDynamicMultibodyJAX', false uses the tradition solver
    # 'NonLinearDynamicMultibody'. Not all constraint definitions may be available for both solvers.
    'use_jax': True,

    # true enforces the CFL condition = 1, where all wake panels have the same streamwise length. false allows for
    # variable wake discretisation by creating increasing length panels as the flow travels downstream. This is more
    # computationally efficient for a wake of the same length
    'cfl1': False,

    # true sweeps the wing tips in the structure definition, useful for simulating the effect of the sweep when the
    # tips do not need to be dynamic (more convenient for trim etc)
    'use_rigid_sweep': False,

    # number of elements to gradually sweep towards the discontinuity at the hinge to prevent very skewed or overlapping
    # panels. Different numbers can be set for the panels inboard and outboard of the hinge
    'num_elem_warp_main': 2,
    'num_elem_warp_tip': 2,

    # initial orientation angles in radians
    'alpha': alpha,
    'yaw': 0.,
    'roll': 0.,

    # chordwise wing discretisation
    'm_wing': m,

    # structure stiffness multiplier
    'sigma': 1.,

    # time step length annd number (governed by chord length, chord discretisation and velocity for coupled case)
    'dt': dt,
    'n_tstep': n_tstep,

    # flow conditions
    'rho': 1.225,
    'u_inf': u_inf,
    'u_inf_dir': np.array((1., 0., 0.)),

    # wake discretisation as a multiple of the chordwise discretisation
    'm_star_fact': m_star_fact,

    # gust properties
    'gust_intensity': 0.5,
    'gust_length': 10.,

    # use gravity
    'gravity_on': True,

    # if true, set aero free flying aircraft. if false, will set aero for a clamped wing
    'free': True,

    # thrust value to use
    'thrust': thrust,

    # plot stride for creating a beam/aero plot (useful for reducing file count for long simulations)
    'plot_stride': 4
    }

# generate model
model = FlexopAeroelastic(case_name, case_route, **settings)

# model initially has no constraints (wingtips are not attached)
# settings for constraints are defined per constraint. For a free wing with hinges
constraint_settings = {'flare_angle': np.deg2rad(10.)}
model.add_constraint('free_hinge', **constraint_settings)

# generate files for simulation
model.generate_h5()
model.generate_settings()

# run simulation
case_data = sharpy.sharpy_main.main(['', case_route + '/' + case_name + '.sharpy'])
