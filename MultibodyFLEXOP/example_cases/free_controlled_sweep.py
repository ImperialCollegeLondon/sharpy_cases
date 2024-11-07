import os
import numpy as np
from create_multibody_flexop import FlexopAeroelastic
import sharpy.sharpy_main

"""
Free case with dynamically swept wingtips
Includes fuselage and tail, sweeping wingtips with a ramp input
Dynamic coupled simulation
"""

# Simulation inputs
case_name = 'free_controlled_sweep'
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

# directions to sweep the wingtips (0 is no movement)
rhs_dir = -1        # -1 is back, 1 is forward
lhs_dir = 1         # -1 is forwards, 1 is back

# maximum angle in ramp input
amp_u = np.deg2rad(30.)
t = np.linspace(dt, physical_time, n_tstep)

time_init_u = 1.
time_end_u = 3.

# if you want to speed it up...
# time_init_u = 0.
# time_end_u = 0.1

i_t_init = np.argmin(np.abs(t - time_init_u))
i_t_end = np.argmin(np.abs(t - time_end_u))

u_ramp_t = np.zeros_like(t)
u_ramp_t[i_t_init:(i_t_end + 1)] = np.linspace(0., amp_u, i_t_end - i_t_init + 1)
u_ramp_t[i_t_end:] = amp_u

u_ramp_t_dot = np.zeros_like(t)
u_ramp_t_dot[i_t_init:(i_t_end + 1)] = amp_u / (time_end_u - time_init_u)

u_rhs = np.zeros((n_tstep, 3))
u_dot_rhs = np.zeros((n_tstep, 3))
u_lhs = np.zeros((n_tstep, 3))
u_dot_lhs = np.zeros((n_tstep, 3))

u_rhs[:, 2] = u_ramp_t * rhs_dir
u_dot_rhs[:, 2] = u_ramp_t_dot * rhs_dir
u_lhs[:, 2] = u_ramp_t * lhs_dir
u_dot_lhs[:, 2] = u_ramp_t_dot * lhs_dir

input_angle_rhs_dir = case_route + f'input_angle_rhs_{case_name}.npy'
input_velocity_rhs_dir = case_route + f'input_velocity_rhs_{case_name}.npy'
input_angle_lhs_dir = case_route + f'input_angle_lhs_{case_name}.npy'
input_velocity_lhs_dir = case_route + f'input_velocity_lhs_{case_name}.npy'

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
    'plot_stride': 4,
    }

# generate model
model = FlexopAeroelastic(case_name, case_route, **settings)

constraint_settings = {'use_control': True,
                       'input_angle_rhs_dir': input_angle_rhs_dir,
                       'input_velocity_rhs_dir': input_velocity_rhs_dir,
                       'input_angle_lhs_dir': input_angle_lhs_dir,
                       'input_velocity_lhs_dir': input_velocity_lhs_dir,
                       'u_rhs': u_rhs,
                       'u_dot_rhs': u_dot_rhs,
                       'u_lhs': u_lhs,
                       'u_dot_lhs': u_dot_lhs}

model.add_constraint('prescribed_hinge', **constraint_settings)

# generate files for simulation
model.generate_h5()
model.generate_settings()

# run simulation
case_data = sharpy.sharpy_main.main(['', case_route + '/' + case_name + '.sharpy'])
