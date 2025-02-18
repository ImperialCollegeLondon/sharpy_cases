import os
import numpy as np
import configobj
import sharpy.sharpy_main
from sharpy.utils import algebra
from pazy_wing_model import PazyWing

output_folder = './output/'
case_route = './cases/'

if not os.path.exists(case_route):
    os.makedirs(case_route)

m = 12                      # chordwise discretisation
n = 20                      # spanwise discretisation
m_star_fact = 2.0           # wake length
gravity_on = False          # gravity
num_surfaces = 2            # number of wings
sigma = 1.0                 # stiffness multiplier
sweep = np.deg2rad(0.0)     # sweep angle
alpha = np.deg2rad(5.0)     # angle of incidence
u_inf = 40.0                # freestream velocity
num_modes = 16              # number of modes
u_inf_dir = [1.0, 0.0, 0.0] # freestream direction
rho = 1.225                 # air density
c_ref = 0.1                 # reference chord
b_ref = 0.55                # reference semispan
dt = c_ref / (u_inf * m)    # time step
skin_on = False             # skin on/off, changing model stiffness


model_settings = {
    'skin_on': skin_on,
    'surface_m': m,
    'num_surfaces': num_surfaces,
    'sweep_angle': sweep,
    'sigma': sigma,
}


settings = dict()
case_name = f"pazy_test_case"

# create pazy wing model as an object
pazy = PazyWing(case_name, case_route, in_settings=model_settings)
pazy.generate_structure()
if num_surfaces == 2:
    pazy.structure.mirror_wing()
pazy.generate_aero()
pazy.save_files()

config = configobj.ConfigObj()
config.filename = f'./{case_name}.sharpy'

settings['SHARPy'] = {
    'flow': [
         'BeamLoader',
         'AerogridLoader',
         'StaticCoupled',
         'BeamPlot',
         'AerogridPlot',
         'Modal',
         'LinearAssembler',
         ],
    'case': case_name,
    'route': case_route,
    'write_screen': True,
    'write_log': 'on',
    'log_folder': output_folder + '/' + case_name + '/intrinsic/',
    'log_file': case_name + '.log'}

settings['BeamLoader'] = {
    'unsteady': True,
    'orientation': algebra.euler2quat((0.0, alpha, 0.0))}

settings['AerogridLoader'] = {
    'unsteady': True,
    'aligned_grid': True,
    'mstar': int(m * m_star_fact),
    'freestream_dir': u_inf_dir,
    'wake_shape_generator': 'StraightWake',
    'wake_shape_generator_input': {'u_inf': u_inf,
                                   'u_inf_direction': u_inf_dir,
                                   'dt': dt}}

settings['Modal'] = {
    'NumLambda': num_modes,
    'rigid_body_modes': False,
    'print_matrices': False,
    'continuous_eigenvalues': False,
    'dt': 0,
    'plot_eigenvalues': False,
    'write_modes_vtk': True,
    'use_undamped_modes': True,
}

settings['BeamPlot'] = {}

settings['AerogridPlot'] = {}

settings['LinearAssembler'] = {
    'linear_system': 'LinearAeroelastic',
    'linearisation_tstep': 0,
    'modal_tstep': 0,
    'inout_coordinates': 'modes',
    'linear_system_settings': {
        'beam_settings': {'modal_projection': True,
                          'inout_coords': 'modes',
                          'discrete_time': True,
                          'newmark_damp': 5e-5,
                          'discr_method': 'newmark',
                          'dt': dt,
                          'proj_modes': 'undamped',
                          'use_euler': False,
                          'num_modes': num_modes,
                          'print_info': True,
                          'gravity': gravity_on,
                          'remove_sym_modes': False,
                          'remove_dofs': []},
        'aero_settings': {'dt': dt,
                          'integr_order': 2,
                          'density': rho,
                          'remove_inputs': ['u_gust'],
                          'remove_predictor': True,
                          'use_sparse': False,
                          'rom_method': ['Krylov'],
                          'rom_method_settings': {'Krylov': {'algorithm': 'mimo_rational_arnoldi',
                                                             'frequency': 0.,
                                                             'single_side': 'observability',
                                                             'r': 6}},
                          }}}

settings['StaticCoupled'] = {
    'relaxation_factor': 0,
    'aero_solver': 'StaticUvlm',
    'aero_solver_settings': {
        'rho': rho,
        'horseshoe': 'off',
        'num_cores': 8,
        'n_rollup': 0,
        'rollup_dt': dt,
        'rollup_aic_refresh': 1,
        'rollup_tolerance': 1e-4,
        'velocity_field_generator': 'SteadyVelocityField',
        'velocity_field_input': {
            'u_inf': u_inf,
            'u_inf_direction': u_inf_dir}},
    'structural_solver': 'NonLinearStatic',
    'structural_solver_settings': {'print_info': 'off',
                                   'max_iterations': 150,
                                   'num_load_steps': 4,
                                   'delta_curved': 1e-1,
                                   'min_delta': 1e-10,
                                   'gravity_on': gravity_on,
                                   'gravity': 9.81}}


for k, v in settings.items():
    config[k] = v
config.write()

# run case
case_data = sharpy.sharpy_main.main(['', case_name + '.sharpy'])
