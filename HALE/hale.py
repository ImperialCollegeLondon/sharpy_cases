#! /usr/bin/env python3
import h5py as h5
import numpy as np
import os
import sharpy.utils.algebra as algebra
import sharpy.sharpy_main
import sharpy.utils.generate_cases as gc
import sys
index = int(sys.argv[1])
from sharpy.utils.constants import deg2rad

# Problem Set up
def generate_pazy(u_inf, case_name, output_folder='/output/', cases_subfolder='', **kwargs):

    case_name = 'simple_HALE'

    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    
    # SHARPy nonlinear reference solution
    route = route_test_dir + '/cases/' + cases_subfolder + '/' + case_name
    if not os.path.exists(route):
        os.makedirs(route)    
    # route = os.path.dirname(os.path.realpath(__file__)) + '/'

    # EXECUTION
    flow = ['BeamLoader',
            'AerogridLoader',
            # 'NonLinearStatic',
            # 'StaticUvlm',
            'DynamicTrim',
            'DynamicCoupled',
            # 'StaticCoupled',
            'BeamLoads',
            # 'AerogridPlot',
            # 'BeamPlot',
            # 'DynamicCoupled',
            # 'Modal',
            # 'LinearAssember',
            # 'AsymptoticStability',
            ]

    # if free_flight is False, the motion of the centre of the wing is prescribed.
    free_flight = True
    if not free_flight:
        case_name += '_prescribed'
        amplitude = 0 * np.pi / 180
        period = 3
        case_name += '_amp_' + str(amplitude).replace('.', '') + '_period_' + str(period)

    # FLIGHT CONDITIONS
    # the simulation is set such that the aircraft flies at a u_inf velocity while
    # the air is calm.
    u_inf = u_inf
    rho = 1.225

#alpha=alpha, thrus=thrus, eleva=eleva

    # trim sigma = 1.5
    alpha = kwargs.get('alpha', 0.) * np.pi / 180
    beta = 0
    roll = 0
    gravity = 'on'
    cs_deflection = kwargs.get('eleva', 0.) * np.pi / 180
    rudder_static_deflection = 0.0
    rudder_step = 0.0 * np.pi / 180
    thrust = kwargs.get('thrus', 0.)     
    sigma = 1.5
    lambda_dihedral = kwargs.get('flare', 0.) * np.pi / 180

    hinge_deg = kwargs.get('flare', 0.)


    # gust settings
    gust_intensity = 0.5
    gust_length = kwargs.get('gusti', 0.) * 2.5 * u_inf
    gust_offset = 5 * u_inf

    # numerics
    n_step = 5
    structural_relaxation_factor = 0.6
    relaxation_factor = 0.35
    tolerance = 1e-6
    fsi_tolerance = 1e-4

    num_cores = 2

    # MODEL GEOMETRY
    # beam
    span_main = 16.0
    lambda_main = 0.25
    ea_main = 0.3

    ea = 1e7
    ga = 1e5
    gj = 1e4
    eiy = 2e4
    eiz = 4e6
    m_bar_main = 0.75
    j_bar_main = 0.075

    length_fuselage = 10
    offset_fuselage = 0
    sigma_fuselage = 10
    m_bar_fuselage = 0.2
    j_bar_fuselage = 0.08

    span_tail = 2.5
    ea_tail = 0.5
    fin_height = 2.5
    ea_fin = 0.5
    sigma_tail = 100
    m_bar_tail = 0.3
    j_bar_tail = 0.08

    # lumped masses
    n_lumped_mass = 1
    lumped_mass_nodes = np.zeros((n_lumped_mass,), dtype=int)
    lumped_mass = np.zeros((n_lumped_mass,))
    lumped_mass[0] = 50
    lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
    lumped_mass_position = np.zeros((n_lumped_mass, 3))

    # aero
    chord_main = 1.0
    chord_tail = 0.5
    chord_fin = 0.5

    # DISCRETISATION
    # spatial discretisation
    # chordiwse panels
    m = 4
    # spanwise elements
    n_elem_multiplier = 2
    n_elem_main = int(4 * n_elem_multiplier)
    n_elem_tail = int(2 * n_elem_multiplier)
    n_elem_fin = int(2 * n_elem_multiplier)
    n_elem_fuselage = int(2 * n_elem_multiplier)
    n_surfaces = 5

    # temporal discretisation
    physical_time = 60
    tstep_factor = 1.
    dt = 1.0 / m / u_inf * tstep_factor
    n_tstep = round(physical_time / dt)

    # END OF INPUT-----------------------------------------------------------------

    clean_test_files(route, case_name)


    # beam processing
    n_node_elem = 3
    span_main1 = (1.0 - lambda_main) * span_main
    span_main2 = lambda_main * span_main

    n_elem_main1 = round(n_elem_main * (1 - lambda_main))
    n_elem_main2 = n_elem_main - n_elem_main1

    # total number of elements
    n_elem = 0
    n_elem += n_elem_main1 + n_elem_main1
    n_elem += n_elem_main2 + n_elem_main2
    n_elem += n_elem_fuselage
    n_elem += n_elem_fin
    n_elem += n_elem_tail + n_elem_tail

    # number of nodes per part
    n_node_main1 = n_elem_main1 * (n_node_elem - 1) + 1
    n_node_main2 = n_elem_main2 * (n_node_elem - 1) + 1
    n_node_main = n_node_main1 + n_node_main2 - 1
    n_node_fuselage = n_elem_fuselage * (n_node_elem - 1) + 1
    n_node_fin = n_elem_fin * (n_node_elem - 1) + 1
    n_node_tail = n_elem_tail * (n_node_elem - 1) + 1

    # total number of nodes
    n_node = 0
    n_node += n_node_main1 + n_node_main1 - 1
    n_node += n_node_main2 - 1 + n_node_main2 - 1
    n_node += n_node_fuselage - 1
    n_node += n_node_fin - 1
    n_node += n_node_tail - 1
    n_node += n_node_tail - 1
    # n_node += 2

    # stiffness and mass matrices
    n_stiffness = 3
    base_stiffness_main = sigma * np.diag([ea, ga, ga, gj, eiy, eiz])
    base_stiffness_fuselage = base_stiffness_main.copy() * sigma_fuselage
    base_stiffness_fuselage[4, 4] = base_stiffness_fuselage[5, 5]
    base_stiffness_tail = base_stiffness_main.copy() * sigma_tail
    base_stiffness_tail[4, 4] = base_stiffness_tail[5, 5]

    n_mass = 3
    base_mass_main = np.diag([m_bar_main, m_bar_main, m_bar_main, j_bar_main, 0.5 * j_bar_main, 0.5 * j_bar_main])
    base_mass_fuselage = np.diag([m_bar_fuselage,
                                m_bar_fuselage,
                                m_bar_fuselage,
                                j_bar_fuselage,
                                j_bar_fuselage * 0.5,
                                j_bar_fuselage * 0.5])
    base_mass_tail = np.diag([m_bar_tail,
                            m_bar_tail,
                            m_bar_tail,
                            j_bar_tail,
                            j_bar_tail * 0.5,
                            j_bar_tail * 0.5])

    # PLACEHOLDERS
    # beam
    x = np.zeros((n_node,))
    y = np.zeros((n_node,))
    z = np.zeros((n_node,))
    beam_number = np.zeros((n_elem,), dtype=int)
    # body_number = np.zeros((n_elem,), dtype=int)
    frame_of_reference_delta = np.zeros((n_elem, n_node_elem, 3))
    structural_twist = np.zeros((n_elem, 3))
    conn = np.zeros((n_elem, n_node_elem), dtype=int)
    stiffness = np.zeros((n_stiffness, 6, 6))
    elem_stiffness = np.zeros((n_elem,), dtype=int)
    mass = np.zeros((n_mass, 6, 6))
    elem_mass = np.zeros((n_elem,), dtype=int)
    boundary_conditions = np.zeros((n_node,), dtype=int)
    app_forces = np.zeros((n_node, 6))

    # aero
    airfoil_distribution = np.zeros((n_elem, n_node_elem), dtype=int)
    surface_distribution = np.zeros((n_elem,), dtype=int) - 1
    surface_m = np.zeros((n_surfaces,), dtype=int)
    m_distribution = 'uniform'
    aero_node = np.zeros((n_node,), dtype=bool)
    twist = np.zeros((n_elem, n_node_elem))
    sweep = np.zeros((n_elem, n_node_elem))
    chord = np.zeros((n_elem, n_node_elem,))
    elastic_axis = np.zeros((n_elem, n_node_elem,))
    
# def generate_fem():
    stiffness[0, ...] = base_stiffness_main
    stiffness[1, ...] = base_stiffness_fuselage
    stiffness[2, ...] = base_stiffness_tail

    mass[0, ...] = base_mass_main
    mass[1, ...] = base_mass_fuselage
    mass[2, ...] = base_mass_tail

    we = 0
    wn = 0
 
    # inner right wing
    beam_number[we:we + n_elem_main1] = 0
    y[wn:wn + n_node_main1] = np.linspace(0.0, span_main1, n_node_main1)

    for ielem in range(n_elem_main1):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]

    elem_stiffness[we:we + n_elem_main1] = 0
    elem_mass[we:we + n_elem_main1] = 0
    boundary_conditions[0] = 1
    # remember this is in B FoR
    app_forces[1] = [0, thrust, 0, 0, 0, 0]
    # app_forces[1] = [0, 0, 0, 0, 0, 0]

    we += n_elem_main1
    wn += n_node_main1

    # outer right wing
    beam_number[we:we + n_elem_main1] = 0
    y[wn:wn + n_node_main2 - 1] = y[wn - 1] + np.linspace(0.0, np.cos(lambda_dihedral) * span_main2, n_node_main2)[1:]
    z[wn:wn + n_node_main2 - 1] = z[wn - 1] + np.linspace(0.0, np.sin(lambda_dihedral) * span_main2, n_node_main2)[1:]
    for ielem in range(n_elem_main2):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    elem_stiffness[we:we + n_elem_main2] = 0
    elem_mass[we:we + n_elem_main2] = 0
    boundary_conditions[wn + n_node_main2 - 2] = -1
    we += n_elem_main2
    wn += n_node_main2 - 1

    # inner left wing
    beam_number[we:we + n_elem_main1 - 1] = 1
    y[wn:wn + n_node_main1 - 1] = np.linspace(0.0, -span_main1, n_node_main1)[1:]
    for ielem in range(n_elem_main1):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    conn[we, 0] = 0
    elem_stiffness[we:we + n_elem_main1] = 0
    elem_mass[we:we + n_elem_main1] = 0
    app_forces[wn] = [0, -thrust, 0, 0, 0, 0]
    we += n_elem_main1
    wn += n_node_main1 - 1

    # outer left wing
    beam_number[we:we + n_elem_main2] = 1
    y[wn:wn + n_node_main2 - 1] = y[wn - 1] + np.linspace(0.0, -np.cos(lambda_dihedral) * span_main2, n_node_main2)[1:]
    z[wn:wn + n_node_main2 - 1] = z[wn - 1] + np.linspace(0.0, np.sin(lambda_dihedral) * span_main2, n_node_main2)[1:]
    for ielem in range(n_elem_main2):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    elem_stiffness[we:we + n_elem_main2] = 0
    elem_mass[we:we + n_elem_main2] = 0
    boundary_conditions[wn + n_node_main2 - 2] = -1
    we += n_elem_main2
    wn += n_node_main2 - 1

    # fuselage
    beam_number[we:we + n_elem_fuselage] = 2
    x[wn:wn + n_node_fuselage - 1] = np.linspace(0.0, length_fuselage, n_node_fuselage)[1:]
    z[wn:wn + n_node_fuselage - 1] = np.linspace(0.0, offset_fuselage, n_node_fuselage)[1:]
    for ielem in range(n_elem_fuselage):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
    conn[we, 0] = 0
    elem_stiffness[we:we + n_elem_fuselage] = 1
    elem_mass[we:we + n_elem_fuselage] = 1
    we += n_elem_fuselage
    wn += n_node_fuselage - 1
    global end_of_fuselage_node
    end_of_fuselage_node = wn - 1

    # fin
    beam_number[we:we + n_elem_fin] = 3
    x[wn:wn + n_node_fin - 1] = x[end_of_fuselage_node]
    z[wn:wn + n_node_fin - 1] = z[end_of_fuselage_node] + np.linspace(0.0, fin_height, n_node_fin)[1:]
    for ielem in range(n_elem_fin):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    conn[we, 0] = end_of_fuselage_node
    elem_stiffness[we:we + n_elem_fin] = 2
    elem_mass[we:we + n_elem_fin] = 2
    we += n_elem_fin
    wn += n_node_fin - 1
    end_of_fin_node = wn - 1

    # right tail
    beam_number[we:we + n_elem_tail] = 4
    x[wn:wn + n_node_tail - 1] = x[end_of_fin_node]
    y[wn:wn + n_node_tail - 1] = np.linspace(0.0, span_tail, n_node_tail)[1:]
    z[wn:wn + n_node_tail - 1] = z[end_of_fin_node]
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    conn[we, 0] = end_of_fin_node
    elem_stiffness[we:we + n_elem_tail] = 2
    elem_mass[we:we + n_elem_tail] = 2
    boundary_conditions[wn + n_node_tail - 2] = -1
    we += n_elem_tail
    wn += n_node_tail - 1

    # left tail
    beam_number[we:we + n_elem_tail] = 5
    x[wn:wn + n_node_tail - 1] = x[end_of_fin_node]
    y[wn:wn + n_node_tail - 1] = np.linspace(0.0, -span_tail, n_node_tail)[1:]
    z[wn:wn + n_node_tail - 1] = z[end_of_fin_node]
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3,)) * (we + ielem) * (n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    conn[we, 0] = end_of_fin_node
    elem_stiffness[we:we + n_elem_tail] = 2
    elem_mass[we:we + n_elem_tail] = 2
    boundary_conditions[wn + n_node_tail - 2] = -1
    we += n_elem_tail
    wn += n_node_tail - 1



    with h5.File(route + '/' + case_name + '.fem.h5', 'a') as h5file:
        coordinates = h5file.create_dataset('coordinates', data=np.column_stack((x, y, z)))
        conectivities = h5file.create_dataset('connectivities', data=conn)
        num_nodes_elem_handle = h5file.create_dataset(
            'num_node_elem', data=n_node_elem)
        num_nodes_handle = h5file.create_dataset(
            'num_node', data=n_node)
        num_elem_handle = h5file.create_dataset(
            'num_elem', data=n_elem)
        stiffness_db_handle = h5file.create_dataset(
            'stiffness_db', data=stiffness)
        stiffness_handle = h5file.create_dataset(
            'elem_stiffness', data=elem_stiffness)
        mass_db_handle = h5file.create_dataset(
            'mass_db', data=mass)
        mass_handle = h5file.create_dataset(
            'elem_mass', data=elem_mass)
        frame_of_reference_delta_handle = h5file.create_dataset(
            'frame_of_reference_delta', data=frame_of_reference_delta)
        structural_twist_handle = h5file.create_dataset(
            'structural_twist', data=structural_twist)
        bocos_handle = h5file.create_dataset(
            'boundary_conditions', data=boundary_conditions)
        beam_handle = h5file.create_dataset(
            'beam_number', data=beam_number)
        # body_handle = h5file.create_dataset(
        #     'body_number', data=body_number)
        app_forces_handle = h5file.create_dataset(
            'app_forces', data=app_forces)
        lumped_mass_nodes_handle = h5file.create_dataset(
            'lumped_mass_nodes', data=lumped_mass_nodes)
        lumped_mass_handle = h5file.create_dataset(
            'lumped_mass', data=lumped_mass)
        lumped_mass_inertia_handle = h5file.create_dataset(
            'lumped_mass_inertia', data=lumped_mass_inertia)
        lumped_mass_position_handle = h5file.create_dataset(
            'lumped_mass_position', data=lumped_mass_position)
        

# def generate_aero_file():
    # global x, y, z
    # control surfaces
    n_control_surfaces = 2
    control_surface = np.zeros((n_elem, n_node_elem), dtype=int) - 1
    control_surface_type = np.zeros((n_control_surfaces,), dtype=int)
    control_surface_deflection = np.zeros((n_control_surfaces,))
    control_surface_chord = np.zeros((n_control_surfaces,), dtype=int)
    control_surface_hinge_coords = np.zeros((n_control_surfaces,), dtype=float)

    # control surface type 0 = static
    # control surface type 1 = dynamic
    control_surface_type[0] = 2
    control_surface_deflection[0] = cs_deflection
    control_surface_chord[0] = m
    control_surface_hinge_coords[0] = -0.25  # nondimensional wrt elastic axis (+ towards the trailing edge)

    control_surface_type[1] = 0
    control_surface_deflection[1] = rudder_static_deflection
    control_surface_chord[1] = 1
    control_surface_hinge_coords[1] = -0.  # nondimensional wrt elastic axis (+ towards the trailing edge)

    we = 0
    wn = 0
    # right wing (surface 0, beam 0)
    i_surf = 0
    airfoil_distribution[we:we + n_elem_main, :] = 0
    surface_distribution[we:we + n_elem_main] = i_surf
    surface_m[i_surf] = m
    aero_node[wn:wn + n_node_main] = True
    temp_chord = np.linspace(chord_main, chord_main, n_node_main)
    temp_sweep = np.linspace(0.0, 0 * np.pi / 180, n_node_main)
    node_counter = 0
    for i_elem in range(we, we + n_elem_main):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = temp_chord[node_counter]
            elastic_axis[i_elem, i_local_node] = ea_main
            sweep[i_elem, i_local_node] = temp_sweep[node_counter]

    we += n_elem_main
    wn += n_node_main

    # left wing (surface 1, beam 1)
    i_surf = 1
    airfoil_distribution[we:we + n_elem_main, :] = 0
    # airfoil_distribution[wn:wn + n_node_main - 1] = 0
    surface_distribution[we:we + n_elem_main] = i_surf
    surface_m[i_surf] = m
    aero_node[wn:wn + n_node_main - 1] = True
    # chord[wn:wn + num_node_main - 1] = np.linspace(main_chord, main_tip_chord, num_node_main)[1:]
    # chord[wn:wn + num_node_main - 1] = main_chord
    # elastic_axis[wn:wn + num_node_main - 1] = main_ea
    temp_chord = np.linspace(chord_main, chord_main, n_node_main)
    node_counter = 0
    for i_elem in range(we, we + n_elem_main):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = temp_chord[node_counter]
            elastic_axis[i_elem, i_local_node] = ea_main
            sweep[i_elem, i_local_node] = -temp_sweep[node_counter]

    we += n_elem_main
    wn += n_node_main - 1

    we += n_elem_fuselage
    wn += n_node_fuselage - 1 - 1
    #
    # # fin (surface 2, beam 3)
    i_surf = 2
    airfoil_distribution[we:we + n_elem_fin, :] = 1
    # airfoil_distribution[wn:wn + n_node_fin] = 0
    surface_distribution[we:we + n_elem_fin] = i_surf
    surface_m[i_surf] = m
    aero_node[wn:wn + n_node_fin] = True
    # chord[wn:wn + num_node_fin] = fin_chord
    for i_elem in range(we, we + n_elem_fin):
        for i_local_node in range(n_node_elem):
            chord[i_elem, i_local_node] = chord_fin
            elastic_axis[i_elem, i_local_node] = ea_fin
            control_surface[i_elem, i_local_node] = 1
    # twist[end_of_fuselage_node] = 0
    # twist[wn:] = 0
    # elastic_axis[wn:wn + num_node_main] = fin_ea
    we += n_elem_fin
    wn += n_node_fin - 1
    #
    # # # right tail (surface 3, beam 4)
    i_surf = 3
    airfoil_distribution[we:we + n_elem_tail, :] = 2
    # airfoil_distribution[wn:wn + n_node_tail] = 0
    surface_distribution[we:we + n_elem_tail] = i_surf
    surface_m[i_surf] = m
    # XXX not very elegant
    aero_node[wn:] = True
    # chord[wn:wn + num_node_tail] = tail_chord
    # elastic_axis[wn:wn + num_node_main] = tail_ea
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(n_node_elem):
            twist[i_elem, i_local_node] = -0
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(n_node_elem):
            chord[i_elem, i_local_node] = chord_tail
            elastic_axis[i_elem, i_local_node] = ea_tail
            control_surface[i_elem, i_local_node] = 0

    we += n_elem_tail
    wn += n_node_tail
    #
    # # left tail (surface 4, beam 5)
    i_surf = 4
    airfoil_distribution[we:we + n_elem_tail, :] = 2
    # airfoil_distribution[wn:wn + n_node_tail - 1] = 0
    surface_distribution[we:we + n_elem_tail] = i_surf
    surface_m[i_surf] = m
    aero_node[wn:wn + n_node_tail - 1] = True
    # chord[wn:wn + num_node_tail] = tail_chord
    # elastic_axis[wn:wn + num_node_main] = tail_ea
    # twist[we:we + num_elem_tail] = -tail_twist
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(n_node_elem):
            twist[i_elem, i_local_node] = -0
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(n_node_elem):
            chord[i_elem, i_local_node] = chord_tail
            elastic_axis[i_elem, i_local_node] = ea_tail
            control_surface[i_elem, i_local_node] = 0
    we += n_elem_tail
    wn += n_node_tail



    with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
        airfoils_group = h5file.create_group('airfoils')
        # add one airfoil
        naca_airfoil_main = airfoils_group.create_dataset('0', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))
        naca_airfoil_tail = airfoils_group.create_dataset('1', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))
        naca_airfoil_fin = airfoils_group.create_dataset('2', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))

        # chord
        chord_input = h5file.create_dataset('chord', data=chord)
        dim_attr = chord_input.attrs['units'] = 'm'

        # twist
        twist_input = h5file.create_dataset('twist', data=twist)
        dim_attr = twist_input.attrs['units'] = 'rad'

        # sweep
        sweep_input = h5file.create_dataset('sweep', data=sweep)
        dim_attr = sweep_input.attrs['units'] = 'rad'

        # airfoil distribution
        airfoil_distribution_input = h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)

        surface_distribution_input = h5file.create_dataset('surface_distribution', data=surface_distribution)
        surface_m_input = h5file.create_dataset('surface_m', data=surface_m)
        m_distribution_input = h5file.create_dataset('m_distribution', data=m_distribution.encode('ascii', 'ignore'))

        aero_node_input = h5file.create_dataset('aero_node', data=aero_node)
        elastic_axis_input = h5file.create_dataset('elastic_axis', data=elastic_axis)

        control_surface_input = h5file.create_dataset('control_surface', data=control_surface)
        control_surface_deflection_input = h5file.create_dataset('control_surface_deflection',
                                                                data=control_surface_deflection)
        control_surface_chord_input = h5file.create_dataset('control_surface_chord', data=control_surface_chord)
        control_surface_hinge_coords_input = h5file.create_dataset('control_surface_hinge_coords',
                                                                data=control_surface_hinge_coords)
        control_surface_types_input = h5file.create_dataset('control_surface_type', data=control_surface_type)

# # def generate_multibody_file():
# #     global x, y, z, n_node_main1, n_node_main, alpha, hinge_deg
#     ######## MULTIBODY ########

#         # Create the MB and BC files
#     # LC1 = gc.LagrangeConstraint()
#     # LC1.behaviour = 'constant_vel_FoR'
#     # LC1.FoR_body = 0
#     # LC1.vel = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
#     # LC1.scalingFactor = 1e7
# #    LC1.penaltyFactor = 1e-6


#     # LC1 = gc.LagrangeConstraint()
#     # LC1.behaviour = 'spherical_FoR'
#     # # LC2.node_in_body = num_node-1
#     # # LC2.body = 0
#     # LC1.body_FoR = 0
#     # # LC2.rot_vect = np.array([0., 0.05, 0.])
#     # # LC2.rel_posB = np.zeros((3))
#     # # LC2.rot_axisB = np.array([1.0,0.0,0.0])
#     # LC1.scalingFactor = 1e8

#     LC2 = gc.LagrangeConstraint()
#     # LC2.behaviour = 'hinge_node_FoR_constant_vel'
#     # LC2.node_in_body = num_node-1
#     # LC2.body = 0
#     # LC2.body_FoR = 1
#     # LC2.rot_vect = np.array([0., 0.05, 0.])
#     # LC2.rel_posB = np.zeros((3))
#     # # LC2.rot_axisB = np.array([1.0,0.0,0.0])
#     # LC2.scalingFactor = 1e8

#     LC2.behaviour = 'hinge_node_FoR'
#     LC2.node_in_body = n_node_main1-1
#     LC2.body = 0
#     LC2.body_FoR = 1
#     # LC2.rot_axisB = np.array([1.0,0.0,0.0])
#     LC2.scalingFactor = 1e8
# #   LC2.penaltyFactor = 1e-12
#     # LC2.rot_axisB = algebra.rotation3d_y(aoa_ini_deg*deg2rad).dot(np.array([np.cos(hinge_deg*deg2rad),-np.sin(hinge_deg*deg2rad),0.0]))
#     LC2.rot_axisA2 = algebra.multiply_matrices(algebra.rotation3d_z(90*deg2rad),np.array([-np.sin(hinge_deg*deg2rad),-np.cos(hinge_deg*deg2rad),0.0]))
#     # algebra.rotation3d_y((aoa_ini_deg+180)*deg2rad)*algebra.rotation3d_z(90*deg2rad)
#     LC2.rot_axisB = (np.array([-np.sin(hinge_deg*deg2rad),-np.cos(hinge_deg*deg2rad),0.0]))
#     print(LC2.rot_axisA2)
#     # import pdb
#     # pdb.set_trace()

#         # Create the MB and BC files
#     # LC3 = gc.LagrangeConstraint()
#     # LC3.behaviour = 'constant_vel_FoR'
#     # LC3.FoR_body = 2
#     # LC3.vel = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
#     # LC3.scalingFactor = 1e7
# #    LC1.penaltyFactor = 1e-6


#     # LC1 = gc.LagrangeConstraint()
#     # LC1.behaviour = 'spherical_FoR'
#     # # LC2.node_in_body = num_node-1
#     # # LC2.body = 0
#     # LC1.body_FoR = 0
#     # # LC2.rot_vect = np.array([0., 0.05, 0.])
#     # # LC2.rel_posB = np.zeros((3))
#     # # LC2.rot_axisB = np.array([1.0,0.0,0.0])
#     # LC1.scalingFactor = 1e8

#     LC4 = gc.LagrangeConstraint()
#     # LC2.behaviour = 'hinge_node_FoR_constant_vel'
#     # LC2.node_in_body = num_node-1
#     # LC2.body = 0
#     # LC2.body_FoR = 1
#     # LC2.rot_vect = np.array([0., 0.05, 0.])
#     # LC2.rel_posB = np.zeros((3))
#     # # LC2.rot_axisB = np.array([1.0,0.0,0.0])
#     # LC2.scalingFactor = 1e8

#     LC4.behaviour = 'hinge_node_FoR'
#     #### NOTE nody_in_body is relative to body node sequence rather than global node sequenece! hence 24 here rather than 29
#     LC4.node_in_body = (n_node_main1-1)*2
#     LC4.body = 0
#     LC4.body_FoR = 2
#     # LC2.rot_axisB = np.array([1.0,0.0,0.0])
#     LC4.scalingFactor = 1e8
# #   LC2.penaltyFactor = 1e-12
#     # LC2.rot_axisB = algebra.rotation3d_y(aoa_ini_deg*deg2rad).dot(np.array([np.cos(hinge_deg*deg2rad),-np.sin(hinge_deg*deg2rad),0.0]))
#     LC4.rot_axisA2 = algebra.multiply_matrices(algebra.rotation3d_z(-90*deg2rad),np.array([-np.sin(hinge_deg*deg2rad),np.cos(hinge_deg*deg2rad),0.0]))
#     # algebra.rotation3d_y((aoa_ini_deg+180)*deg2rad)*algebra.rotation3d_z(90*deg2rad)
#     LC4.rot_axisB = (np.array([-np.sin(hinge_deg*deg2rad),np.cos(hinge_deg*deg2rad),0.0]))
#     print(LC4.rot_axisA2)
#     # import pdb
#     # pdb.set_trace()
    
#     LC = []
#     # LC.append(LC1)
#     LC.append(LC2)
#     # LC.append(LC3)
#     LC.append(LC4)

#     MB1 = gc.BodyInformation()
#     MB1.body_number = 0
#     MB1.FoR_position = np.zeros((6,),)
#     MB1.FoR_velocity = np.zeros((6,),)
#     MB1.FoR_acceleration = np.zeros((6,),)
#     MB1.FoR_movement = 'free'
#     MB1.quat = algebra.euler2quat([0, alpha , 0])

#     MB2 = gc.BodyInformation()
#     MB2.body_number = 1
#     MB2.FoR_position = np.array([x[n_node_main1], y[n_node_main1], z[n_node_main1], 0.0, 0.0, 0.0])
#     # print(MB2.FoR_position)
#     MB2.FoR_velocity = np.zeros((6,),)
#     MB2.FoR_acceleration = np.zeros((6,),)
#     MB2.FoR_movement = 'free'
#     MB2.quat = algebra.euler2quat([0, alpha , 0])

#     # MB3 = gc.BodyInformation()
#     # MB3.body_number = 0
#     # MB3.FoR_position = np.zeros((6,),)
#     # MB3.FoR_velocity = np.zeros((6,),)
#     # MB3.FoR_acceleration = np.zeros((6,),)
#     # MB3.FoR_movement = 'prescribed'
#     # MB3.quat = algebra.euler2quat([0, aoa_ini_deg * np.pi/180, 0])

#     MB4 = gc.BodyInformation()
#     MB4.body_number = 2
#     MB4.FoR_position = np.array([x[n_node_main+n_node_main1], y[n_node_main+n_node_main1], z[n_node_main+n_node_main1], 0.0, 0.0, 0.0])
#     # print(MB4.FoR_position)
#     MB4.FoR_velocity = np.zeros((6,),)
#     MB4.FoR_acceleration = np.zeros((6,),)
#     MB4.FoR_movement = 'free'
#     MB4.quat = algebra.euler2quat([0, alpha , 0])


#     MB = []
#     MB.append(MB1)
#     MB.append(MB2)
#     # MB.append(MB3)
#     MB.append(MB4)


#     gc.generate_multibody_file(LC, MB, route, case_name)




# def generate_dyn_file():
#     global dt
#     global n_tstep
#     global route
#     global case_name
#     global num_elem
#     global num_node_elem
#     global num_node
#     global amplitude
#     global period
#     global free_flight

    dynamic_forces_time = None
    with_dynamic_forces = False
    with_forced_vel = False
    if not free_flight:
        with_forced_vel = True

    if with_dynamic_forces:
        f1 = 100
        dynamic_forces = np.zeros((num_node, 6))
        app_node = [int(num_node_main - 1), int(num_node_main)]
        dynamic_forces[app_node, 2] = f1
        force_time = np.zeros((n_tstep,))
        limit = round(0.05 / dt)
        force_time[50:61] = 1

        dynamic_forces_time = np.zeros((n_tstep, num_node, 6))
        for it in range(n_tstep):
            dynamic_forces_time[it, :, :] = force_time[it] * dynamic_forces

    forced_for_vel = None
    if with_forced_vel:
        forced_for_vel = np.zeros((n_tstep, 6))
        forced_for_acc = np.zeros((n_tstep, 6))
        for it in range(n_tstep):
            # if dt*it < period:
            # forced_for_vel[it, 2] = 2*np.pi/period*amplitude*np.sin(2*np.pi*dt*it/period)
            # forced_for_acc[it, 2] = (2*np.pi/period)**2*amplitude*np.cos(2*np.pi*dt*it/period)

            forced_for_vel[it, 3] = 2 * np.pi / period * amplitude * np.sin(2 * np.pi * dt * it / period)
            forced_for_acc[it, 3] = (2 * np.pi / period) ** 2 * amplitude * np.cos(2 * np.pi * dt * it / period)

    if with_dynamic_forces or with_forced_vel:
        with h5.File(route + '/' + case_name + '.dyn.h5', 'a') as h5file:
            if with_dynamic_forces:
                h5file.create_dataset(
                    'dynamic_forces', data=dynamic_forces_time)
            if with_forced_vel:
                h5file.create_dataset(
                    'for_vel', data=forced_for_vel)
                h5file.create_dataset(
                    'for_acc', data=forced_for_acc)
            h5file.create_dataset(
                'num_steps', data=n_tstep)




# def generate_solver_file():
    file_name = route + '/' + case_name + '.sharpy'
    settings = dict()
    settings['SHARPy'] = {'case': case_name,
                        'route': route,
                        'flow': flow,
                        'write_screen': 'on',
                        'write_log': 'on',
                        'log_folder': route + '/output/',
                        'log_file': case_name + '.log'}

    settings['BeamLoader'] = {'unsteady': 'on',
                            'orientation': algebra.euler2quat(np.array([roll,
                                                                        alpha,
                                                                        beta]))}
    settings['AerogridLoader'] = {'unsteady': 'on',
                                'aligned_grid': 'off',
                                'mstar': int(60 / tstep_factor),
                                'freestream_dir': ['1', '0', '0'],
                                'wake_shape_generator': 'StraightWake',
                                'wake_shape_generator_input': {'u_inf': u_inf,
                                                                'u_inf_direction': ['1', '0', '0'],
                                                                'dt': dt}}

    settings['NonLinearStatic'] = {'print_info': 'off',
                                'max_iterations': 150,
                                'num_load_steps': 1,
                                'delta_curved': 1e-1,
                                'min_delta': tolerance,
                                'gravity_on': gravity,
                                'gravity': 9.81}

    settings['StaticUvlm'] = {'print_info': 'on',
                            'horseshoe': 'off',
                            'num_cores': num_cores,
                            'n_rollup': 0,
                            'rollup_dt': dt,
                            'rollup_aic_refresh': 1,
                            'rollup_tolerance': 1e-4,
                            'velocity_field_generator': 'SteadyVelocityField',
                            'velocity_field_input': {'u_inf': u_inf,
                                                    'u_inf_direction': [1., 0, 0]},
                            'rho': rho}

    settings['StaticCoupled'] = {'print_info': 'off',
                                'structural_solver': 'NonLinearStatic',
                                'structural_solver_settings': settings['NonLinearStatic'],
                                'aero_solver': 'StaticUvlm',
                                'aero_solver_settings': settings['StaticUvlm'],
                                'max_iter': 100,
                                'n_load_steps': n_step,
                                'tolerance': fsi_tolerance,
                                'relaxation_factor': structural_relaxation_factor}



    settings['NonLinearDynamicCoupledStep'] = {'print_info': 'off',
                                            'max_iterations': 950,
                                            'delta_curved': 1e-1,
                                            'min_delta': tolerance,
                                            'newmark_damp': 5e-3,
                                            'gravity_on': gravity,
                                            'gravity': 9.81,
                                            'num_steps': n_tstep,
                                            'dt': dt,
                                            'initial_velocity': u_inf}

    settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'off',
                                                'max_iterations': 950,
                                                'delta_curved': 1e-1,
                                                'min_delta': tolerance,
                                                'newmark_damp': 5e-3,
                                                'gravity_on': gravity,
                                                'gravity': 9.81,
                                                'num_steps': n_tstep,
                                                'dt': dt,
                                                'initial_velocity': u_inf * int(free_flight)}

    # settings['NonLinearDynamicMultibody'] = {'min_delta': 1e-3,
    #                                             'max_iterations': 199,
    #                                             'abs_threshold': 1e-6,
    #                                             'time_integrator': 'NewmarkBeta',
    #                                             'time_integrator_settings': {'newmark_damp': 0.0001,
    #                                                                             'dt': dt},
    #                                             'gravity_on': gravity,
    #                                             'gravity': 9.81,
    #                                             'num_steps': n_tstep,
    #                                             'write_lm': True,
    #                                             'initial_velocity': u_inf * int(free_flight)}                                                  





    relative_motion = 'off'
    if not free_flight:
        relative_motion = 'on'
    settings['StepUvlm'] = {'print_info': 'off',
                            'num_cores': num_cores,
                            'convection_scheme': 3,
                            'gamma_dot_filtering': 6,
                            'velocity_field_generator': 'GustVelocityField',
                            'velocity_field_input': {'u_inf': int(not free_flight) * u_inf,
                                                    'u_inf_direction': [1., 0, 0],
                                                    'gust_shape': '1-cos',
                                                    'gust_parameters': {'gust_length': gust_length,
                                                                        'gust_intensity': gust_intensity * u_inf},
                                                    'offset': gust_offset,
                                                    'relative_motion': relative_motion},
                            'rho': rho,
                            'n_time_steps': n_tstep,
                            'dt': dt}

    if free_flight:
        solver = 'NonLinearDynamicCoupledStep'
    else:
        solver = 'NonLinearDynamicPrescribedStep'

    gains = -np.array([0.9, 6.0, 0.75])

    pitch_file = route + 'pitch.csv'

    alpha_hist = np.linspace(0, n_tstep*dt, n_tstep)
    alpha_hist = 0.0/180.0*np.pi*alpha_hist
    np.savetxt(pitch_file, alpha_hist)

    settings['DynamicCoupled'] = {'structural_solver': solver,
                                'structural_solver_settings': settings[solver],
                                'aero_solver': 'StepUvlm',
                                'aero_solver_settings': settings['StepUvlm'],
                                'fsi_substeps': 200,
                                'fsi_tolerance': fsi_tolerance,
                                'relaxation_factor': relaxation_factor,
                                'minimum_steps': 1,
                                'relaxation_steps': 150,
                                'final_relaxation_factor': 0.5,
                                'n_time_steps': n_tstep,
                                'dt': dt,
                                'include_unsteady_force_contribution': 'on',
                                #   'controller_id': {'controller_elevator': 'ControlSurfacePidController'},
                                #   'controller_settings': {'controller_elevator': {'P': gains[0],
                                #                                             'I': gains[1],
                                #                                             'D': gains[2],
                                #                                             'dt': dt,
                                #                                             'input_type': 'pitch',
                                #                                             'write_controller_log': True,
                                #                                             'controlled_surfaces': [0],
                                #                                             'time_history_input_file': 'pitch.csv'}},
                                'postprocessors': ['BeamLoads', 'BeamPlot', 'AerogridPlot', 'WriteVariablesTime'],
                                # 'postprocessors': ['BeamLoads', 'BeamPlot', 'AerogridPlot'],
                                'postprocessors_settings': {'BeamLoads': {'csv_output': 'on'},
                                                            'BeamPlot': {'include_rbm': 'on',
                                                                        'include_applied_forces': 'on',
                                                                        'stride': 2},
                                                            'AerogridPlot': {
                                                                'include_rbm': 'on',
                                                                'include_applied_forces': 'on',
                                                                'minus_m_star': 0,
                                                                'stride': 2},
                                                            'WriteVariablesTime' : {
                                                                'structure_variables': ['pos','pos_dot', 'pos_ddot', 'psi','psi_dot', 'psi_ddot','psi_local','psi_dot_local','q', 'dqdt','dqddt'],
                                                                # 'FoR_variables': ['mb_quat','mb_dquatdt','in_global_AFoR','mb_FoR_vel', 'mb_FoR_acc'] ,
                                                                # 'FoR_number': np.array([0,1,2], dtype=int),
                                                                'structure_nodes': list(range(0, 64))
                                                            }
                                                            }}


    settings['DynamicTrim'] = {'solver': 'DynamicCoupled',
                            'solver_settings': settings['DynamicCoupled'],
                            'initial_alpha': alpha,
                            'initial_deflection': cs_deflection,
                            'save_info': True,
                            'initial_thrust': thrust,
                            'fz_tolerance': 0.05,
                            'm_tolerance':0.05,
                            'fx_tolerance':0.05,
                            'thrust_nodes':[1,17],
                            'notrim_relax_iter': 1600}


    settings['BeamLoads'] = {'csv_output': 'on'}

    # settings['BeamPlot'] = {'include_rbm': 'on',
    #                         'include_applied_forces': 'on'}


    # settings['AerogridPlot'] = {'include_rbm': 'on',
    #                             'include_forward_motion': 'off',
    #                             'include_applied_forces': 'on',
    #                             'minus_m_star': 0,
    #                             'u_inf': u_inf,
    #                             'dt': dt}

    settings['Modal'] = {'print_info': True,
                        'use_undamped_modes': True,
                        'NumLambda': 30,
                        'rigid_body_modes': True,
                        'write_modes_vtk': 'on',
                        'print_matrices': 'on',
                        'continuous_eigenvalues': 'off',
                        'dt': dt,
                        'plot_eigenvalues': False}

    settings['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                                'linear_system_settings': {
                                    'beam_settings': {'modal_projection': False,
                                                        'inout_coords': 'nodes',
                                                        'discrete_time': True,
                                                        'newmark_damp': 0.05,
                                                        'discr_method': 'newmark',
                                                        'dt': dt,
                                                        'proj_modes': 'undamped',
                                                        'use_euler': 'off',
                                                        'num_modes': 40,
                                                        'print_info': 'on',
                                                        'gravity': 'on',
                                                        'remove_dofs': []},
                                    'aero_settings': {'dt': dt,
                                                        'integr_order': 2,
                                                        'density': rho,
                                                        'remove_predictor': False,
                                                        'use_sparse': True,
                                                        'remove_inputs': ['u_gust']}
                                }}

    settings['AsymptoticStability'] = {'print_info': 'on',
                                    'modes_to_plot': [],
                                    'display_root_locus': 'off',
                                    'frequency_cutoff': 0,
                                    'export_eigenvalues': 'off',
                                    'num_evals': 40}

    import configobj
    config = configobj.ConfigObj()
    config.filename = file_name
    for k, v in settings.items():
        config[k] = v
    config.write()

    # generate_fem()
    # generate_aero_file()
    # generate_multibody_file()
    # generate_solver_file()
    # generate_dyn_file()

    print('Running {}'.format(route + '/' + case_name + '.sharpy'))
    case_data = sharpy.sharpy_main.main(['', route + '/' + case_name + '.sharpy'])    

    # extract information
    n_tsteps = len(case_data.structure.timestep_info)
    dt = case_data.settings['DynamicCoupled']['dt']
    time_vec = np.linspace(0, n_tsteps*dt, n_tsteps)
    loads = np.zeros((n_tsteps, 3))
    for it in range(n_tsteps):
        loads[it, 0:3] = case_data.structure.timestep_info[it].postproc_cell['loads'][0, 3:6]

    route_export = route_test_dir + '/output'
    if not os.path.exists(route_export):
        os.makedirs(route_export)  
    dest_file = route_export + '/root_gusti{:04g}_flare{:04g}.txt'.format(gust_length * 100, hinge_deg * 100)
    np.savetxt(dest_file, np.column_stack(loads))
    print('Saved root array to {}'.format(dest_file))

    # extract information rootxyz
    n_tsteps = len(case_data.structure.timestep_info)
    root = np.zeros((n_tsteps, 3))
    for it in range(n_tsteps):
        pos = case_data.structure.timestep_info[it].for_pos[0:3]
        root[it] = pos

    route_export = route_test_dir + '/output'
    if not os.path.exists(route_export):
        os.makedirs(route_export)  
    dest_file = route_export + '/rootxyz_gusti{:04g}_flare{:04g}.txt'.format(gust_length * 100, hinge_deg * 100)
    np.savetxt(dest_file, root)
    print('Saved rootxyz array to {}'.format(dest_file))

    # extract information dih1xyz
    n_tsteps = len(case_data.structure.timestep_info)
    dih1 = np.zeros((n_tsteps, 3))
    for it in range(n_tsteps):
        pos = case_data.structure.timestep_info[it].for_pos[0:3] + np.dot(algebra.quat2rotation(case_data.structure.timestep_info[it].quat), case_data.structure.timestep_info[it].pos[12, 0:3])
        dih1[it] = pos

    route_export = route_test_dir + '/output'
    if not os.path.exists(route_export):
        os.makedirs(route_export)  
    dest_file = route_export + '/dih1xyz_gusti{:04g}_flare{:04g}.txt'.format(gust_length * 100, hinge_deg * 100)
    np.savetxt(dest_file, dih1)
    print('Saved dih1xyz array to {}'.format(dest_file))
 
    # extract information dih2xyz
    n_tsteps = len(case_data.structure.timestep_info)
    dih2 = np.zeros((n_tsteps, 3))
    for it in range(n_tsteps):
        pos = case_data.structure.timestep_info[it].for_pos[0:3] + np.dot(algebra.quat2rotation(case_data.structure.timestep_info[it].quat), case_data.structure.timestep_info[it].pos[28, 0:3])
        dih2[it] = pos

    route_export = route_test_dir + '/output'
    if not os.path.exists(route_export):
        os.makedirs(route_export)  
    dest_file = route_export + '/dih2xyz_gusti{:04g}_flare{:04g}.txt'.format(gust_length * 100, hinge_deg * 100)
    np.savetxt(dest_file, dih2)
    print('Saved dih2xyz array to {}'.format(dest_file))   

    # extract information tip1xyz
    n_tsteps = len(case_data.structure.timestep_info)
    tip1 = np.zeros((n_tsteps, 3))
    for it in range(n_tsteps):
        pos = case_data.structure.timestep_info[it].for_pos[0:3] + np.dot(algebra.quat2rotation(case_data.structure.timestep_info[it].quat), case_data.structure.timestep_info[it].pos[16, 0:3])
        tip1[it] = pos

    route_export = route_test_dir + '/output'
    if not os.path.exists(route_export):
        os.makedirs(route_export)  
    dest_file = route_export + '/tip1xyz_gusti{:04g}_flare{:04g}.txt'.format(gust_length * 100, hinge_deg * 100)
    np.savetxt(dest_file, tip1)
    print('Saved tip1xyz array to {}'.format(dest_file))
 
    # extract information tip2xyz
    n_tsteps = len(case_data.structure.timestep_info)
    tip2 = np.zeros((n_tsteps, 3))
    for it in range(n_tsteps):
        pos = case_data.structure.timestep_info[it].for_pos[0:3] + np.dot(algebra.quat2rotation(case_data.structure.timestep_info[it].quat), case_data.structure.timestep_info[it].pos[32, 0:3])
        tip2[it] = pos

    route_export = route_test_dir + '/output'
    if not os.path.exists(route_export):
        os.makedirs(route_export)  
    dest_file = route_export + '/tip2xyz_gusti{:04g}_flare{:04g}.txt'.format(gust_length * 100, hinge_deg * 100)
    np.savetxt(dest_file, tip2)
    print('Saved tip2xyz array to {}'.format(dest_file))   






    # FUNCTIONS-------------------------------------------------------------
def clean_test_files(route, case_name):
    fem_file_name = route + '/' + case_name + '.fem.h5'
    if os.path.isfile(fem_file_name):
        os.remove(fem_file_name)

    dyn_file_name = route + '/' + case_name + '.dyn.h5'
    if os.path.isfile(dyn_file_name):
        os.remove(dyn_file_name)

    aero_file_name = route + '/' + case_name + '.aero.h5'
    if os.path.isfile(aero_file_name):
        os.remove(aero_file_name)

    # mb_file_name = route + '/' + case_name + '.mb.h5'
    # if os.path.isfile(mb_file_name):
    #     os.remove(mb_file_name)    

    solver_file_name = route + '/' + case_name + '.sharpy'
    if os.path.isfile(solver_file_name):
        os.remove(solver_file_name)

    flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
    if os.path.isfile(flightcon_file_name):
        os.remove(flightcon_file_name)
def generate_naca_camber(M=0, P=0):
    mm = M * 1e-2
    p = P * 1e-1

    def naca(x, mm, p):
        if x < 1e-6:
            return 0.0
        elif x < p:
            return mm / (p * p) * (2 * p * x - x * x)
        elif x > p and x < 1 + 1e-6:
            return mm / ((1 - p) * (1 - p)) * (1 - 2 * p + 2 * p * x - x * x)

    x_vec = np.linspace(0, 1, 1000)
    y_vec = np.array([naca(x, mm, p) for x in x_vec])
    return x_vec, y_vec


if __name__ == '__main__':
    from datetime import datetime

    # u_inf_vec = np.linspace(2, 40, 191)
    # u_inf_vec = np.linspace(2,3,2)
    u_inf_vec = [10]


    # alpha = 4.7393 * np.pi / 180
    # beta = 0
    # roll = 0
    # gravity = 'on' / 
    # cs_deflection = -2.3502 * np.pi / 180
    # rudder_static_deflection = 0.0
    # rudder_step = 0.0 * np.pi / 180
    # thrust = 3.1626

    gusti_list = np.linspace(0.0, 2.0, 21) 
    flare_list = np.linspace(0.0, 90.0, 10)

    gusti = gusti_list[(index-1)%int(gusti_list.size)]
    flare = flare_list[(index-1)//int(gusti_list.size)]    
    
    alpha_list = np.linspace(4.32, 4.32, 1) 
    thrus_list = np.linspace(3.36, 3.36, 1)

    # thrus_list = np.linspace(3.1626, 3.1626, 1)
    eleva_list = np.linspace(-2.08, -2.08, 1)  
    # alpha = alpha_list[index-1]
    # alpha = alpha_list[((index-1)%(int(alpha_list.size)*int(thrus_list.size)))%int(alpha_list.size)]
    # thrus = thrus_list[((index-1)%(int(alpha_list.size)*int(thrus_list.size)))//int(alpha_list.size)]
    # eleva = eleva_list[(index-1)//(int(alpha_list.size)*int(thrus_list.size))]
    alpha = alpha_list[0]
    thrus = thrus_list[0]
    eleva = eleva_list[0]
    # u_inf = 1
    gravity_on = True


    # M = 8 #used to be 16
    # N = 32 #don't think 32 is ever used - 21 was num_nodes
    # Ms = 5 #how many times wake? used to be 10
    # M = 6
    # N = 2  # michigan discretisation
    # Ms = 5

    batch_log = 'batch_log_gusti{:04g}_flare{:04g}'.format(gusti * 100, flare * 100)

    with open('./{:s}.txt'.format(batch_log), 'w') as f:
        # dd/mm/YY H:M:S
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('SHARPy launch - START\n')
        f.write('Date and time = %s\n\n' % dt_string)

    for i, u_inf in enumerate(u_inf_vec):
        print('RUNNING SHARPY %f %f %f\n' % (gusti, flare, u_inf))
        case_name = 'pazy_uinf{:04g}_gusti{:04g}_flare{:04g}'.format(u_inf*10, gusti*100, flare*100)
        try:
            generate_pazy(u_inf, case_name, alpha=alpha, thrus=thrus, eleva=eleva,
                          output_folder='/output/pazy_gusti{:04g}_flare{:04g}/'.format(
                            gusti*100, flare*100),
                          cases_subfolder='/gusti{:04g}_flare{:04g}_uinf{:04g}/'.format(
                            gusti*100, flare*100, u_inf),
                          gusti=gusti, flare=flare,
                          gravity_on=gravity_on)
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            with open('./{:s}.txt'.format(batch_log), 'a') as f:
                f.write('%s Ran case %i :::: u_inf = %f\n\n' % (dt_string, i, u_inf))
        except AssertionError:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            with open('./{:s}.txt'.format(batch_log), 'a') as f:
                f.write('%s ERROR RUNNING case %f\n\n' % (dt_string, u_inf))
    
        
        






# # Loading of the used packages
# import numpy as np              # basic mathematical and array functions
# import os                       # Functions related to the operating system
# import matplotlib.pyplot as plt # Plotting library

# import sharpy.sharpy_main                  # Run SHARPy inside jupyter notebooks
# import sharpy.utils.plotutils as pu        # Plotting utilities
# from sharpy.utils.constants import deg2rad # Constant to conver degrees to radians
# import sharpy.utils.algebra as algebra

# import sharpy.utils.generate_cases as gc





# Gather data about available solvers
# Simulation details









