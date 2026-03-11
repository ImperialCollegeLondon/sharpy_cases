"""
Batch SHARPy simulation script for a flexible multibody wing with flared hinged wing tips.

Data for numerical model kindly provided by Dr. Fintan Healy, University of Bristol, as in refs
https://doi.org/10.2514/1.C036877
https://doi.org/10.2514/1.C037167
https://hdl.handle.net/1983/aa279e98-72c7-411d-bc9b-2295280cc3a8

The code generates a symmetric full wing plus wing tips, with structural and aerodynamic properties given,
defines hinge constraints, runs dynamic simulation with optional polar corrections from jig shape, 
and postprocesses then saves the time history of both hinge angles.

Tested on SHARPy v2.4, run by "python multibodywing.py (integer>=1)" 
"""

import numpy as np
import os
import unittest
import sharpy.sharpy_main
import sharpy.utils.algebra as algebra
import sharpy.utils.generate_cases as gc
import sys
index = int(sys.argv[1])
from sharpy.utils.constants import deg2rad


# Problem Set up
def generate_multibodywing(u_inf, case_name, output_folder='/output/', cases_subfolder='', **kwargs):

    from sharpy.utils.constants import deg2rad

    num_chord_panels = 4    # Number of aerodynamic panels in the chordwise direction
    num_points_camber = 200 # The camber line of the wing will be defined by a series of (x,y)
                            # coordintes. Here, we define the size of the (x,y) vectors

    chord = 0.15         # Chord of the wing
    rho = 1.225       # Air density
    u_inf_direction = [1.0,0.0,0.0]

    # Time discretization
    end_time = 50.0                   # End time of the simulation
    dt = chord/num_chord_panels/u_inf # Always keep one timestep per panel
    n_time_steps = np.rint(end_time/dt).astype(int)
    print(n_time_steps)

    aoa_ini_deg = kwargs.get('alpha', 0.)        # Angle of attack at the beginning of the simulation
    aoa_end_deg = kwargs.get('alpha', 0.)        # Angle of attack at the end of the simulation

    hinge_deg = kwargs.get('flare', 0.)
    hinge_ini = 0.

    # # gust settings
    # gust_intensity = 0.20
    # gust_length = 1 * u_inf
    # gust_offset = 0.5 * u_inf

    num_modes = 8

    c_ref = chord 

    gravity_on = kwargs.get('gravity_on', False)

    # Lattice Discretisation
    M = kwargs.get('M', 16)
    N = kwargs.get('N', 32)
    M_star_fact = kwargs.get('Ms', 10)
    mstar = M*M_star_fact

    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    # SHARPy nonlinear reference solution
    case_route = route_test_dir + '/cases/' + cases_subfolder + '/' + case_name
    if not os.path.exists(case_route):
        os.makedirs(case_route)

    wing = gc.AeroelasticInformation()

##  Bristol wing
    span = 1.0  # span
    wake_length = 10   # Length of the wake in chord lengths
    thickness = np.array([0.005, 0.0133])
    width = np.array([0.030, 0.0193])
    E = np.array([193.0e9, 1.65e9])
    nu = np.array([0.27, 0.3])
    rho = np.array([8000.0, 0.001])
    xarea = thickness*width
    G = E/(2.0*(1.0+nu))
    Ay = thickness*span
    Az = width*span
    
    mass_per_unit_length = rho*xarea # Mass per unit length
    mass_iner_y = mass_per_unit_length*thickness*thickness/12.0          # Mass inertia around the local y axis
    mass_iner_z = mass_per_unit_length*width*width/12.0          # Mass inertia around the local z axis
    mass_iner_x = mass_iner_y+mass_iner_z

    # Discretization
    num_node = 13           # Number of nodes in the structural discretisation
                            # The number of nodes will also define the aerodynamic panels in the
                            # spanwise direction
    num_chord_panels = 4    # Number of aerodynamic panels in the chordwise direction
    num_points_camber = 200 # The camber line of the wing will be defined by a series of (x,y)
                            # coordintes. Here, we define the size of the (x,y) vectors

    wing.StructuralInformation.num_node = num_node
    wing.StructuralInformation.num_node_elem = 3
    wing.StructuralInformation.compute_basic_num_elem()
    wing.StructuralInformation.create_simple_connectivities()

    node_r = np.zeros((num_node, 3), dtype=float)
    node_r[:,1] = np.array([0.0, 0.035, 0.0875, 0.1925, 0.2975, 0.4025, 0.5075, 0.6125, 0.7175, 0.8225, 0.875, 0.952, 1.0])
    print(node_r)
    elem_stiffness = np.array([0, 0, 0, 0, 0, 1])
    wing.StructuralInformation.create_stiff_db_from_vector(E*xarea,
                                                           G*Ay,
                                                           G*Az,
                                                           G*mass_iner_x/rho,
                                                           E*mass_iner_y/rho,
                                                           E*mass_iner_z/rho,
                                                           vec_EIyz=None)
    elem_mass = np.array([0, 0, 0, 0, 0, 1])
    pos_cg_B = np.zeros((num_node, 3), dtype=float)
    wing.StructuralInformation.create_mass_db_from_vector(mass_per_unit_length,
                                   mass_iner_x,
                                   mass_iner_y,
                                   mass_iner_z,
                                   pos_cg_B,
                                   vec_mass_iner_yz=None)
    wing.StructuralInformation.create_frame_of_reference_delta(np.array([-1,0,0]))

    wing.StructuralInformation.generate_full_structure(wing.StructuralInformation.num_node_elem, 
                                                    wing.StructuralInformation.num_node, 
                                                    wing.StructuralInformation.num_elem, 
                                                    node_r, 
                                                    wing.StructuralInformation.connectivities, 
                                                    elem_stiffness, 
                                                    wing.StructuralInformation.stiffness_db, 
                                                    elem_mass, 
                                                    wing.StructuralInformation.mass_db, 
                                                    wing.StructuralInformation.frame_of_reference_delta, 
                                                    np.zeros((wing.StructuralInformation.num_elem, 3)), #structural_twist
                                                    np.zeros((wing.StructuralInformation.num_node), dtype=int), #boundary_conditions
                                                    np.zeros((wing.StructuralInformation.num_elem), dtype=int), #beam_number
                                                    np.zeros((wing.StructuralInformation.num_node, 6)), #app_forces
                                                    lumped_mass_nodes=None, 
                                                    lumped_mass=None, 
                                                    lumped_mass_inertia=None, 
                                                    lumped_mass_position=None, 
                                                    lumped_mass_mat_nodes=None, 
                                                    lumped_mass_mat=None)

    print(wing.StructuralInformation.connectivities)
    wing.StructuralInformation.boundary_conditions[0] = 1
    wing.StructuralInformation.boundary_conditions[-1] = -1

    m1 = 0.142
    m2 = 0.353

    wing.StructuralInformation.lumped_mass_nodes = np.array([2, 3, 4, 5, 6, 7, 8, 9, 11], dtype = int)
    wing.StructuralInformation.lumped_mass = np.append(np.ones((8,))*m1,m2)
    wing.StructuralInformation.lumped_mass_inertia = np.append(np.tile(np.diag([171.0/1e6, 85.0/1e6, 0]),(8,1,1)), np.array([np.diag([269.0/1e6, 355.0/1e6, 0])]),0)
    wing.StructuralInformation.lumped_mass_position = np.append(np.tile(np.array([0, -30.5/1e3, 0]),(8,1)), np.array([[0, -23.2/1e3, 0]]),0)

    # Define the coordinates of the camber line of the wing
    wing_camber = np.zeros((1, num_points_camber, 2))
    wing_camber[0, :, 0] = np.linspace(0, 1, num_points_camber)

    # Generate blade aerodynamics
    wing.AerodynamicInformation.create_one_uniform_aerodynamics(wing.StructuralInformation,
                                     chord = chord,
                                     twist = 0.,
                                     sweep = 0.,
                                     num_chord_panels = num_chord_panels,
                                     m_distribution = 'uniform',
                                     elastic_axis = 0.25,
                                     num_points_camber = num_points_camber,
                                     airfoil = wing_camber)
    
    wing.AerodynamicInformation.sweep[wing.StructuralInformation.num_elem-1, :] = np.array([0, -np.deg2rad(hinge_deg),0])
    print(wing.AerodynamicInformation.sweep)

    wing_left = gc.AeroelasticInformation()

##  Bristol wing_left
    span = 1.0  # span
    wake_length = 10   # Length of the wake in chord lengths
    thickness = np.array([0.005, 0.0133])
    width = np.array([0.030, 0.0193])
    E = np.array([193.0e9, 1.65e9])
    nu = np.array([0.27, 0.3])
    rho = np.array([8000.0, 0.001])
    xarea = thickness*width
    G = E/(2.0*(1.0+nu))
    Ay = thickness*span
    Az = width*span
    
    mass_per_unit_length = rho*xarea # Mass per unit length
    mass_iner_y = mass_per_unit_length*thickness*thickness/12.0          # Mass inertia around the local y axis
    mass_iner_z = mass_per_unit_length*width*width/12.0          # Mass inertia around the local z axis
    mass_iner_x = mass_iner_y+mass_iner_z

    # Discretization
    num_node = 13           # Number of nodes in the structural discretisation
                            # The number of nodes will also define the aerodynamic panels in the
                            # spanwise direction
    num_chord_panels = 4    # Number of aerodynamic panels in the chordwise direction
    num_points_camber = 200 # The camber line of the wing_left will be defined by a series of (x,y)
                            # coordintes. Here, we define the size of the (x,y) vectors

    wing_left.StructuralInformation.num_node = num_node
    wing_left.StructuralInformation.num_node_elem = 3
    wing_left.StructuralInformation.compute_basic_num_elem()
    wing_left.StructuralInformation.create_simple_connectivities()

    node_r_left = np.zeros((num_node, 3), dtype=float)
    node_r_left[:,1] = np.array([-0.0, -0.035, -0.0875, -0.1925, -0.2975, -0.4025, -0.5075, -0.6125, -0.7175, -0.8225, -0.875, -0.952, -1.0])
    print(node_r_left)
    elem_stiffness = np.array([0, 0, 0, 0, 0, 1])
    wing_left.StructuralInformation.create_stiff_db_from_vector(E*xarea,
                                                           G*Ay,
                                                           G*Az,
                                                           G*mass_iner_x/rho,
                                                           E*mass_iner_y/rho,
                                                           E*mass_iner_z/rho,
                                                           vec_EIyz=None)
    elem_mass = np.array([0, 0, 0, 0, 0, 1])
    pos_cg_B = np.zeros((num_node, 3), dtype=float)
    wing_left.StructuralInformation.create_mass_db_from_vector(mass_per_unit_length,
                                   mass_iner_x,
                                   mass_iner_y,
                                   mass_iner_z,
                                   pos_cg_B,
                                   vec_mass_iner_yz=None)
    wing_left.StructuralInformation.create_frame_of_reference_delta(np.array([1,0,0]))

    wing_left.StructuralInformation.generate_full_structure(wing_left.StructuralInformation.num_node_elem, 
                                                    wing_left.StructuralInformation.num_node, 
                                                    wing_left.StructuralInformation.num_elem, 
                                                    node_r_left, 
                                                    wing_left.StructuralInformation.connectivities, 
                                                    elem_stiffness, 
                                                    wing_left.StructuralInformation.stiffness_db, 
                                                    elem_mass, 
                                                    wing_left.StructuralInformation.mass_db, 
                                                    wing_left.StructuralInformation.frame_of_reference_delta, 
                                                    np.zeros((wing_left.StructuralInformation.num_elem, 3)), #structural_twist
                                                    np.zeros((wing_left.StructuralInformation.num_node), dtype=int), #boundary_conditions
                                                    np.zeros((wing_left.StructuralInformation.num_elem), dtype=int), #beam_number
                                                    np.zeros((wing_left.StructuralInformation.num_node, 6)), #app_forces
                                                    lumped_mass_nodes=None, 
                                                    lumped_mass=None, 
                                                    lumped_mass_inertia=None, 
                                                    lumped_mass_position=None, 
                                                    lumped_mass_mat_nodes=None, 
                                                    lumped_mass_mat=None)



    print(wing_left.StructuralInformation.connectivities)
    wing_left.StructuralInformation.boundary_conditions[0] = 0
    wing_left.StructuralInformation.boundary_conditions[-1] = -1

    m1 = 0.142
    m2 = 0.353

    wing_left.StructuralInformation.lumped_mass_nodes = np.array([2, 3, 4, 5, 6, 7, 8, 9, 11], dtype = int)
    wing_left.StructuralInformation.lumped_mass = np.append(np.ones((8,))*m1,m2)
    wing_left.StructuralInformation.lumped_mass_inertia = np.append(np.tile(np.diag([171.0/1e6, 85.0/1e6, 0]),(8,1,1)), np.array([np.diag([269.0/1e6, 355.0/1e6, 0])]),0)
    wing_left.StructuralInformation.lumped_mass_position = np.append(np.tile(np.array([0, 30.5/1e3, 0]),(8,1)), np.array([[0, 23.2/1e3, 0]]),0)

    # Define the coordinates of the camber line of the wing_left
    wing_left_camber = np.zeros((1, num_points_camber, 2))
    wing_left_camber[0, :, 0] = np.linspace(0, 1, num_points_camber)

    # Generate blade aerodynamics
    wing_left.AerodynamicInformation.create_one_uniform_aerodynamics(wing_left.StructuralInformation,
                                     chord = chord,
                                     twist = 0.,
                                     sweep = 0.,
                                     num_chord_panels = num_chord_panels,
                                     m_distribution = 'uniform',
                                     elastic_axis = 0.25,
                                     num_points_camber = num_points_camber,
                                     airfoil = wing_left_camber)
    
    wing_left.AerodynamicInformation.sweep[wing_left.StructuralInformation.num_elem-1, :] = np.array([0, np.deg2rad(hinge_deg),0])
    print(wing_left.AerodynamicInformation.sweep)

    ## winglet params
    span_winglet = 0.345*span
    num_node_winglet = 7
    # r_winglet = np.linspace(0.0, span_winglet, num_node_winglet)
    r_winglet = np.array([0.0, 0.06, 0.12, 0.1799, 0.24, 0.30, span_winglet])

    # mass_per_unit_length_winglet = 0.520/0.345
    mass_per_unit_length_winglet = 0.001
    mass_iner_x_winglet = 0.001
    mass_iner_y_winglet = 0.001
    mass_iner_z_winglet = 0.001
    pos_cg_B_winglet = pos_cg_B[0]
    EA_winglet = 1e11
    GAy_winglet = 1e11
    GAz_winglet = 1e11
    GJ_winglet = 1e11
    EIy_winglet = 1e11
    EIz_winglet = 1e11

    winglet = gc.AeroelasticInformation()
    # Define the number of nodes and the number of nodes per element
    winglet.StructuralInformation.num_node = num_node_winglet
    winglet.StructuralInformation.num_node_elem = 3
    # Compute the number of elements assuming basic connections
    winglet.StructuralInformation.compute_basic_num_elem()

    # Generate an array with the location of the nodes
    node_r_winglet = np.zeros((num_node_winglet, 3))

    node_x = np.sin(np.deg2rad(hinge_deg))*np.sin(np.deg2rad(hinge_deg))+np.cos(np.deg2rad(hinge_deg))*(1+np.cos(np.deg2rad(2*hinge_deg)))/np.sqrt(2+2*np.cos(np.deg2rad(2*hinge_deg)))*np.cos(np.deg2rad(hinge_ini))
    node_y = np.sin(np.deg2rad(hinge_deg))*np.cos(np.deg2rad(hinge_deg))-np.cos(np.deg2rad(hinge_deg))*np.sin(np.deg2rad(2*hinge_deg))/np.sqrt(2+2*np.cos(np.deg2rad(2*hinge_deg)))*np.cos(np.deg2rad(hinge_ini))
    node_z = np.cos(np.deg2rad(hinge_deg))*np.sin(np.deg2rad(hinge_ini))

    node_r_winglet[:,0] = node_r[-1, 0]+r_winglet*node_y
    node_r_winglet[:,1] = node_r[-1, 1]+r_winglet*node_x
    node_r_winglet[:,2] = node_r[-1, 2]+r_winglet*node_z
    print(node_r_winglet)

    winglet.StructuralInformation.generate_uniform_beam(node_r_winglet,
                        mass_per_unit_length_winglet,
                        mass_iner_x_winglet,
                        mass_iner_y_winglet,
                        mass_iner_z_winglet,
                        pos_cg_B_winglet,
                        EA_winglet,
                        GAy_winglet,
                        GAz_winglet,
                        GJ_winglet,
                        EIy_winglet,
                        EIz_winglet,
                        num_node_elem = winglet.StructuralInformation.num_node_elem,
                        y_BFoR = np.array([-1,0,0]),
                        num_lumped_mass=0)


    print(winglet.StructuralInformation.connectivities)

    mwinglet = 0.520

    winglet.StructuralInformation.boundary_conditions[0] = 1
    winglet.StructuralInformation.boundary_conditions[-1] = -1
    winglet.StructuralInformation.lumped_mass_nodes = np.array([3], dtype = int)
    winglet.StructuralInformation.lumped_mass = np.ones((1,))*mwinglet
    winglet.StructuralInformation.lumped_mass_inertia = np.array([np.diag([680/1e6, 3858/1e6, 2807/1e6])])
    winglet.StructuralInformation.lumped_mass_position = np.array([[0, -23.8/1e3, 0]])
    # Compute the number of panels in the wake (streamwise direction) based on the previous paramete
    wake_panels = int(wake_length*chord/dt)

    # Define the coordinates of the camber line of the wing
    winglet_camber = np.zeros((1, num_points_camber, 2))
    winglet_camber[0, :, 0] = np.linspace(0, 1, num_points_camber)

    # Generate blade aerodynamics
    winglet.AerodynamicInformation.create_one_uniform_aerodynamics(winglet.StructuralInformation,
                                     chord = chord,
                                     twist = 0.,
                                     sweep = 0.,
                                     num_chord_panels = num_chord_panels,
                                     m_distribution = 'uniform',
                                     elastic_axis = 0.25,
                                     num_points_camber = num_points_camber,
                                     airfoil = winglet_camber)      

    winglet.AerodynamicInformation.sweep[0, :] = np.array([-np.deg2rad(hinge_deg),0,0])
    print(winglet.AerodynamicInformation.sweep)                           

    # winglet_left params
    span_winglet_left = 0.345*span
    num_node_winglet_left = 7
    # r_winglet_left = np.linspace(0.0, span_winglet_left, num_node_winglet_left)
    r_winglet_left = np.array([-0.0, -0.06, -0.12, -0.1799, -0.24, -0.30, -span_winglet_left])
    
    # mass_per_unit_length_winglet_left = 0.520/0.345
    mass_per_unit_length_winglet_left = 0.001
    mass_iner_x_winglet_left = 0.001
    mass_iner_y_winglet_left = 0.001
    mass_iner_z_winglet_left = 0.001
    pos_cg_B_winglet_left = pos_cg_B[0]
    EA_winglet_left = 1e11
    GAy_winglet_left = 1e11
    GAz_winglet_left = 1e11
    GJ_winglet_left = 1e11
    EIy_winglet_left = 1e11
    EIz_winglet_left = 1e11

    winglet_left = gc.AeroelasticInformation()
    # Define the number of nodes and the number of nodes per element
    winglet_left.StructuralInformation.num_node = num_node_winglet_left
    winglet_left.StructuralInformation.num_node_elem = 3
    # Compute the number of elements assuming basic connections
    winglet_left.StructuralInformation.compute_basic_num_elem()

    # Generate an array with the location of the nodes
    node_r_winglet_left = np.zeros((num_node_winglet_left, 3))

    node_x = np.sin(np.deg2rad(hinge_deg))*np.sin(np.deg2rad(hinge_deg))+np.cos(np.deg2rad(hinge_deg))*(1+np.cos(np.deg2rad(2*hinge_deg)))/np.sqrt(2+2*np.cos(np.deg2rad(2*hinge_deg)))*np.cos(np.deg2rad(hinge_ini))
    node_y = np.sin(np.deg2rad(hinge_deg))*np.cos(np.deg2rad(hinge_deg))-np.cos(np.deg2rad(hinge_deg))*np.sin(np.deg2rad(2*hinge_deg))/np.sqrt(2+2*np.cos(np.deg2rad(2*hinge_deg)))*np.cos(np.deg2rad(hinge_ini))
    node_z = np.cos(np.deg2rad(hinge_deg))*np.sin(np.deg2rad(hinge_ini))

    node_r_winglet_left[:,0] = node_r_left[-1, 0]+r_winglet_left*node_y
    node_r_winglet_left[:,1] = node_r_left[-1, 1]+r_winglet_left*node_x
    node_r_winglet_left[:,2] = node_r_left[-1, 2]+r_winglet_left*node_z
    print(node_r_winglet_left)

    winglet_left.StructuralInformation.generate_uniform_beam(node_r_winglet_left,
                        mass_per_unit_length_winglet_left,
                        mass_iner_x_winglet_left,
                        mass_iner_y_winglet_left,
                        mass_iner_z_winglet_left,
                        pos_cg_B_winglet_left,
                        EA_winglet_left,
                        GAy_winglet_left,
                        GAz_winglet_left,
                        GJ_winglet_left,
                        EIy_winglet_left,
                        EIz_winglet_left,
                        num_node_elem = winglet_left.StructuralInformation.num_node_elem,
                        y_BFoR = np.array([1,0,0]),
                        num_lumped_mass=0)

    mwinglet_left = 0.520

    winglet_left.StructuralInformation.boundary_conditions[0] = 1
    winglet_left.StructuralInformation.boundary_conditions[-1] = -1
    winglet_left.StructuralInformation.lumped_mass_nodes = np.array([3], dtype = int)
    winglet_left.StructuralInformation.lumped_mass = np.ones((1,))*mwinglet_left
    winglet_left.StructuralInformation.lumped_mass_inertia = np.array([np.diag([680/1e6, 3858/1e6, 2807/1e6])])
    winglet_left.StructuralInformation.lumped_mass_position = np.array([[0, 23.8/1e3, 0]])
    # Compute the number of panels in the wake (streamwise direction) based on the previous paramete
    wake_panels = int(wake_length*chord/dt)

    # Define the coordinates of the camber line of the wing
    winglet_left_camber = np.zeros((1, num_points_camber, 2))
    winglet_left_camber[0, :, 0] = np.linspace(0, 1, num_points_camber)

    # Generate blade aerodynamics
    winglet_left.AerodynamicInformation.create_one_uniform_aerodynamics(winglet_left.StructuralInformation,
                                     chord = chord,
                                     twist = 0.,
                                     sweep = 0.,
                                     num_chord_panels = num_chord_panels,
                                     m_distribution = 'uniform',
                                     elastic_axis = 0.25,
                                     num_points_camber = num_points_camber,
                                     airfoil = winglet_left_camber)      

    winglet_left.AerodynamicInformation.sweep[0, :] = np.array([np.deg2rad(hinge_deg),0,0])                          

    polar_route = route_test_dir + '/xf-naca0015-il-200000.txt'
    polar_raw_data = np.loadtxt(polar_route, skiprows=12)
    # import pdb
    # pdb.set_trace()

    winglet.AerodynamicInformation.polars = np.column_stack((polar_raw_data[:, 0] * np.pi / 180, # aoa
                                       polar_raw_data[:, 1], # cl
                                       polar_raw_data[:, 2], # cd
                                       polar_raw_data[:, 4])) #cm
    winglet_left.AerodynamicInformation.polars = np.column_stack((polar_raw_data[:, 0] * np.pi / 180, # aoa
                                       polar_raw_data[:, 1], # cl
                                       polar_raw_data[:, 2], # cd
                                       polar_raw_data[:, 4])) #cm

    wing.AerodynamicInformation.polars = np.column_stack((polar_raw_data[:, 0] * np.pi / 180, # aoa
                                       polar_raw_data[:, 1], # cl
                                       polar_raw_data[:, 2], # cd
                                       polar_raw_data[:, 4])) #cm
    wing_left.AerodynamicInformation.polars = np.column_stack((polar_raw_data[:, 0] * np.pi / 180, # aoa
                                       polar_raw_data[:, 1], # cl
                                       polar_raw_data[:, 2], # cd
                                       polar_raw_data[:, 4])) #cm
    wing.assembly(wing_left)
    wing.remove_duplicated_points(1e-15)

    wing.StructuralInformation.body_number *= 0

    wing.assembly(winglet)
    wing.assembly(winglet_left)


    SimInfo = gc.SimulationInformation()
    SimInfo.set_default_values()
    SimInfo.set_variable_all_dicts('dt', dt)
    SimInfo.set_variable_all_dicts('u_inf_direction',np.array([1., 0., 0.]))
    u_inf_direction = np.array([1., 0., 0.])


    SimInfo.solvers['SHARPy']['flow'] = ['BeamLoader', 'AerogridLoader',
             'DynamicCoupled',
             'SaveParametricCase'
             ]

    SimInfo.solvers['SHARPy']['case'] = case_name 
    SimInfo.solvers['SHARPy']['write_screen'] = 'on'
    SimInfo.solvers['SHARPy']['route'] = case_route
    SimInfo.solvers['SHARPy']['log_folder'] =  SimInfo.solvers['SHARPy']['route'] + '/output/'



    SimInfo.solvers['BeamLoader']['unsteady']= 'on'
    SimInfo.solvers['BeamLoader']['orientation']= algebra.euler2quat([0, aoa_ini_deg * np.pi/180, 0])


    SimInfo.solvers['AerogridLoader']['unsteady'] = 'on'
    SimInfo.solvers['AerogridLoader']['mstar'] = mstar
    SimInfo.solvers['AerogridLoader']['aligned_grid'] = 'off'
    SimInfo.solvers['AerogridLoader']['freestream_dir'] = u_inf_direction
    SimInfo.solvers['AerogridLoader']['wake_shape_generator'] = 'StraightWake'
    SimInfo.solvers['AerogridLoader']['wake_shape_generator_input'] = {'u_inf':u_inf,
                                                                           'u_inf_direction': u_inf_direction,
                                                                           'dt': dt}

    SimInfo.solvers['AerogridPlot']['include_rbm'] = 'off'
    SimInfo.solvers['AerogridPlot']['include_applied_forces'] = 'on'
    SimInfo.solvers['AerogridPlot']['minus_m_star'] = 0
    SimInfo.solvers['AerogridPlot']['stride'] = 10

    SimInfo.solvers['BeamPlot']['include_rbm'] = 'off'
    SimInfo.solvers['BeamPlot']['include_applied_forces'] = 'on'
    SimInfo.solvers['BeamPlot']['stride'] = 10


    SimInfo.solvers['StepUvlm']['convection_scheme'] = 3
    # SimInfo.solvers['StepUvlm']['velocity_field_generator'] = 'GustVelocityField'
    SimInfo.solvers['StepUvlm']['velocity_field_generator'] = 'SteadyVelocityField'
    SimInfo.solvers['StepUvlm']['vortex_radius_wake_ind'] = 1e-6        # increase if wake panels do weird spike up
    SimInfo.solvers['StepUvlm']['vortex_radius'] = 1e-6                 # increase if wake panels do weird spike up
    SimInfo.solvers['StepUvlm']['velocity_field_input'] = {'u_inf': u_inf,
                                                             'u_inf_direction': u_inf_direction
                                                            #  'gust_shape': '1-cos',
                                                            #  'gust_parameters': {'gust_length': gust_length,
                                                            #                      'gust_intensity': gust_intensity * u_inf},
                                                            #                      'offset': gust_offset,
                                                            #                      'relative_motion': 'off'}
    }

    SimInfo.solvers['NonLinearDynamicMultibody']['min_delta'] = 1e-3
    SimInfo.solvers['NonLinearDynamicMultibody']['abs_threshold'] = 1e-6
    SimInfo.solvers['NonLinearDynamicMultibody']['gravity_on'] = gravity_on
    SimInfo.solvers['NonLinearDynamicMultibody']['gravity'] = 9.81
    SimInfo.solvers['NonLinearDynamicMultibody']['time_integrator'] = 'NewmarkBeta'
    SimInfo.solvers['NonLinearDynamicMultibody']['time_integrator_settings'] = {'newmark_damp': 0.0001,
                                                                                'dt': dt}
    SimInfo.solvers['NonLinearDynamicMultibody']['write_lm'] = True
    SimInfo.solvers['NonLinearDynamicMultibody']['num_steps'] = n_time_steps
    SimInfo.solvers['NonLinearDynamicMultibody']['max_iterations'] = 199
 
    SimInfo.solvers['WriteVariablesTime']['structure_variables'] = ['pos','psi','psi_dot','psi_local','psi_dot_local']
    SimInfo.solvers['WriteVariablesTime']['FoR_variables'] = ['mb_quat','in_global_AFoR','mb_FoR_vel']
    SimInfo.solvers['WriteVariablesTime']['FoR_number'] = np.array([0, 1], dtype=int)
    SimInfo.solvers['WriteVariablesTime']['structure_nodes'] = list(range(0, (num_node+num_node_winglet)*2-1))
        
    SimInfo.solvers['DynamicCoupled']['structural_solver'] = 'NonLinearDynamicMultibody'
    SimInfo.solvers['DynamicCoupled']['structural_solver_settings'] = SimInfo.solvers['NonLinearDynamicMultibody']
    SimInfo.solvers['DynamicCoupled']['aero_solver'] = 'StepUvlm'
    SimInfo.solvers['DynamicCoupled']['aero_solver_settings'] = SimInfo.solvers['StepUvlm']

    SimInfo.solvers['DynamicCoupled']['n_time_steps'] = n_time_steps

    SimInfo.solvers['DynamicCoupled']['postprocessors'] = ['WriteVariablesTime', 'BeamPlot', 'AerogridPlot']
    SimInfo.solvers['DynamicCoupled']['postprocessors_settings'] = {'WriteVariablesTime': SimInfo.solvers['WriteVariablesTime'],
                                                                'BeamPlot': SimInfo.solvers['BeamPlot'],
                                                                'AerogridPlot': SimInfo.solvers['AerogridPlot']}

    # # If polar corrections needed...
    # SimInfo.solvers['DynamicCoupled']['correct_forces_method'] = 'PolarCorrection'
    # SimInfo.solvers['DynamicCoupled']['correct_forces_settings'] = {'cd_from_cl': 'off',
    #                                                             'correct_lift': 'on',
    #                                                             'moment_from_polar': 'on'}


    SimInfo.solvers['SaveParametricCase']['save_case'] = False
    SimInfo.solvers['SaveParametricCase']['parameters'] = {'u_inf': u_inf, 'alpha': aoa_ini_deg, 'flare': flare}
    

    # Create the MB and BC files
    LC2 = gc.LagrangeConstraint()
    LC2.behaviour = 'hinge_node_FoR'
    LC2.node_in_body = num_node-1
    LC2.body = 0
    LC2.body_FoR = 1
    LC2.scalingFactor = 1e8
    LC2.rot_axisA2 = algebra.rotation3d_z(90*deg2rad) @ np.array([-np.sin(hinge_deg*deg2rad),-np.cos(hinge_deg*deg2rad),0.0])
    LC2.rot_axisB = (np.array([-np.sin(hinge_deg*deg2rad),-np.cos(hinge_deg*deg2rad),0.0]))
    print(LC2.rot_axisA2)

    LC4 = gc.LagrangeConstraint()
    LC4.behaviour = 'hinge_node_FoR'
    LC4.node_in_body = num_node*2-2
    LC4.body = 0
    LC4.body_FoR = 2
    LC4.scalingFactor = 1e8
    LC4.rot_axisA2 = algebra.rotation3d_z(-90*deg2rad) @ np.array([-np.sin(hinge_deg*deg2rad),np.cos(hinge_deg*deg2rad),0.0])
    LC4.rot_axisB = (np.array([-np.sin(hinge_deg*deg2rad),np.cos(hinge_deg*deg2rad),0.0]))
    print(LC4.rot_axisA2)

    LC = []
    LC.append(LC2)
    LC.append(LC4)

    MB1 = gc.BodyInformation()
    MB1.body_number = 0
    MB1.FoR_position = np.zeros((6,),)
    MB1.FoR_velocity = np.zeros((6,),)
    MB1.FoR_acceleration = np.zeros((6,),)
    MB1.FoR_movement = 'prescribed'
    MB1.quat = algebra.euler2quat([0, aoa_ini_deg * np.pi/180, 0])

    MB2 = gc.BodyInformation()
    MB2.body_number = 1
    MB2.FoR_position = np.array([node_r_winglet[0, 0], node_r_winglet[0, 1], node_r_winglet[0, 2], 0.0, 0.0, 0.0])
    MB2.FoR_velocity = np.zeros((6,),)
    MB2.FoR_acceleration = np.zeros((6,),)
    MB2.FoR_movement = 'free'
    MB2.quat = algebra.euler2quat([0, aoa_ini_deg * np.pi/180, 0])

    MB4 = gc.BodyInformation()
    MB4.body_number = 2
    MB4.FoR_position = np.array([node_r_winglet_left[0, 0], node_r_winglet_left[0, 1], node_r_winglet_left[0, 2], 0.0, 0.0, 0.0])
    MB4.FoR_velocity = np.zeros((6,),)
    MB4.FoR_acceleration = np.zeros((6,),)
    MB4.FoR_movement = 'free'
    MB4.quat = algebra.euler2quat([0, aoa_ini_deg * np.pi/180, 0])


    MB = []
    MB.append(MB1)
    MB.append(MB2)
    MB.append(MB4)

                               
    SimInfo.with_forced_vel = False
    SimInfo.with_dynamic_forces = False

    gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])

    wing.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
    gc.generate_multibody_file(LC, MB,SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])

    SimInfo.generate_solver_file()
    SimInfo.generate_dyn_file(n_time_steps)

    print('Running {}'.format(case_route + '/' + case_name + '.sharpy'))
    case_data = sharpy.sharpy_main.main(['', case_route + '/' + case_name + '.sharpy'])

    ## postprocess for hinge angle
    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    u_inf = u_inf
    hinge_deg = flare
    alpha_deg = aoa_ini_deg


    # extract information hinge1
    n_tsteps = len(case_data.structure.timestep_info)
    theta1 = np.zeros((n_tsteps, 1))
    for it in range(n_tsteps):
        psiH = case_data.structure.timestep_info[it].psi[12, 0]
        psiB = case_data.structure.timestep_info[it].psi[5, 1]
        theta1[it] = algebra.quat2euler(algebra.rotation2quat(((algebra.crv2rotation(psiH).T) @ algebra.crv2rotation(psiB))))[1]

    route_export = route_test_dir + '/output'
    if not os.path.exists(route_export):
        os.makedirs(route_export)  
    dest_file = route_export + '/theta1_uinf{:04g}_flare{:04g}_alpha{:04g}.txt'.format(u_inf * 100, hinge_deg * 100, alpha_deg * 100)
    np.savetxt(dest_file, np.column_stack(theta1))
    print('Saved theta1 array to {}'.format(dest_file))

    # extract information hinge2
    n_tsteps = len(case_data.structure.timestep_info)
    theta2 = np.zeros((n_tsteps, 1))
    for it in range(n_tsteps):
        psiH = case_data.structure.timestep_info[it].psi[15, 0]
        psiB = case_data.structure.timestep_info[it].psi[11, 1]
        theta2[it] = algebra.quat2euler(algebra.rotation2quat(((algebra.crv2rotation(psiH).T) @ algebra.crv2rotation(psiB))))[1]

    route_export = route_test_dir + '/output'
    if not os.path.exists(route_export):
        os.makedirs(route_export)  
    dest_file = route_export + '/theta2_uinf{:04g}_flare{:04g}_alpha{:04g}.txt'.format(u_inf * 100, hinge_deg * 100, alpha_deg * 100)
    np.savetxt(dest_file, np.column_stack(theta2))
    print('Saved theta2 array to {}'.format(dest_file))    

if __name__ == '__main__':
    from datetime import datetime

    # vector of freestream velocity
    u_inf_vec = [12]

    # vector of angle of attacks under investigation
    alpha_list = np.array([-7.5, -5.5, -5., -4.5, -3., -2.5, -2., -0.5, 0., 0.5, 1.5, 2., 2.5, 3., 4.5, 5., 5.5, 7., 7.5, 8., 9.5, 10., 10.5, 12., 12.5, 13., 15.])
 
    # vector of flare angles under investigation
    flare_list = np.linspace(5, 35, 7)

    alpha = alpha_list[(index-1)%int(alpha_list.size)]
    flare = flare_list[(index-1)//int(alpha_list.size)]
    gravity_on = True


    M = 8 #used to be 16
    N = 32 #don't think 32 is ever used - 21 was num_nodes
    Ms = 5 #how many times wake? used to be 10

    batch_log = 'batch_log_alpha{:04g}'.format(alpha * 100)

    with open('./{:s}.txt'.format(batch_log), 'w') as f:
        # dd/mm/YY H:M:S
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('SHARPy launch - START\n')
        f.write('Date and time = %s\n\n' % dt_string)

    for i, u_inf in enumerate(u_inf_vec):
        print('RUNNING SHARPY %f %f %f\n' % (alpha, flare, u_inf))
        case_name = 'mbwing_uinf{:04g}_alpha{:04g}_flare{:04g}'.format(u_inf*10, alpha*100, flare*100)
        try:
            generate_multibodywing(u_inf, case_name,
                          output_folder='/output/mbwing_M{:g}N{:g}Ms{:g}_alpha{:04g}_flare{:04g}/'.format(
                              M, N, Ms, alpha*100, flare*100),
                          cases_subfolder='/M{:g}N{:g}Ms{:g}/'.format(
                              M, N, Ms),
                          M=M, N=N, Ms=Ms, alpha=alpha, flare=flare,
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
    




