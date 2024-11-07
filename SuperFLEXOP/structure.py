#! /usr/bin/env python3
import h5py as h5
import pandas as pd
import numpy as np
from scipy.io import loadmat, matlab

# Define all main parameters in the aircraft 
# MODEL GEOMETRY
# beam
span_main = 7.07
half_wing_span = span_main*0.5
sweep_LE_main = np.deg2rad(20.)
chord_main_root = 0.471 
chord_main_tip = 0.236

# calculated inputs
x_tip = half_wing_span*np.tan(sweep_LE_main)
sweep_quarter_chord = np.arctan((x_tip+chord_main_tip/4-chord_main_root/4)/(half_wing_span))
sweep_TE_main= np.arctan((x_tip + chord_main_tip - chord_main_root)/(half_wing_span))

# calculated inputs
x_tip = span_main*0.5*np.tan(sweep_LE_main)
sweep_quarter_chord = np.arctan((x_tip+chord_main_tip/4-chord_main_root/4)/(span_main*0.5))

# Fuselage information
length_fuselage = 3.44
offset_fuselage_vertical = 0
offset_wing_nose = 0.87692 + chord_main_root * 0.57 # see FLEXOP Report wing COS position (0.57 := shear center y=0)
offset_tail_nose = 2.86236881559
sigma_fuselage = 10
m_bar_fuselage = 0.3
j_bar_fuselage = 0.1

# Tail
tail_chord_tip = 0.180325
tail_sweep_LE = np.deg2rad(19.51951)
tail_sweep_TE = np.deg2rad(18.0846)
half_tail_span = 1.318355
tail_span = 2*half_tail_span

# calculated inputs
tail_x_tip = half_tail_span*np.tan(tail_sweep_LE)
tail_chord_root = 0.35 # tail_x_tip + tail_chord_tip - half_tail_span*np.tan(tail_sweep_TE)
tail_sweep_quarter_chord = np.arctan((tail_x_tip+tail_chord_tip/4-tail_chord_root/4)/(half_tail_span))

v_tail_angle = np.deg2rad(35.)
tail_sweep_quarter_chord
span_tail = 2.5
ea_tail = 0.5
sigma_tail = 10
m_bar_tail = 0.3
j_bar_tail = 0.1

# Material
material = "reference"
n_stiffness_per_wing = 17



# Ailerons
numb_ailerons = 4
y_coord_ailerons= np.array([0.862823, 2.820273, 4.301239, 5.653424, 6.928342])/2.

# Elevators
numb_elevators = 2
y_coord_elevators = np.array([0.258501, 0.788428, 1.318355])/2.

# Fuselage = 
y_coord_junction = 0.144



class FLEXOPStructure:

    def __init__(self, case_name, case_route, source_directory, **kwargs):
        self.material = material
        self.sigma = kwargs.get('sigma', 1)
        self.n_elem_multiplier = kwargs.get('n_elem_multiplier', 1.5)
        self.n_elem_multiplier_tail = kwargs.get('n_elem_multiplier_tail', self.n_elem_multiplier)
        self.n_elem_multiplier_fuselage = kwargs.get('n_elem_multiplier_fuselage', 2)

        self.route = case_route
        self.case_name = case_name

        self.thrust = kwargs.get('thrust', 0.)

        self.n_elem = None
        self.n_node = None
        self.n_node_elem = 3

        self.x = None
        self.y = None
        self.z = None

        self.n_elem_main = None
        self.n_elem_root_main = None
        self.n_elem_junction_main = None
        self.n_elem_per_aileron = None
        self.n_elem_tip_main = None

        self.n_elem_fuselage = None

        self.n_elem_tail = None
        self.n_elem_per_elevator = None
        self.n_elem_junction_tail = None

        self.n_node_main = None
        self.n_node_fuselage = None
        self.n_node_tail = None

        self.span_main = span_main
        self.span_tail = span_tail
        self.y_coord_junction = y_coord_junction # Radius fuselage at wing

        self.wing_only = kwargs.get('wing_only', True)
        self.lifting_only = kwargs.get('lifting_only', True)

        self.ignore_lumped_masses_wing = kwargs.get('ignore_lumped_masses_wing', False)

        self.n_stiffness_per_wing = n_stiffness_per_wing
        self.n_ailerons_per_wing = numb_ailerons
        self.n_elev_per_tail_surf = numb_elevators
        self.v_tail_angle = v_tail_angle
        self.tail_sweep_quarter_chord = tail_sweep_quarter_chord
        self.sweep_quarter_chord = 0.319923584301128
        self.dx_payload = kwargs.get('delta_x_payload', 0.)
        self.source_directory = source_directory

    def set_thrust(self, value):
        self.thrust = value

    def generate(self):
        # Set Elements
        
        self.tail = not self.wing_only
        self.n_elem_junction_main = int(0.5*self.n_elem_multiplier)
        if self.n_elem_junction_main < 1:
            self.n_elem_junction_main = 1
        self.n_elem_root_main = int(1*self.n_elem_multiplier)
        self.n_elem_tip_main = int(0.5*self.n_elem_multiplier)
        if self.n_elem_tip_main < 1:
            self.n_elem_tip_main = 1
        self.n_elem_per_aileron = int(4*self.n_elem_multiplier)
        self.n_elem_main = int(self.n_elem_junction_main + self.n_elem_root_main + self.n_ailerons_per_wing * self.n_elem_per_aileron + self.n_elem_tip_main)

        self.n_elem_per_elevator =  int(3*self.n_elem_multiplier_tail)
        self.n_elem_junction_tail = int(2*self.n_elem_multiplier_tail)
        self.n_elem_tail = int(self.n_elem_junction_tail + self.n_elev_per_tail_surf * self.n_elem_per_elevator)
        self.n_elem_fuselage = int(10*self.n_elem_multiplier_fuselage) + 1

        # lumped masses
        df_lumped_masses = self.read_lumped_masses()
        n_lumped_mass_wing = df_lumped_masses.shape[0]
        n_lumped_mass = n_lumped_mass_wing * 2
        if not self.wing_only:
            n_lumped_mass += 2 #for payload, engine, fuel, system
        self.lumped_mass_nodes = np.zeros((n_lumped_mass, ), dtype=int)
        self.lumped_mass = np.zeros((n_lumped_mass, ))
        self.lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
        self.lumped_mass_position = np.zeros((n_lumped_mass, 3))
        
        # total number of elements
        self.n_elem = self.n_elem_main + self.n_elem_main
        if not self.lifting_only or self.tail:
            self.n_elem += self.n_elem_fuselage
            if self.tail:
                self.n_elem += self.n_elem_tail + self.n_elem_tail

        # number of nodes per part
        self.n_node_main = self.n_elem_main*(self.n_node_elem - 1) + 1
        self.n_node_tail = self.n_elem_tail*(self.n_node_elem - 1) + 1
        self.n_node_fuselage = (self.n_elem_fuselage+1)*(self.n_node_elem - 1) -1

        # total number of nodes
        self.n_node = self.n_node_main + self.n_node_main - 1
        if not self.lifting_only or self.tail:
            self.n_node += self.n_node_fuselage - 1
            if self.tail:
                self.n_node += self.n_node_tail - 1
                self.n_node += self.n_node_tail - 1


        # Aeroelastic properties
        n_stiffness = self.n_stiffness_per_wing * 2
        n_mass = self.n_elem_main * 2
        if not self.lifting_only or self.tail:
            n_stiffness += 1
            n_mass += 1
            if self.tail:
                n_stiffness += 1
                n_mass += 1

        m_bar_fuselage = 0.3 * 10
        j_bar_fuselage = 0.08

        sigma_tail = 100
        m_bar_tail = 0.3 * 4
        j_bar_tail = 0.08

        # beam
        self.x = np.zeros((self.n_node, ))
        self.y = np.zeros((self.n_node, ))
        self.z = np.zeros((self.n_node, ))
        structural_twist = np.zeros((self.n_elem, self.n_node_elem))
        self.beam_number = np.zeros((self.n_elem, ), dtype=int)
        frame_of_reference_delta = np.zeros((self.n_elem, self.n_node_elem, 3))
        self.conn = np.zeros((self.n_elem, self.n_node_elem), dtype=int)
        stiffness = np.zeros((n_stiffness, 6, 6))
        self.elem_stiffness = np.zeros((self.n_elem, ), dtype=int)
        self.mass = np.zeros((n_mass, 6, 6))
        self.elem_mass = np.zeros((self.n_elem, ), dtype=int)
        boundary_conditions = np.zeros((self.n_node, ), dtype=int)
        app_forces = np.zeros((self.n_node, 6))
        self.elastic_axis = np.zeros((self.n_elem, self.n_node_elem,))





        list_spanwise_shear_center = self.read_spanwise_shear_center()
        y_cross_sections = self.load_y_cross_sections()
        # Load data from file
        if self.material == "reference":
            file = self.source_directory + '/dynamics_reference.mat'
        else:
            file = self.source_directory + '/dynamics_tailored.mat'

        matlab_data = load_mat(file)
        nodal_coordinates = matlab_data['dynamics']['str']['xyz']
        N_nodes = int(matlab_data['dynamics']['str']['Nnode'])

        # Transform data
        coords = np.zeros((N_nodes, 3))
        counter = 0
        for irow in range(N_nodes):
            # skip first row
            coords[irow, :] = np.transpose(nodal_coordinates[counter:counter+3])
            counter += 3

        ###############
        # right wing
        ###############
        we = 0
        wn = 0
        self.beam_number[we:we + self.n_elem_main] = 0
        # junction (part without ailerons)
        n_node_junctions = int(3 + 2*(self.n_elem_junction_main-1))
        self.y[wn:wn + n_node_junctions] = np.linspace(0.0, y_coord_junction, n_node_junctions)
        n_node_root = int(3 + 2*(self.n_elem_root_main-1))
        self.y[wn + n_node_junctions:wn + n_node_junctions+n_node_root-1] = np.linspace(y_coord_junction, y_coord_ailerons[0], n_node_root)[1:]
        
        n_nodes_per_cs = (self.n_elem_per_aileron)*2+1
        wn_end = 0
        n_node_tip = int(3 + 2*(self.n_elem_tip_main-1))
        for i_control_surface in range(self.n_ailerons_per_wing):
            wn_start = wn +  n_node_junctions -1 + n_node_root- 1 + i_control_surface*(n_nodes_per_cs-1)
            wn_end= wn_start + n_nodes_per_cs
            self.y[wn_start:wn_end] = np.linspace(y_coord_ailerons[i_control_surface], 
                                            y_coord_ailerons[i_control_surface+1], 
                                            n_nodes_per_cs)  
        # Aileron to tip
        self.y[wn_end:wn_end + n_node_tip-1] = np.linspace(y_coord_ailerons[-1], self.span_main*0.5, n_node_tip)[1:]
        self.x[wn+n_node_junctions:wn + self.n_node_main] += (abs(self.y[wn+n_node_junctions:wn + self.n_node_main])-y_coord_junction) * np.tan(self.sweep_quarter_chord)
    
        # Set stiffness, mass. For, and elastic axis
        self.elem_stiffness[we:we + self.n_elem_main] = 0
        self.elem_mass[we:we + self.n_elem_main] = 0
        node_counter = wn
        for ielem in range(self.n_elem_main):
            self.conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(self.n_node_elem - 1)) +
                                [0, 2, 1])          
            # if all nodes of an element are aero_nodes, the element has aero material properties
            index_position= self.find_index_of_closest_entry(y_cross_sections,self.y[node_counter+ 2])
            
            if y_cross_sections[index_position] < self.y[node_counter+ 2]:
                i_material = index_position + 1
            else:
                i_material = index_position
            
            self.elem_stiffness[we + ielem] = i_material
            self.elem_mass[we + ielem] = we + ielem
            for inode in range(self.n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]  
                self.elastic_axis[ielem, inode] = list_spanwise_shear_center[self.elem_stiffness[ielem]]
            node_counter += 2

        app_forces[wn] = [0, self.thrust, 0, 0, 0, 0]
        
        boundary_conditions[0] = 1
        ###############
        # left wing
        ###############
        we += self.n_elem_main
        wn += self.n_node_main

        # outer right wing

        self.beam_number[we:we + self.n_elem_main] = 1
        # Mirror coordinates from left wing
        self.y[wn:wn + self.n_node_main - 1] = -self.y[1:self.n_node_main]
        self.x[wn:wn + self.n_node_main - 1] = self.x[1:self.n_node_main]
        self.z[wn:wn + self.n_node_main - 1] = self.z[1:self.n_node_main]


        self.elem_stiffness[we:we + self.n_elem_main] = 0
        self.elem_mass[we:we + self.n_elem_main] = 0


        for ielem in range(self.n_elem_main):
            self.conn[we + ielem, :] = ((np.ones((3, ))*(we+ielem)*(self.n_node_elem - 1)) +
                                [0, 2, 1])
            self.elem_stiffness[we + ielem] =self.elem_stiffness[ielem] + self.n_stiffness_per_wing
            self.elem_mass[we + ielem] = we + ielem
        
            for inode in range(self.n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0] 
                self.elastic_axis[we + ielem, inode] = self.elastic_axis[ielem, inode]

        self.conn[we, 0] = 0
        boundary_conditions[wn-1] = -1 # tip right wing
        we += self.n_elem_main
        wn += self.n_node_main - 1
        boundary_conditions[wn-1] = -1 # tip left wing

            
        if not self.lifting_only or self.tail:   
            # remember this is in B FoR
            self.beam_number[we:we + self.n_elem_fuselage] = 2
            x_fuselage = np.linspace(0.0, length_fuselage, self.n_node_fuselage) - offset_wing_nose
            z_fuselage = np.linspace(0.0, offset_fuselage_vertical, self.n_node_fuselage)
            idx_junction = self.find_index_of_closest_entry(x_fuselage, self.x[0])
            x_fuselage = np.delete(x_fuselage, idx_junction)
            z_fuselage = np.delete(z_fuselage, idx_junction)
            self.x[wn:wn + self.n_node_fuselage-1] = x_fuselage 
            self.z[wn:wn + self.n_node_fuselage-1] = z_fuselage

            for ielem in range(self.n_elem_fuselage):
                self.conn[we + ielem, :] = ((np.ones((3,))*(we + ielem)*(self.n_node_elem - 1)) +
                                    2 + [0, 2, 1]) - 1

                for inode in range(self.n_node_elem):
                    frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
            for ielem in range(self.n_elem_fuselage):
                if (self.conn[we + ielem, :] ==  wn+idx_junction).any():
                    if (self.conn[we + ielem, 0] == wn+idx_junction):
                        # junction at nose
                        self.conn[:,:] -= 1
                        self.conn[we,0 ]= 0
                        break
                    elif (self.conn[we + ielem, 2] == wn+idx_junction):
                        # junction at center of an element
                        self.conn[we + ielem, 2] = 0
                        self.conn[we + ielem, 1] -= 1 
                        self.conn[we + ielem + 1:we + self.n_elem_fuselage, :] -= 1 
                    elif  (self.conn[we + ielem, 1] == wn+idx_junction):
                        # junction at last node of an element and first of the second one
                        self.conn[we + ielem, 1] = 0
                        self.conn[we + ielem + 1:we + self.n_elem_fuselage, :] -= 1 
                        self.conn[we + ielem + 1, 0] = 0
                    break
            boundary_conditions[wn] = - 1

            if self.tail:
                self.elem_stiffness[we:we + self.n_elem_fuselage] = n_stiffness - 2
                self.elem_mass[we:we + self.n_elem_fuselage] = n_mass - 2
            else:
                self.elem_stiffness[we:we + self.n_elem_fuselage] = n_stiffness - 1
                self.elem_mass[we:we + self.n_elem_fuselage] = n_mass - 1
            self.index_tail_start = wn + self.find_index_of_closest_entry(self.x[wn:wn + self.n_node_fuselage-1], offset_tail_nose-offset_wing_nose)
            we += self.n_elem_fuselage
            wn += self.n_node_fuselage - 1
            boundary_conditions[wn - 1] = -1
            boundary_conditions[self.index_tail_start] = 0
            if self.tail:
                ###############
                # right tail
                ###############
                self.beam_number[we:we + self.n_elem_tail] = 3
                self.x[wn:wn + self.n_node_tail - 1] = self.x[self.index_tail_start]
                wn_right_tail_start = wn
                n_node_junctions = int(3 + 2*(self.n_elem_junction_tail-1))
                self.y[wn:wn + n_node_junctions - 1] = np.linspace(0.0, y_coord_elevators[0], n_node_junctions)[1:]
                n_nodes_per_cs = (self.n_elem_per_elevator)*2+1

                for i_control_surface in range(self.n_elev_per_tail_surf):
                    wn_start = wn +  n_node_junctions - 1 + i_control_surface*(n_nodes_per_cs-1) -1
                    wn_end= wn_start + n_nodes_per_cs
                    self.y[wn_start:wn_end] = np.linspace(y_coord_elevators[i_control_surface], 
                                                        y_coord_elevators[i_control_surface+1], 
                                                        n_nodes_per_cs)[:]                
                                        
                self.x[wn:wn + self.n_node_tail - 1]  += abs(self.y[wn:wn + self.n_node_tail - 1])* np.tan(self.tail_sweep_quarter_chord)
                self.z[wn:wn + self.n_node_tail - 1] = self.z[self.index_tail_start]
                self.z[wn:wn + self.n_node_tail - 1] += self.y[wn:wn + self.n_node_tail - 1] * np.tan(self.v_tail_angle)

                self.elem_stiffness[we:we + self.n_elem_tail] = n_stiffness - 1
                self.elem_mass[we:we + self.n_elem_tail] = n_mass - 1
                for ielem in range(self.n_elem_tail):
                    self.conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(self.n_node_elem - 1)) +
                                        [0, 2, 1])
                    for inode in range(self.n_node_elem):
                        frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]     
                self.conn[we, 0] =  self.index_tail_start 

                we += self.n_elem_tail
                wn += self.n_node_tail - 1

                boundary_conditions[wn -1] = - 1#+ self.n_node_tail - 2] = -1
                ###############
                # left tail
                ###############
                self.beam_number[we:we + self.n_elem_tail] = 4

                self.y[wn:wn + self.n_node_tail - 1] = -self.y[wn-self.n_node_tail+1:wn]
                self.x[wn:wn + self.n_node_tail - 1] = self.x[wn-self.n_node_tail+1:wn]
                self.z[wn:wn + self.n_node_tail - 1] = self.z[wn-self.n_node_tail+1:wn]

        
                self.elem_stiffness[we:we + self.n_elem_tail] = n_stiffness - 1
                self.elem_mass[we:we + self.n_elem_tail] = n_mass - 1
                for ielem in range(self.n_elem_tail):
                    self.conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(self.n_node_elem - 1)) +
                                        [0, 2, 1])
                    for inode in range(self.n_node_elem):
                        frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]

                self.conn[we, 0] =  self.index_tail_start 
                boundary_conditions[-1] = -1
                we += self.n_elem_tail
                wn += self.n_node_tail - 1

        # lumped masses 
        self.place_lumped_masses_wing(df_lumped_masses, n_lumped_mass_wing)


        if self.lifting_only and self.tail:
            # map payload to fuselage node
            self.lumped_mass[-1] = 42 - 10.799 - 0.35833756498172# payload, kg
            self.lumped_mass_position[-1, 0] = 0 
            self.lumped_mass_position[-1, 1] = 0
            self.lumped_mass_position[-1, 2] = -0.25
            x_lm_payload = 0.2170 + self.dx_payload
            wn_fuselage_start = self.n_node_main  * 2- 1
            self.lumped_mass_nodes[-1] = wn_fuselage_start +  self.find_index_of_closest_entry(self.x[wn_fuselage_start:wn_fuselage_start + self.n_node_fuselage], x_lm_payload)
            self.lumped_mass_position[-1, 0] = x_lm_payload - self.x[self.lumped_mass_nodes[-1]]
        
        # Stiffness and mass properties
        list_stiffness_matrix, list_mass_matrix, y_cross_sections = self.load_stiffness_and_mass_matrix_from_matlab_file()
        for i in  range(int(self.n_stiffness_per_wing * 2)):
            stiffness[i, ...] = list_stiffness_matrix[i] 
            self.mass[i, ...] =  list_mass_matrix[i]
        for i in range(int(self.n_elem_main * 2)):
            self.mass[i, ...] =  list_mass_matrix[i]
        if not self.lifting_only or self.tail:
            ea = 1e7
            ga = 1e5
            gj = 1e4
            eiy = 2e4
            eiz = 4e6
            if self.tail:
                stiffness[-2, ...] = np.diag([ea, ga, ga, gj, eiy, eiz])*sigma_fuselage
                stiffness[-1, ...] = np.diag([ea, ga, ga, gj, eiy, eiz])*sigma_tail

                self.mass[-2, ...] = self.generate_mass_matrix(m_bar_fuselage, j_bar_fuselage)
                self.mass[-1, ...] = self.generate_mass_matrix(m_bar_tail, j_bar_tail)

            else:
                stiffness[-1, ...] = np.diag([ea, ga, ga, gj, eiy, eiz])*sigma_fuselage
                self.mass[-1, ...] = self.generate_mass_matrix(m_bar_fuselage, j_bar_fuselage)
                
        with h5.File(self.route + '/' + self.case_name + '.fem.h5', 'a') as h5file:
            h5file.create_dataset('coordinates', data=np.column_stack((self.x, self.y, self.z)))
            h5file.create_dataset('connectivities', data=self.conn)
            h5file.create_dataset('num_node_elem', data=self.n_node_elem)
            h5file.create_dataset('num_node', data=self.n_node)
            h5file.create_dataset('num_elem', data=self.n_elem)
            h5file.create_dataset('stiffness_db', data=stiffness)
            h5file.create_dataset('elem_stiffness', data=self.elem_stiffness)
            h5file.create_dataset('mass_db', data=self.mass)
            h5file.create_dataset('elem_mass', data=self.elem_mass)
            h5file.create_dataset('frame_of_reference_delta', data=frame_of_reference_delta)
            h5file.create_dataset('structural_twist', data=structural_twist)
            h5file.create_dataset('boundary_conditions', data=boundary_conditions)
            h5file.create_dataset('beam_number', data=self.beam_number)
            h5file.create_dataset('app_forces', data=app_forces)
            h5file.create_dataset('lumped_mass_nodes', data=self.lumped_mass_nodes)
            h5file.create_dataset('lumped_mass', data=self.lumped_mass)
            h5file.create_dataset('lumped_mass_inertia', data=self.lumped_mass_inertia)
            h5file.create_dataset('lumped_mass_position', data=self.lumped_mass_position)

    def place_lumped_masses_wing(self, df_lumped_masses, n_lumped_mass_wing):
            for imass in range(n_lumped_mass_wing):                
                self.lumped_mass[imass] =  df_lumped_masses.iloc[imass, 0]
                self.lumped_mass_nodes[imass] = self.find_index_of_closest_entry(self.y[:self.n_node_main], 
                                                                                 df_lumped_masses.iloc[imass,1])
                self.set_lumped_mass_position_B_frame(imass,
                                                      np.array(df_lumped_masses.iloc[imass, 1:4]),
                                                      self.lumped_mass_nodes[imass])
                # mirror lumped masses for left wing
                idx_symmetric = n_lumped_mass_wing + imass
                self.lumped_mass[idx_symmetric] =  self.lumped_mass[imass]                 
                if self.lumped_mass_nodes[imass] == 0:
                    self.lumped_mass_nodes[idx_symmetric] = 0
                else:
                    self.lumped_mass_nodes[idx_symmetric] = self.n_node_main - 1 + self.lumped_mass_nodes[imass]
                
                self.lumped_mass[idx_symmetric] =  self.lumped_mass[imass]
                self.lumped_mass_position[idx_symmetric, 0] = self.lumped_mass_position[imass, 0]
                self.lumped_mass_position[idx_symmetric, 1] = -self.lumped_mass_position[imass, 1]
                self.lumped_mass_position[idx_symmetric, 2] = self.lumped_mass_position[imass, 2] 

                    
    def set_lumped_mass_position_B_frame(self, imass, position, inode):
        """
        This function converts the lumped mass position given in the G frame into the local coordinate of the linked structural node.

        """
        position[0] -= 0.22
        self.lumped_mass_position[imass, 2] = position[2]
        self.lumped_mass_position[imass, 0] = position[1] - self.y[inode]
        self.lumped_mass_position[imass, 1] = position[0] - self.x[inode]
        if self.y[inode] > self.y_coord_junction:
            # local COS rotated around z-axis by beam sweep angle
            self.lumped_mass_position[imass, 0] /=  np.cos(self.sweep_quarter_chord)
            self.lumped_mass_position[imass, 1] -= self.lumped_mass_position[imass, 0] * np.sin(self.sweep_quarter_chord)
            self.lumped_mass_position[imass, 1] *= np.cos(self.sweep_quarter_chord)

    def load_y_cross_sections(self):
        # Load data from file
        if self.material == "reference":
            file = self.source_directory + '/dynamics_reference.mat'
        else:
            file = self.source_directory + '/dynamics_tailored.mat'

        matlab_data = load_mat(file)
        nodal_coordinates = matlab_data['dynamics']['str']['xyz']
        N_nodes = int(matlab_data['dynamics']['str']['Nnode'])

        # Transform data
        coords = np.zeros((N_nodes, 3))
        counter = 0
        for irow in range(N_nodes):
            # skip first row
            coords[irow, :] = np.transpose(nodal_coordinates[counter:counter+3])
            counter += 3
        return coords[1:,1]

    def load_stiffness_and_mass_matrix_from_matlab_file(self):
        # Load data from file
        if self.material == "reference":
            file = self.source_directory + '/dynamics_reference.mat'
        else:
            file = self.source_directory + '/dynamics_tailored.mat'

        matlab_data = load_mat(file)
        matrices_cross_stiffness = matlab_data['dynamics']['str']['elm']['C'] * self.sigma
        matrices_cross_mass = matlab_data['dynamics']['str']['elm']['A']
        matrices_cross_moment_of_inertia = matlab_data['dynamics']['str']['elm']['I']
        matrices_cross_first_moment = matlab_data['dynamics']['str']['elm']['Q']
        nodal_coordinates = matlab_data['dynamics']['str']['xyz']
        N_nodes = int(matlab_data['dynamics']['str']['Nnode'])

        # Transform data
        coords = np.zeros((N_nodes, 3))
        counter = 0
        for irow in range(N_nodes):
            # skip first row
            coords[irow, :] = np.transpose(nodal_coordinates[counter:counter+3])
            counter += 3

        list_stiffness_matrix = []
        list_mass_matrix_data = []
        list_mass_matrix = []
        list_Jy = []

        counter = 0
        inertia_counter = 0
        row_counter = 0
        #### Stiffness ####
        # Right wing
        while counter < matrices_cross_stiffness.shape[0]:
            # list_stiffness_matrix.append(np.diag(np.diagonal(np.array(matrices_cross_stiffness[counter:counter+6, :]))))
            tmp_stiffness_matrix = np.array(matrices_cross_stiffness[counter:counter+6, :])
            tmp_stiffness_matrix[5, 5] /= self.sigma/2
            list_stiffness_matrix.append(tmp_stiffness_matrix)
            
            mass_matrix = np.zeros((6,6))
            # mass distribution
            mass = float(matrices_cross_mass[row_counter])
            for i in range(3):
                mass_matrix[i,i] = mass
            mass_matrix[3:,3:] = matrices_cross_moment_of_inertia[inertia_counter:inertia_counter+3,:3]
            
            mass_matrix[3:,:3] = self.get_first_moment_matrix(0, 
                                                              matrices_cross_first_moment[row_counter,1], 
                                                              - matrices_cross_first_moment[row_counter,0])
            list_Jy.append(matrices_cross_first_moment[row_counter,1])
            mass_matrix[:3,3:] = -mass_matrix[3:,:3]
            
            # list_mass_matrix_data.append(np.diag(np.diagonal(mass_matrix)))
            list_mass_matrix_data.append(mass_matrix)
            counter += 6
            inertia_counter += 3
            row_counter += 1

        # # left wing
        for i_material in range(self.n_stiffness_per_wing):
            stiffness_matrix = list_stiffness_matrix[i_material].copy()
            stiffness_matrix[2,3] *= -1
            stiffness_matrix[3,2] *= -1

            stiffness_matrix[4,5] *= -1
            stiffness_matrix[5,4] *= -1

            stiffness_matrix[1,2] *= -1
            stiffness_matrix[2,1] *= -1
            
            stiffness_matrix[0,5] *= -1
            stiffness_matrix[5,0] *= -1

            list_stiffness_matrix.append(stiffness_matrix)
 
        for ielem in range(self.n_elem_main):
            mass_matrix = list_mass_matrix_data[self.elem_stiffness[ielem]].copy()
            mass_matrix[3:,:3] = self.correct_first_moment(mass_matrix[3:,:3], 
                                                           self.get_chord(self.y[ielem * 2]), 
                                                           mass_matrix[0,0], 
                                                           self.elastic_axis[ielem, 2],
                                                           self.y[ielem * 2])
            mass_matrix[:3,3:] = -mass_matrix[3:,:3]
            list_mass_matrix.append(mass_matrix)


        for ielem in range(self.n_elem_main):
            mass_matrix = list_mass_matrix[ielem].copy()
            # cg x component mirror in upper right partition
            mass_matrix[1, 5] *= -1
            mass_matrix[2, 4] *= -1

            # cg x component mirror in lower left partition
            mass_matrix[5, 1] *= -1
            mass_matrix[4, 2] *= -1

            # cg y component mirror in upper right partition
            mass_matrix[0, 5] *= -1
            mass_matrix[2, 3] *= -1

            # cg y component mirror in lower left partition
            mass_matrix[5, 0] *= -1
            mass_matrix[3, 2] *= -1          

            # 45 - Iyz -
            mass_matrix[4, 5] *= -1
            mass_matrix[5, 4] *= -1

            list_mass_matrix.append(mass_matrix)

        return list_stiffness_matrix, list_mass_matrix, coords[1:,1]   

    def get_chord(self, y):
        if y <= self.y_coord_junction:
            return chord_main_root
        else:
            y -= self.y_coord_junction
            x_LE = np.tan(sweep_LE_main) * y
            x_TE = chord_main_root + np.tan(sweep_TE_main) * y
            return abs(x_LE - x_TE)

    def generate_mass_matrix(self, m_bar, j_bar):
        return np.diag([m_bar, m_bar, m_bar, 
                        j_bar, 0.5*j_bar, 0.5*j_bar])

    def find_index_of_closest_entry(self, array_values, target_value):
        return np.argmin(np.abs(array_values - target_value))

    def read_lumped_masses(self):
        file = self.source_directory + '/lumped_masses.csv'
        df = pd.read_csv(file, sep=';', header = None)
        return df

    def get_first_moment_matrix(self, Jx,Jy,Jz):
        matrix = np.zeros((3,3))
        matrix[0,1] = -Jz
        matrix[1,0] = +Jz
        matrix[0,2] = +Jy
        matrix[2,0] = -Jy
        matrix[1,2] = -Jx
        matrix[2,1] = +Jx
        return matrix

    def correct_first_moment(self, J_matrix, chord, mass_elem, c_n, y):
        # Correct y coordinate for definition from reference frame
        Jy = J_matrix[0,2]
        chi_y = ((0.71 * chord) + J_matrix[0,2] / mass_elem) - c_n * chord
        Jy = chi_y * mass_elem
        J_matrix[0,2] = +Jy
        J_matrix[2,0] = -Jy
        return J_matrix

    def calculate_aircraft_mass(self):
        # get structural mass for each component (beam ID)
        list_elem_mass = []
        center_of_gravity = np.zeros((3, ))
        for i_elem in range(self.n_elem):
            start_node = self.conn[i_elem, 0]
            end_node = self.conn[i_elem, 1]
            # calculate length assuming that elem is straight (unloaded)
            length_elem = np.sqrt((self.x[start_node]-self.x[end_node])**2 
                                  + (self.y[start_node]-self.y[end_node])**2 
                                  +(self.z[start_node]-self.z[end_node])**2 )
            distance = [self.x[self.conn[i_elem, -1]],
                        self.y[self.conn[i_elem, -1]],
                        self.z[self.conn[i_elem, -1]]]
            distributed_mass_elem = self.mass[self.elem_mass[i_elem], 0, 0]
            mass_elem = distributed_mass_elem * length_elem
            list_elem_mass.append(mass_elem)
            
            for i_dim in range(3):
                center_of_gravity[i_dim] += mass_elem * distance[i_dim]
        total_mass_structure = sum(list_elem_mass)
        
        print("Total structural mass = ", total_mass_structure)
        for i_beam in set(self.beam_number):
            structural_mass_beam = sum(np.array(list_elem_mass)[self.beam_number == int(i_beam)])
            print("Total structural mass for beam {} is {} kg".format(i_beam, structural_mass_beam))
        
        # Get lumped masses

        df_lumped_masses = self.read_lumped_masses()
        n_lumped_masses_wing = df_lumped_masses.shape[0]
        
        for i_mass in range(len(self.lumped_mass)):      
                  
            if i_mass < n_lumped_masses_wing:
                position_G_frame = np.array(df_lumped_masses.iloc[i_mass, 1:4])
                position_G_frame[0] -= 0.24
                center_of_gravity[:] += np.dot(self.lumped_mass[i_mass], position_G_frame)
                # considered mirrored wing
                position_G_frame[1] *= -1
                center_of_gravity[:] += np.dot(self.lumped_mass[i_mass], position_G_frame)
            elif i_mass < n_lumped_masses_wing * 2:
                # Skip left wing as it contributions to the cg has been calculated before
                pass
            else:

                print("imass {} with weight {}".format(i_mass, self.lumped_mass[i_mass]))
                
                if self.lumped_mass[i_mass] > 0:
                    print("imass = ", i_mass)
                    center_of_gravity[0] +=  self.lumped_mass[i_mass] * (self.lumped_mass_position[i_mass, 0] + self.x[self.lumped_mass_nodes[i_mass]])
                    center_of_gravity[1] +=  self.lumped_mass[i_mass] * (self.lumped_mass_position[i_mass, 1] + self.y[self.lumped_mass_nodes[i_mass]])
                    center_of_gravity[2] +=  self.lumped_mass[i_mass] * (self.lumped_mass_position[i_mass, 2] + self.z[self.lumped_mass_nodes[i_mass]])

        total_mass_lumped_masses = sum(self.lumped_mass)
        total_mass = total_mass_lumped_masses + total_mass_structure
        center_of_gravity /= total_mass
        
        print("x nose = ", min(self.x))
        print("x junction = ", self.x[0])
        print("x tail = ", max(self.x))
        print("Total lumped masses ", sum(self.lumped_mass))
        print("Total mass aircraft = ", total_mass)
        print("Center of Gravity = ", center_of_gravity)
        print("Center of Gravity - difference = ", center_of_gravity - [0.606 + 0.8769 +   min(self.x), -0.1, -0.25])

    def read_spanwise_shear_center(self):
        reference_shear_center = 0.71 # given by Jurij
        df = pd.read_csv(self.source_directory + '/shear_center.csv',
                                sep=';')
        if self.material == "reference":
            column = 1
        else:
            column = 2
        print(self.material, column)
        return (reference_shear_center + df.iloc[:,column]).to_list()

def load_mat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    References:
        This glorious tool was obtained from:
        https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries/29126361#29126361
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list)
        else:
            return ndarray

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)
