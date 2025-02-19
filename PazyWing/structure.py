import h5py as h5
import numpy as np
import pandas as pd
import os
from sharpy.utils import algebra as algebra


class PazyStructure:

    def __init__(self, **kwargs):
        # settings
        local_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        self.source_path = local_path + '/src/'
        self.skin = kwargs.get('skin_on', False)
        self.discretisation_method = kwargs.get('discretisation_method', 'michigan')
        self.init_discretisation = kwargs.get('num_elem', 2)

        self.mirrored = False
        self.sweep_angle = -kwargs.get('sweep_angle', 0.0)
        self.sigma = kwargs.get('sigma', 1.0)

        # coordinates
        self.x = None
        self.y = None
        self.z = None

        # inertia
        self.mass_db = None
        self.elem_mass = None
        self.lumped_mass = None
        self.lumped_mass_nodes = None
        self.lumped_mass_position = None
        self.lumped_mass_inertia = None

        # stiffness
        self.stiffness_db = None
        self.elem_stiffness = None

        # FEM
        self.n_node = None
        self.n_elem = None
        self.num_node_elem = 3
        self.connectivities = None
        self.frame_of_reference_delta = None
        self.boundary_conditions = None
        self.beam_number = None
        self.structural_twist = None
        self.app_forces = None

        self.source = dict()

        self.debug = False

    def generate(self):
        self.coordinates(method_tuple=(self.discretisation_method, self.init_discretisation))
        self.load_mass()
        self.load_stiffness()

        self.create_fem()

    def coordinates(self, method_tuple):

        method, discretisation = method_tuple
        coords_file = self.source_path + 'coordinates_{}_skin.xlsx'.format(self._get_skin())
        df = pd.read_excel(io=coords_file)

        x = np.array(df['x'], dtype=float)
        y = np.array(df['y'], dtype=float)
        z = np.array(df['z'], dtype=float)
        # MICHIGAN adds the reference line offset onto the x coordinate, we will do so in the aero part

        self.source['x'] = x
        self.source['y'] = y
        self.source['z'] = z

        self.source['coords'] = np.column_stack((x, y, z))

        if method == 'michigan':
            n_fact = discretisation  # doubling factor
            self.n_elem = 2 ** (n_fact - 1) * (len(x) - 1)
            self.n_node = self.n_elem * (self.num_node_elem - 1) + 1
            self.y = np.zeros(self.n_node)
            i_node = 0
            for i_um_node in range(len(y) - 1):
                self.y[i_node] = y[i_um_node]
                self.y[i_node + 1] = 0.5 * (y[i_um_node + 1] - y[i_um_node]) + y[i_um_node]

                in_between_nodes = np.linspace(y[i_um_node], y[i_um_node + 1], 1 + 2 ** n_fact)
                self.y[i_node + 1: i_node + 2 ** n_fact] = in_between_nodes[1:-1]
                self.y[i_node + 2 ** n_fact] = y[i_um_node + 1]

                i_node += 2 ** n_fact
        elif method == 'even':
            self.n_elem = discretisation
            self.n_node = self.n_elem * (self.num_node_elem - 1) + 1
            self.y = np.linspace(0, y[-1], self.n_node)
        elif method == 'fine_root_tip':
            self.n_elem = discretisation
            self.n_node = self.n_elem * (self.num_node_elem - 1) + 1
            self.y = np.zeros(self.n_node)
            init_region = 0.07
            end_region = 0.45

            idx_init = int(self.n_node // 3)
            self.y[:idx_init] = np.linspace(0, init_region, int(self.n_node // 3))
            self.y[idx_init:2 * idx_init] = np.linspace(init_region, end_region, int(self.n_node // 3) + 1)[1:]
            self.y[2 * idx_init:] = np.linspace(end_region, y[-1], len(self.y[2 * idx_init:]) + 1)[1:]
        else:
            raise NameError('Unknown discretisation method')

        self.x = np.zeros_like(self.y)
        self.z = np.zeros_like(self.y)

        # connectivities
        self.connectivities = np.zeros((self.n_elem, self.num_node_elem), dtype=int)
        self.connectivities[:, 0] = np.arange(0, self.n_node - 2, 2)
        self.connectivities[:, 1] = np.arange(2, self.n_node, 2)
        self.connectivities[:, 2] = np.arange(1, self.n_node - 1, 2)

        if self.debug:
            np.savetxt('./coords_sharpy.txt', np.column_stack((self.x, self.y, self.z)))
            np.savetxt('./coords_um.txt', self.source['coords'])

    def _get_skin(self):
        return 'w' if self.skin else 'wo'


    def load_mass(self):

        mass_file = self.source_path + 'inertia_{}_skin.xlsx'.format(self._get_skin())
        df = pd.read_excel(io=mass_file)

        nodal_mass = np.array(df['mass'], dtype=float)
        n_mass = len(nodal_mass)
        c_gb = np.zeros((n_mass, 3), dtype=float)
        c_gb[:, 1] = -df['cgx']
        c_gb[:, 0] = df['cgy']  # change indices for SHARPy B frame, original in A frame
        c_gb[:, 2] = df['cgz']

        mass_data_nodes = np.zeros((n_mass, 6, 6), dtype=float)
        mass_data_nodes[:, 0, 0] = nodal_mass
        mass_data_nodes[:, 1, 1] = nodal_mass
        mass_data_nodes[:, 2, 2] = nodal_mass

        cg_factor = 1
        for i in range(n_mass):
            mass_data_nodes[i, :3, 3:] = -algebra.skew(mass_data_nodes[i, 0, 0] * c_gb[i]) * cg_factor
            mass_data_nodes[i, 3:, :3] = algebra.skew(mass_data_nodes[i, 0, 0] * c_gb[i]) * cg_factor

        cross_term_factor = 1
        # inertia data expressed in A frame and required in B frame
        mass_data_nodes[:, 4, 4] = df['Ixx']
        mass_data_nodes[:, 3, 3] = df['Iyy']
        mass_data_nodes[:, 5, 5] = df['Izz']

        mass_data_nodes[:, 3, 4] = -df['Ixy'] * cross_term_factor
        mass_data_nodes[:, 4, 3] = -df['Ixy'] * cross_term_factor

        mass_data_nodes[:, 4, 5] = -df['Ixz'] * cross_term_factor
        mass_data_nodes[:, 5, 4] = -df['Ixz'] * cross_term_factor

        mass_data_nodes[:, 3, 5] = df['Iyz'] * cross_term_factor
        mass_data_nodes[:, 5, 3] = df['Iyz'] * cross_term_factor

        inertia_tensor = np.zeros((n_mass, 3, 3))
        inertia_tensor[:, 1, 1] = df['Ixx']
        inertia_tensor[:, 0, 0] = df['Iyy']
        inertia_tensor[:, 2, 2] = df['Izz']

        inertia_tensor[:, 0, 1] = -df['Ixy'] * cross_term_factor
        inertia_tensor[:, 1, 0] = -df['Ixy'] * cross_term_factor

        inertia_tensor[:, 1, 2] = -df['Ixz'] * cross_term_factor
        inertia_tensor[:, 2, 1] = -df['Ixz'] * cross_term_factor

        inertia_tensor[:, 0, 2] = df['Iyz'] * cross_term_factor
        inertia_tensor[:, 2, 0] = df['Iyz'] * cross_term_factor

        if self.debug:
            np.savetxt('./um_inertia.txt', np.column_stack((inertia_tensor[:, 0, 0],
                                                            inertia_tensor[:, 1, 1],
                                                            inertia_tensor[:, 2, 2])))

        # equivalent distribute mass
        um_element_ends = self.source['y']
        um_distributed_mass = np.zeros(n_mass)
        um_elem_length = np.diff(um_element_ends)

        # distributed inertia
        um_distributed_inertia = np.zeros((n_mass, 6))

        transformed_inertia = self.transform_inertia(inertia_tensor, nodal_mass, c_gb, self.source['coords'] * 0)
        list_of_inertias = [transformed_inertia[:, 0, 0],
                            transformed_inertia[:, 1, 1],
                            transformed_inertia[:, 2, 2],
                            transformed_inertia[:, 0, 1],
                            transformed_inertia[:, 0, 2],
                            transformed_inertia[:, 1, 2]]

        for i_um_node in range(1, len(um_element_ends) - 1):
            if i_um_node == 1:
                # first node
                um_distributed_mass[0] = nodal_mass[0] / (0.5 * um_elem_length[0])
                um_distributed_mass[1] = nodal_mass[1] / (0.5 * um_elem_length[0] + 0.5 * um_elem_length[1])
                for i in range(6):
                    um_distributed_inertia[i_um_node, i] = list_of_inertias[i][i_um_node] / (
                                0.5 * um_elem_length[i_um_node] + 0.5 * um_elem_length[i_um_node - 1])
                    um_distributed_inertia[0, i] = list_of_inertias[i][0] / (0.5 * um_elem_length[0])

            elif i_um_node == len(um_elem_length) - 2:
                um_distributed_mass[i_um_node] = nodal_mass[i_um_node] / (
                            0.5 * um_elem_length[i_um_node] + 0.5 * um_elem_length[i_um_node - 1])
                um_distributed_mass[-1] = nodal_mass[-1] / (0.5 * um_elem_length[-1])
                for i in range(6):
                    um_distributed_inertia[i_um_node, i] = list_of_inertias[i][i_um_node] / (
                                0.5 * um_elem_length[i_um_node] + 0.5 * um_elem_length[i_um_node - 1])
                    um_distributed_inertia[-1, i] = list_of_inertias[i][-1] / (0.5 * um_elem_length[-1])

            else:
                um_distributed_mass[i_um_node] = nodal_mass[i_um_node] / (
                            0.5 * um_elem_length[i_um_node] + 0.5 * um_elem_length[i_um_node - 1])

                for i in range(6):
                    um_distributed_inertia[i_um_node, i] = list_of_inertias[i][i_um_node] / (
                                0.5 * um_elem_length[i_um_node] + 0.5 * um_elem_length[i_um_node - 1])

        sharpy_mu_elem = np.zeros(self.n_elem)
        mid_elem = np.zeros((self.n_elem, 3))
        coords = np.column_stack((self.x, self.y, self.z))

        sharpy_inertia_elem = np.zeros((self.n_elem, 6))

        sharpy_elem_length = np.zeros(self.n_elem)
        # i_um = 0
        for i_elem in range(self.n_elem):
            mid_elem[i_elem] = coords[self.connectivities[i_elem, -1]]
            sharpy_elem_length[i_elem] = coords[self.connectivities[i_elem, 1], 1] - coords[
                self.connectivities[i_elem, 0], 1]

            sharpy_mu_elem[i_elem] = np.interp(mid_elem[i_elem, 1], self.source['y'], um_distributed_mass)

            for i in range(6):
                sharpy_inertia_elem[i_elem, i] = np.interp(mid_elem[i_elem, 1], self.source['y'],
                                                           um_distributed_inertia[:, i])

        if self.debug:
            np.savetxt('./um_nodal_mass.txt', nodal_mass)
            np.savetxt('./um_distributed_mass.txt', um_distributed_mass)
            np.savetxt('./sharpy_distributed_mass.txt', sharpy_mu_elem)
            np.savetxt('./sharpy_mid_elem_coord.txt', mid_elem)
            np.savetxt('./sharpy_elem_length.txt', sharpy_elem_length)
            np.savetxt('./um_distributed_inertia.txt', um_distributed_inertia)
            np.savetxt('./um_transformed_inertia.txt', np.column_stack(tuple(list_of_inertias)))
            np.savetxt('./sharpy_distributed_inertia.txt', sharpy_inertia_elem)

        # interpolate CG position of the beam element from the cg origin data
        cg_elem = np.zeros((self.n_elem, 3))
        mid_elem = np.zeros((self.n_elem, 3))

        self.mass_db = np.zeros((self.n_elem, 6, 6), dtype=float)
        for i_elem in range(self.n_elem):
            mid_elem[i_elem] = coords[self.connectivities[i_elem, -1]]
            for i in range(3):
                cg_elem[i_elem, i] = np.interp(mid_elem[i_elem, 1], self.source['coords'][:, 1], c_gb[:, i])

            self.mass_db[i_elem, 0, 0] = sharpy_mu_elem[i_elem]
            self.mass_db[i_elem, 1, 1] = sharpy_mu_elem[i_elem]
            self.mass_db[i_elem, 2, 2] = sharpy_mu_elem[i_elem]
            self.mass_db[i_elem, :3, 3:] = -algebra.skew(cg_elem[i_elem, :]) * sharpy_mu_elem[i_elem]
            self.mass_db[i_elem, 3:, :3] = algebra.skew(cg_elem[i_elem, :]) * sharpy_mu_elem[i_elem]

            self.mass_db[i_elem, 3, 3] = sharpy_inertia_elem[i_elem, 0]
            self.mass_db[i_elem, 4, 4] = sharpy_inertia_elem[i_elem, 1]
            self.mass_db[i_elem, 5, 5] = sharpy_inertia_elem[i_elem, 2]

            self.mass_db[i_elem, 3, 4] = sharpy_inertia_elem[i_elem, 3]
            self.mass_db[i_elem, 4, 3] = sharpy_inertia_elem[i_elem, 3]

            self.mass_db[i_elem, 4, 5] = sharpy_inertia_elem[i_elem, 5]
            self.mass_db[i_elem, 5, 4] = sharpy_inertia_elem[i_elem, 5]

            self.mass_db[i_elem, 3, 5] = sharpy_inertia_elem[i_elem, 4]
            self.mass_db[i_elem, 5, 3] = sharpy_inertia_elem[i_elem, 4]

        if self.debug:
            np.savetxt('./cg_sharpy.txt', np.column_stack((mid_elem[:, 1], cg_elem)))
            np.savetxt('./cg_um.txt', np.column_stack((self.source['y'], c_gb)))

    def transform_inertia(self, inertia_tensor_array, m_node, cg, r_node):

        n_node = inertia_tensor_array.shape[0]

        transformed_inertia = np.zeros_like(inertia_tensor_array)

        for i_node in range(n_node):
            d = r_node[i_node] - cg[i_node]
            m = m_node[i_node]
            transformed_inertia[i_node, :] = self.parallel_axes(inertia_tensor_array[i_node, :], m, d)

        return transformed_inertia

    @staticmethod
    def parallel_axes(ic, m, d):
        return ic - m * algebra.skew(d).dot(algebra.skew(d))

    def load_stiffness(self):

        stiffness_file = self.source_path + 'stiffness_{}_skin.xlsx'.format(self._get_skin())
        df = pd.read_excel(io=stiffness_file)

        ea = df['K11']
        n_stiffness = len(ea)  # stiffness per element, mass was per node
        um_stiffness = np.zeros((n_stiffness, 6, 6), dtype=float)

        if self.debug:
            np.savetxt('./um_stiffness.txt',
                       np.column_stack((df['K11'], df['K22'],  df['K33'], df['K44'], df['K12'],
                                        df['K13'], df['K14'], df['K23'], df['K24'], df['K34'])))

        ga = 3e6
        um_stiffness[:, 0, 0] = ea
        um_stiffness[:, 1, 1] = ga
        um_stiffness[:, 2, 2] = ga
        um_stiffness[:, 3, 3] = df['K22']
        um_stiffness[:, 4, 4] = df['K33']
        um_stiffness[:, 5, 5] = df['K44']

        # cross terms

        cross_term_factor = 1
        um_stiffness[:, 0, 3] = df['K12'] * cross_term_factor
        um_stiffness[:, 3, 0] = df['K12'] * cross_term_factor

        um_stiffness[:, 0, 4] = df['K13'] * cross_term_factor
        um_stiffness[:, 4, 0] = df['K13'] * cross_term_factor

        um_stiffness[:, 0, 5] = df['K14'] * cross_term_factor
        um_stiffness[:, 5, 0] = df['K14'] * cross_term_factor

        um_stiffness[:, 3, 4] = df['K23'] * cross_term_factor
        um_stiffness[:, 4, 3] = df['K23'] * cross_term_factor

        um_stiffness[:, 3, 5] = df['K24'] * cross_term_factor
        um_stiffness[:, 5, 3] = df['K24'] * cross_term_factor

        um_stiffness[:, 4, 5] = df['K34'] * cross_term_factor
        um_stiffness[:, 5, 4] = df['K34'] * cross_term_factor

        self.stiffness_db = np.zeros((self.n_elem, 6, 6))
        coords = np.column_stack((self.x, self.y, self.z))
        mid_elem = np.zeros((self.n_elem, 3))

        mid_elem_um = np.zeros(n_stiffness)
        for i_um_elem in range(1, n_stiffness + 1):
            mid_elem_um[i_um_elem - 1] = 0.5 * (self.source['y'][i_um_elem] - self.source['y'][i_um_elem - 1]) + \
                                         self.source['y'][i_um_elem - 1]

        if self.debug:
            np.savetxt('./um_mid_elem.txt', mid_elem_um)

        for i_elem in range(self.n_elem):
            mid_elem[i_elem] = coords[self.connectivities[i_elem, -1]]
            self.stiffness_db[i_elem, 0, 0] = np.interp(mid_elem[i_elem, 1], mid_elem_um, um_stiffness[:, 0, 0])
            self.stiffness_db[i_elem, 1, 1] = ga
            self.stiffness_db[i_elem, 2, 2] = ga
            self.stiffness_db[i_elem, 3, 3] = np.interp(mid_elem[i_elem, 1], mid_elem_um, um_stiffness[:, 3, 3])
            self.stiffness_db[i_elem, 4, 4] = np.interp(mid_elem[i_elem, 1], mid_elem_um, um_stiffness[:, 4, 4])
            self.stiffness_db[i_elem, 5, 5] = np.interp(mid_elem[i_elem, 1], mid_elem_um, um_stiffness[:, 5, 5])

            self.stiffness_db[i_elem, 0, 3] = np.interp(mid_elem[i_elem, 1], mid_elem_um, um_stiffness[:, 0, 3])
            self.stiffness_db[i_elem, 0, 4] = np.interp(mid_elem[i_elem, 1], mid_elem_um, um_stiffness[:, 0, 4])
            self.stiffness_db[i_elem, 0, 5] = np.interp(mid_elem[i_elem, 1], mid_elem_um, um_stiffness[:, 0, 5])
            self.stiffness_db[i_elem, 3, 0] = self.stiffness_db[i_elem, 0, 3]
            self.stiffness_db[i_elem, 4, 0] = self.stiffness_db[i_elem, 0, 4]
            self.stiffness_db[i_elem, 5, 0] = self.stiffness_db[i_elem, 0, 5]

            self.stiffness_db[i_elem, 3, 4] = np.interp(mid_elem[i_elem, 1], mid_elem_um, um_stiffness[:, 3, 4])
            self.stiffness_db[i_elem, 3, 5] = np.interp(mid_elem[i_elem, 1], mid_elem_um, um_stiffness[:, 3, 5])
            self.stiffness_db[i_elem, 4, 3] = self.stiffness_db[i_elem, 3, 4]
            self.stiffness_db[i_elem, 5, 3] = self.stiffness_db[i_elem, 3, 5]

            self.stiffness_db[i_elem, 4, 5] = np.interp(mid_elem[i_elem, 1], mid_elem_um, um_stiffness[:, 4, 5])
            self.stiffness_db[i_elem, 5, 4] = self.stiffness_db[i_elem, 4, 5]

    def add_lumped_mass(self, items):

        if type(items) is tuple:
            self._new_lumped_mass(*items)
        elif type(items) is list:
            for mass_items in items:
                self._new_lumped_mass(*mass_items)

    def _new_lumped_mass(self, mass, node, inertia=np.zeros((3, 3)), position=np.zeros(3)):
        if inertia is None:
            inertia = np.zeros((3, 3))
        if position is None:
            position = np.zeros(3)

        if self.lumped_mass is None:
            self.lumped_mass = np.array([mass])
            self.lumped_mass_nodes = np.array([node], dtype=int)
            inertia.shape = (1, 3, 3)
            self.lumped_mass_inertia = inertia
            position.shape = (1, 3)
            self.lumped_mass_position = position
        else:
            self.lumped_mass = np.concatenate([self.lumped_mass, np.array([mass])])
            self.lumped_mass_nodes = np.concatenate([self.lumped_mass_nodes, np.array([node])])
            position.shape = (1, 3)
            self.lumped_mass_position = np.vstack((self.lumped_mass_position, position))
            inertia.shape = (1, 3, 3)
            self.lumped_mass_inertia = np.concatenate((self.lumped_mass_inertia, inertia), axis=0)

    def create_fem(self):

        # stiffness assignment
        self.elem_stiffness = np.zeros(self.n_elem, dtype=int)
        self.elem_stiffness[:] = np.arange(0, self.n_elem)

        # mass assignment
        # CAUTION: mass data is in element-end format, SHARPy uses per element
        self.elem_mass = np.zeros(self.n_elem, dtype=int)
        self.elem_mass[:] = np.arange(0, self.n_elem)

        # frame of reference delta
        frame_of_reference_delta = np.zeros((self.n_elem, self.num_node_elem, 3))
        frame_of_reference_delta[:, :, 0] = -1

        self.frame_of_reference_delta = frame_of_reference_delta

        # boundary conditions
        self.boundary_conditions = np.zeros(self.n_node, dtype=int)
        self.boundary_conditions[0] = 1
        self.boundary_conditions[-1] = -1

        # beam number
        self.beam_number = np.zeros(self.n_elem, dtype=int)

        # structural twist - unused
        self.structural_twist = np.zeros((self.n_elem, self.num_node_elem))

        # externally applied forces
        self.app_forces = np.zeros((self.n_node, 6))

    def save_fem_file(self, case_name, case_route='./'):

        filepath = case_route + f'/{case_name}.fem.h5'

        # perform sweep last, as original structural code would need to be modified otherwise
        x_sweep = -np.sin(self.sweep_angle) * self.y
        y_sweep = np.cos(-self.sweep_angle) * self.y
        if self.mirrored:
            x_sweep[int((self.n_node + 1) // 2):] = np.sin(self.sweep_angle) * self.y[int((self.n_node + 1) // 2):]
            y_sweep[int((self.n_node + 1) // 2):] = np.cos(-self.sweep_angle) * self.y[int((self.n_node + 1) // 2):]

        self.x = x_sweep
        self.y = y_sweep

        with h5.File(filepath, 'w') as h5file:
            h5file.create_dataset('coordinates', data=np.column_stack((self.x, self.y, self.z)))
            h5file.create_dataset('connectivities', data=self.connectivities)
            h5file.create_dataset('num_node_elem', data=self.num_node_elem)
            h5file.create_dataset('num_node', data=self.n_node)
            h5file.create_dataset('num_elem', data=self.n_elem)
            h5file.create_dataset('stiffness_db', data=self.stiffness_db * self.sigma)
            h5file.create_dataset('elem_stiffness', data=self.elem_stiffness)
            h5file.create_dataset('mass_db', data=self.mass_db)
            h5file.create_dataset('elem_mass', data=self.elem_mass)
            h5file.create_dataset('frame_of_reference_delta', data=self.frame_of_reference_delta)
            h5file.create_dataset('structural_twist', data=self.structural_twist)
            h5file.create_dataset('boundary_conditions', data=self.boundary_conditions)
            h5file.create_dataset('beam_number', data=self.beam_number)
            h5file.create_dataset('app_forces', data=self.app_forces)

            if self.lumped_mass is not None:
                h5file.create_dataset('lumped_mass_nodes', data=self.lumped_mass_nodes)
                h5file.create_dataset('lumped_mass', data=self.lumped_mass)
                h5file.create_dataset('lumped_mass_inertia', data=self.lumped_mass_inertia)
                h5file.create_dataset('lumped_mass_position', data=self.lumped_mass_position)

    def mirror_wing(self):
        # mirror on xa-za plane
        if self.mirrored:
            print('wing already mirrored')
            return

        new_connectivities = np.zeros_like(self.connectivities)
        new_connectivities[:, 0] = np.arange(self.n_node, 2 * self.n_node - 2, 2)
        new_connectivities[:, 1] = np.arange(self.n_node + 2, 2 * self.n_node, 2)
        new_connectivities[:, 2] = np.arange(self.n_node + 1, 2 * self.n_node - 1, 2)
        new_connectivities[-1, 1] = 0

        self.connectivities = np.concatenate((self.connectivities, new_connectivities))

        self.elem_mass = np.concatenate((self.elem_mass, self.elem_mass[::-1]))
        self.app_forces = np.concatenate((self.app_forces, self.app_forces[1:][::-1]))

        self.n_elem *= 2
        self.n_node = 2 * self.n_node - 1

        self.y = np.concatenate((self.y, -self.y[1:][::-1]))
        self.x = np.zeros_like(self.y)
        self.z = np.zeros_like(self.y)

        self.beam_number = np.concatenate((self.beam_number, self.beam_number + 1))
        self.structural_twist = np.zeros((self.n_elem, self.num_node_elem))

        self.boundary_conditions = np.concatenate((self.boundary_conditions, self.boundary_conditions[1:][::-1]))

        self.frame_of_reference_delta = np.concatenate((self.frame_of_reference_delta, self.frame_of_reference_delta))

        # mirror stiffness matrix
        self.stiffness_db = np.concatenate((self.stiffness_db, self.stiffness_db[::-1]))
        self.elem_stiffness = np.arange(self.n_elem)

        self.stiffness_db[self.n_elem // 2:, 0, 3] *= -1
        self.stiffness_db[self.n_elem // 2:, 3, 0] *= -1
        self.stiffness_db[self.n_elem // 2:, 3, 4:] *= -1
        self.stiffness_db[self.n_elem // 2:, 4:, 3] *= -1

        # mirror inertia matrix
        self.mass_db = np.concatenate((self.mass_db, self.mass_db[::-1]))
        self.elem_mass = np.arange(self.n_elem)

        self.mass_db[self.n_elem // 2:, 3, 4:] *= -1
        self.mass_db[self.n_elem // 2:, 4:, 3] *= -1
        self.mass_db[self.n_elem // 2:, 1, 5] *= -1
        self.mass_db[self.n_elem // 2:, 2, 4] *= -1
        self.mass_db[self.n_elem // 2:, 5, 1] *= -1
        self.mass_db[self.n_elem // 2:, 4, 2] *= -1

        self.mirrored = True
