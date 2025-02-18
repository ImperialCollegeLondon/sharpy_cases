import numpy as np
import h5py as h5
import sharpy.utils.geo_utils as geo_utils


class PazyAero:

    def __init__(self, main_ea, pazy_structure, **kwargs):
        self.m = kwargs.get('surface_m', 4)
        self.n_surfaces = kwargs.get('num_surfaces', 2)

        # assert pazy_structure.mirrored is True, 'Pazy beam not mirrored'
        self.num_elem_tot = pazy_structure.n_elem
        self.num_elem_surf = self.num_elem_tot // self.n_surfaces
        self.num_node_tot = pazy_structure.n_node + 1
        self.num_node_surf = pazy_structure.n_node // self.n_surfaces + 1

        self.main_chord = 0.1

        self.num_node_elem = pazy_structure.num_node_elem

        # elastic axis location::: changes with skin
        self.main_ea = main_ea

        # Aerofoil shape: root and tip
        self.root_airfoil_P = 0
        self.root_airfoil_M = 0
        self.tip_airfoil_P = 0
        self.tip_airfoil_M = 0

        self.conn_surf = pazy_structure.connectivities[:self.num_elem_surf, :]
        self.surface_number = pazy_structure.beam_number

        self.airfoils_surf = None
        self.airfoil_distribution = None

        ### others
        self.aero_node = None
        self.surface_m = None

        self.twist = None
        self.chord = None  # for .aero.h5 file in (num_elem, 3) format
        self.elastic_axis = None  # for .aero.h5 file in (num_elem, 3) format

        #control surface parameters
        self.pct_flap = 0.2
        cs_deflection = [0, 0]
        self.n_control_surfaces = len(cs_deflection)
        self.control_surface = np.zeros((self.num_elem_tot, 3), dtype=int) - 1
        self.control_surface_deflection = np.zeros(self.n_control_surfaces, dtype=float)
        for i in range(len(cs_deflection)):
            self.control_surface_deflection[i] = cs_deflection[i] * np.pi / 180
        self.control_surface_chord = self.m // 2 * np.ones(self.n_control_surfaces, dtype=int)
        self.control_surface_type = np.zeros(self.n_control_surfaces, dtype=int)

    def generate_aero(self):

        n_surfaces = self.n_surfaces
        num_node_surf = self.num_node_surf
        num_node_tot = self.num_node_tot
        num_elem_surf = self.num_elem_surf
        num_elem_tot = self.num_elem_tot
        pct_flap = self.pct_flap

        control_surface = self.control_surface

        ### Generate aerofoil profiles. Only on surf 0.
        airfoils_surf = []
        if n_surfaces == 2:
            for inode in range(num_node_surf):
                eta = inode / num_node_surf
                airfoils_surf.append(
                    np.column_stack(
                        geo_utils.interpolate_naca_camber(
                            eta,
                            self.root_airfoil_M, self.root_airfoil_P,
                            self.tip_airfoil_M, self.tip_airfoil_P)))

            ws_elem = 0
            for i_surf in range(2):
                for i_elem in range(num_elem_surf):
                    for i_local_node in range(self.num_node_elem):
                        if i_elem >= int(num_elem_surf * (1 - pct_flap)):
                            if i_surf == 0:
                                control_surface[ws_elem + i_elem, i_local_node] = 0  # Right flap
                            else:
                                control_surface[ws_elem + i_elem, i_local_node] = 0  # Left flap
                ws_elem += num_elem_surf

            airfoil_distribution_surf = self.conn_surf
            airfoil_distribution = np.concatenate([airfoil_distribution_surf,
                                                   airfoil_distribution_surf[::-1, [1, 0, 2]]])
            control_surface[-num_elem_surf:] = control_surface[-num_elem_surf:, :][::-1]

        elif n_surfaces == 1:
            num_node_half = (num_node_surf + 1) // 2
            for inode in range(num_node_half):
                eta = inode / num_node_half
                airfoils_surf.append(
                    np.column_stack(
                        geo_utils.interpolate_naca_camber(
                            eta,
                            self.root_airfoil_M, self.root_airfoil_P,
                            self.tip_airfoil_M, self.tip_airfoil_P)))
            airfoil_distribution_surf = self.conn_surf[:num_elem_surf // 2, :]
            airfoil_distribution = np.concatenate([
                airfoil_distribution_surf[::-1, [1, 0, 2]],
                airfoil_distribution_surf])
        else:
            raise Exception('number of surfaces should be 1 or 2')

        self.airfoils_surf = airfoils_surf
        self.airfoil_distribution = airfoil_distribution

        ### others
        self.aero_node = np.ones((num_node_tot,), dtype=bool)
        self.surface_m = self.m * np.ones((n_surfaces,), dtype=int)

        self.twist = np.zeros((num_elem_tot, 3))
        self.chord = self.main_chord * np.ones((num_elem_tot, 3))
        self.elastic_axis = self.main_ea * np.ones((num_elem_tot, 3,))
        self.control_surface = control_surface

    def save_files(self, case_name, case_route='./'):

        filepath = case_route + '/{}.aero.h5'.format(case_name)

        with h5.File(case_route + '/' + case_name + '.aero.h5', 'w') as h5file:
            airfoils_group = h5file.create_group('airfoils')
            # add one airfoil
            for aa in range(len(self.airfoils_surf)):
                airfoils_group.create_dataset('%d' % aa, data=self.airfoils_surf[aa])

            chord_input = h5file.create_dataset('chord', data=self.chord)
            chord_input.attrs['units'] = 'm'

            twist_input = h5file.create_dataset('twist', data=self.twist)
            twist_input.attrs['units'] = 'rad'

            # airfoil distribution
            h5file.create_dataset('airfoil_distribution', data=self.airfoil_distribution)
            h5file.create_dataset('surface_distribution', data=self.surface_number)
            h5file.create_dataset('surface_m', data=self.surface_m)
            h5file.create_dataset('m_distribution', data='uniform'.encode('ascii', 'ignore'))
            h5file.create_dataset( 'aero_node', data=self.aero_node)
            h5file.create_dataset('elastic_axis', data=self.elastic_axis)

            if self.control_surface is not None:
                h5file.create_dataset('control_surface', data=self.control_surface)
                h5file.create_dataset('control_surface_type', data=self.control_surface_type)
                h5file.create_dataset('control_surface_deflection', data=self.control_surface_deflection)
                h5file.create_dataset('control_surface_chord', data=self.control_surface_chord)
