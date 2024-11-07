import pandas as pd
import numpy as np
from typing import Any, Optional
from scipy.io import loadmat, matlab
import h5py as h5
import os
import configobj
from time import time

from sharpy.utils.algebra import euler2quat, quat2rotation


class FlexopStructure:
    def __init__(self, case_name, case_route, **kwargs):
        self.case_name = case_name
        self.case_route = case_route
        self.source_directory = self.source_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                     'aeroelastic_properties')

        self.input_settings = kwargs
        self.settings = dict()
        self.constraint_settings = dict()

        # base parameters
        self.sigma: float = kwargs.get('sigma', 1.)
        self.sigma_fuselage: float = kwargs.get('sigma_fuselage', 1.)
        self.sigma_tail: float = kwargs.get('sigma_tail', 1.)
        self.material: str = kwargs.get('material', 'reference')
        self.hinge_eta: float = kwargs.get('hinge_eta', 0.7)
        self.include_tail: bool = kwargs.get('include_tail', True)
        self.include_fuselage: bool = kwargs.get('include_fuselage', False)
        self.use_multibody: bool = kwargs.get('use_multibody', False)
        self.num_node_elem: int = 3
        self.num_elem_mult: float = kwargs.get('n_elem_multiplier', 1.)
        self.num_elem_mult_tail: float = kwargs.get('n_elem_multiplier_tail', self.num_elem_mult)
        self.num_elem_mult_fuselage: float = kwargs.get('n_elem_multiplier_fuselage', self.num_elem_mult)
        self.dx_payload: float = kwargs.get('dx_payload', 0.)
        self.alpha: float = kwargs.get('alpha', 0.)
        self.roll: float = kwargs.get('roll', 0.)
        self.yaw: float = kwargs.get('roll', 0.)
        self.in_quat = euler2quat(np.array((self.roll, self.alpha, self.yaw)))
        self.in_rot = quat2rotation(self.in_quat)
        self.use_rigid_sweep: bool = kwargs.get('use_rigid_sweep', False)
        self.rigid_sweep_ang: float = kwargs.get('rigid_sweep_ang', 0.)
        self.num_elem_warp_main: int = kwargs.get('num_elem_warp_main', 4)
        self.num_elem_warp_tip: int = kwargs.get('num_elem_warp_tip', 2)
        self.gravity_on: bool = kwargs.get('gravity_on', True)
        self.free: bool = kwargs.get('free', False)
        self.use_jax: bool = kwargs.get('use_jax', True)
        self.start_time = time()

        # create geometric parameters
        self.dimensions: dict[str, float] = dict()
        self.create_dimensions()

        # create slices for indexing nodes and elements
        self.num_elem: dict[str, int] = dict()
        self.elem_slice: dict[str, np.ndarray] = dict()
        self.create_elem_indexing()

        self.num_node: dict[str, int] = dict()
        self.node_slice: dict[str, np.ndarray] = dict()
        self.create_node_indexing()

        # load lumped mass database
        self.lumped_mass_db: pd.DataFrame = pd.read_csv(self.source_directory + '/lumped_masses.csv', sep=';',
                                                        header=None)

        # number of lumped masses per wing
        self.num_lumped_masses_wing: int = self.lumped_mass_db.shape[0]

        # total number of lumped masses, including 2 for payload, engine, fuel
        self.num_lumped_masses: int = 2 * self.num_lumped_masses_wing + 2 * self.include_tail
        self.lumped_mass_nodes = np.zeros(self.num_lumped_masses, dtype=int)
        self.lumped_mass = np.zeros(self.num_lumped_masses)
        self.lumped_mass_inertia = np.zeros((self.num_lumped_masses, 3, 3))
        self.lumped_mass_position = np.zeros((self.num_lumped_masses, 3))

        # stiffness properties
        self.num_stiffness_per_wing: int = 17
        self.num_stiffness: int = self.num_stiffness_per_wing * 2 + 2 * self.include_tail
        self.stiffness = np.zeros((self.num_stiffness, 6, 6))
        self.elem_stiffness = np.zeros(self.num_elem['total'], dtype=int)

        # base stiffness properties for fuselage and tail
        self.stiffness_base: dict[str, float] = {'ea': 1e7, 'gay': 1e5, 'gaz': 1e5, 'gj': 1e4, 'eiy': 2e4, 'eiz': 4e6}
        if self.include_tail:
            self.stiffness[-2, ...] = self.sigma_fuselage * np.diag(tuple(self.stiffness_base.values()))
            self.stiffness[-1, ...] = self.sigma_tail * np.diag(tuple(self.stiffness_base.values()))

        # inertia properties
        self.num_mass: int = self.num_elem['both_wings'] + 2 * self.include_tail
        self.m_bar: dict[str, float] = {'fuselage': kwargs.get('m_bar_fuselage', 3.),
                                        'tail': kwargs.get('m_bar_tail', 1.2)}

        self.j_bar: dict[str, float] = {'fuselage': kwargs.get('m_bar_fuselage', 0.08),
                                        'tail': kwargs.get('m_bar_tail', 0.08)}

        self.mass = np.zeros((self.num_mass, 6, 6))
        self.elem_mass = np.zeros(self.num_elem['total'], dtype=int)
        if self.include_tail:
            self.mass[-2, ...] = np.diag((*([self.m_bar['fuselage']] * 3), self.j_bar['fuselage'],
                                          0.5 * self.j_bar['fuselage'], 0.5 * self.j_bar['fuselage']))
            self.mass[-1, ...] = np.diag((*([self.m_bar['tail']] * 3), self.j_bar['tail'],
                                          0.5 * self.j_bar['tail'], 0.5 * self.j_bar['tail']))

        # node placement
        self.x = np.zeros(self.num_node['total'])
        self.y = np.zeros(self.num_node['total'])
        self.z = np.zeros(self.num_node['total'])

        # when using rigid sweep, as the coordines change this messes up finding properties dependent on the spanwise
        # position, so we keep a copy of the original values here
        self.x_unswept = np.zeros(self.num_node['total'])
        self.y_unswept = np.zeros(self.num_node['total'])
        self.z_unswept = np.zeros(self.num_node['total'])

        # other fem stuff
        self.beam_number = np.zeros(self.num_elem['total'], dtype=int)
        self.body_number = np.zeros(self.num_elem['total'], dtype=int)
        self.connectivity = np.zeros((self.num_elem['total'], self.num_node_elem), dtype=int)
        self.boundary_conditions = np.zeros(self.num_node['total'], dtype=int)
        self.boundary_conditions[0] = 1  # base node
        self.for_delta = np.zeros((self.num_elem['total'], self.num_node_elem, 3))
        self.applied_forces = np.zeros((self.num_elem['total'], 6))
        self.elastic_axis = np.zeros((self.num_elem['total'], self.num_node_elem))
        self.applied_forces = np.zeros((self.num_node['total'], 6))
        self.struct_twist = np.zeros((self.num_elem['total'], self.num_node_elem))
        self.spanwise_shear_center: np.ndarray = self.read_spanwise_shear_center()
        self.thrust = self.input_settings.get('thrust', 0.)
        self.applied_forces[0, 1] = self.thrust

        # other multibody stuff
        if self.use_multibody:
            self.num_bodies: int = 3
        else:
            self.num_bodies: int = 1

        self.for_movement: list[str] = ['free'] * self.num_bodies
        self.for_acceleration = np.zeros((self.num_bodies, 6))
        self.for_velocity = np.zeros((self.num_bodies, 6))
        self.for_position = np.zeros((self.num_bodies, 6))
        self.body_quat = np.zeros((self.num_bodies, 4))
        self.body_quat[0, :] = self.in_quat
        self.constraints: list[dict[str, Any]] = []
        self.num_constraints: int = 0

        self.y_cross_sections = self.load_y_cross_sections()

        self.generate_right_wing_fem()
        self.generate_left_wing_fem()
        if self.include_tail:
            self.generate_fuselage_fem()
            self.generate_right_tail_fem()
            self.generate_left_tail_fem()

        # load mass and stiffness databases
        list_stiffness, list_mass = self.load_stiffness_and_mass_matrix()
        for i in range(int(self.num_stiffness_per_wing * 2)):
            self.stiffness[i, ...] = list_stiffness[i]
            self.mass[i, ...] = list_mass[i]
        for i in range(int(self.num_elem['both_wings'])):
            self.mass[i, ...] = list_mass[i]

        mass_data: pd.DataFrame = self.read_lumped_masses()
        self.num_lumped_mass_each_wing: int = mass_data.shape[0]
        self.num_lumped_mass: int = 2 * (self.num_lumped_mass_each_wing + self.include_tail)
        self.place_lumped_masses_wing(mass_data)

    def create_dimensions(self) -> None:
        self.dimensions.update({'kink_y': 0.144,
                                'main_span': 7.07,
                                'sweep_wing_le': np.deg2rad(20.),
                                'chord_wing_root': 0.471,
                                'chord_wing_tip': 0.236,
                                'sweep_wing_qchord': 0.319923584301128,
                                'length_fuselage': 3.44,
                                'tail_v_angle': np.deg2rad(35.),
                                'tail_semi_span': 1.318355 / 2.,
                                'chord_tail_tip': 0.180325,
                                'chord_tail_root': 0.35,
                                'tail_sweep_le': np.deg2rad(19.51951),
                                'tail_sweep_te': np.deg2rad(18.0846),
                                'ea_tail': 0.3})

        self.dimensions['semi_span'] = self.dimensions['main_span'] / 2.
        self.dimensions['hinge_y'] = (self.dimensions['kink_y']
                                      + (self.dimensions['semi_span'] - self.dimensions['kink_y']) * self.hinge_eta)
        self.dimensions['tip_x'] = self.dimensions['semi_span'] * np.tan(self.dimensions['sweep_wing_le'])

        self.dimensions['sweep_wing_te'] = np.arctan((self.dimensions['tip_x'] + self.dimensions['chord_wing_tip']
                                                      - self.dimensions['chord_wing_root'])
                                                     / self.dimensions['semi_span'])
        self.dimensions['tip_span'] = (1. - self.hinge_eta) * np.sqrt((self.dimensions['tip_x'] ** 2
                                                                       + (self.dimensions['semi_span']
                                                                          - self.dimensions['kink_y']) ** 2))
        self.dimensions['sweep_wing_beam'] = np.arctan(self.dimensions['tip_x'] / (self.dimensions['semi_span']
                                                                                   - self.dimensions['kink_y']))
        self.dimensions['tip_x_qchord'] = ((self.dimensions['semi_span'] - self.dimensions['kink_y'])
                                           * np.tan(self.dimensions['sweep_wing_qchord']))

        self.dimensions['hinge_x'] = self.dimensions['tip_x_qchord'] * self.hinge_eta

        self.dimensions['length_fuselage_front'] = 0.87692 + self.dimensions['chord_wing_root'] * 0.57
        self.dimensions['length_fuselage_main'] = 2.86236881559 - self.dimensions['length_fuselage_front']
        self.dimensions['length_fuselage_rear'] = (self.dimensions['length_fuselage']
                                                   - self.dimensions['length_fuselage_front']
                                                   - self.dimensions['length_fuselage_main'])
        self.dimensions['tail_tip_x'] = self.dimensions['tail_semi_span'] * np.tan(self.dimensions['tail_sweep_le'])
        self.dimensions['tail_tip_z'] = self.dimensions['tail_semi_span'] * np.tan(self.dimensions['tail_v_angle'])

        self.dimensions['tail_sweep_qchord'] = np.arctan((self.dimensions['tail_tip_x']
                                                          + self.dimensions['chord_tail_tip'] / 4
                                                          - self.dimensions['chord_tail_root'] / 4)
                                                         / self.dimensions['tail_semi_span'])

    def create_elem_indexing(self) -> None:
        self.num_elem.update({'wing_inner': np.ceil(1. * self.num_elem_mult).astype(int),
                              'wing_main': int(27. * self.hinge_eta * self.num_elem_mult),
                              'wing_tip': int(27. * (1. - self.hinge_eta) * self.num_elem_mult)})

        self.num_elem['each_wing'] = sum(self.num_elem.values())
        self.num_elem['both_wings'] = 2 * self.num_elem['each_wing']
        self.num_elem['total'] = self.num_elem['both_wings']

        if self.include_tail:
            self.num_elem.update({'fuselage_front': int(7. * self.num_elem_mult_fuselage),
                                  'fuselage_main': int(11. * self.num_elem_mult_fuselage),
                                  'fuselage_rear': int(4. * self.num_elem_mult_fuselage),
                                  'each_tail': int(11. * self.num_elem_mult_tail)})
            self.num_elem['total'] += (self.num_elem['fuselage_front'] + self.num_elem['fuselage_main']
                                       + self.num_elem['fuselage_rear']) + 2 * self.num_elem['each_tail']

        # element index in system
        elem_index_order: list[int] = [0, self.num_elem['wing_inner'], self.num_elem['wing_main'],
                                       self.num_elem['wing_tip'], self.num_elem['wing_inner'],
                                       self.num_elem['wing_main'], self.num_elem['wing_tip']]
        if self.include_tail:
            elem_index_order.extend([self.num_elem['fuselage_front'], self.num_elem['fuselage_main'],
                                     self.num_elem['fuselage_rear'], self.num_elem['each_tail'],
                                     self.num_elem['each_tail']])

        elem_index_cum: list[int] = list(np.cumsum(elem_index_order))
        self.elem_slice.update({'wing1_inner': np.arange(elem_index_cum[0], elem_index_cum[1]),
                                'wing1_main': np.arange(elem_index_cum[1], elem_index_cum[2]),
                                'wing1_tip': np.arange(elem_index_cum[2], elem_index_cum[3]),
                                'wing2_inner': np.arange(elem_index_cum[3], elem_index_cum[4]),
                                'wing2_main': np.arange(elem_index_cum[4], elem_index_cum[5]),
                                'wing2_tip': np.arange(elem_index_cum[5], elem_index_cum[6]),
                                'wing1': np.arange(elem_index_cum[0], elem_index_cum[3]),
                                'wing2': np.arange(elem_index_cum[3], elem_index_cum[6])})
        if self.include_tail:
            self.elem_slice.update({'fuselage_front': np.arange(elem_index_cum[6], elem_index_cum[7]),
                                    'fuselage_main': np.arange(elem_index_cum[7], elem_index_cum[8]),
                                    'fuselage_rear': np.arange(elem_index_cum[8], elem_index_cum[9]),
                                    'fuselage': np.arange(elem_index_cum[6], elem_index_cum[9]),
                                    'tail1': np.arange(elem_index_cum[9], elem_index_cum[10]),
                                    'tail2': np.arange(elem_index_cum[10], elem_index_cum[11])})

    def create_node_indexing(self) -> None:
        self.num_node.update({'wing_inner': self.num_elem['wing_inner'] * (self.num_node_elem - 1) + 1,
                              'wing_main': self.num_elem['wing_main'] * (self.num_node_elem - 1) + 1,
                              'wing_tip': self.num_elem['wing_tip'] * (self.num_node_elem - 1) + 1})
        if self.include_tail:
            self.num_node.update({'fuselage_front': self.num_elem['fuselage_front'] * (self.num_node_elem - 1) + 1,
                                  'fuselage_main': self.num_elem['fuselage_main'] * (self.num_node_elem - 1) + 1,
                                  'fuselage_rear': self.num_elem['fuselage_rear'] * (self.num_node_elem - 1) + 1,
                                  'each_tail': self.num_elem['each_tail'] * (self.num_node_elem - 1) + 1})
        self.num_node['each_wing'] = (self.num_node['wing_inner'] + self.num_node['wing_main']
                                      + self.num_node['wing_tip'] - 2)
        if self.use_multibody:
            self.num_node['each_wing'] += 1

        self.num_node['both_wings'] = 2 * self.num_node['each_wing'] - 1
        self.num_node['total'] = self.num_node['both_wings']
        if self.include_tail:
            self.num_node['total'] += (self.num_node['fuselage_front'] + self.num_node['fuselage_main']
                                       + self.num_node['fuselage_rear'] + 2 * self.num_node['each_tail'] - 5)

        # node index in system
        num_node_order = np.array((0, self.num_node['wing_inner'] - 1, self.num_node['wing_main'] - 1,
                                   self.num_node['wing_tip'] - 1, self.num_node['wing_inner'] - 1,
                                   self.num_node['wing_main'] - 1, self.num_node['wing_tip'] - 1))

        if self.include_tail:
            num_node_order = np.append(num_node_order,
                                       (self.num_node['fuselage_front'] - 1, self.num_node['fuselage_main'] - 1,
                                        self.num_node['fuselage_rear'] - 1, self.num_node['each_tail'] - 1,
                                        self.num_node['each_tail'] - 1))

        node_index_cum: list[int] = list(np.cumsum(num_node_order))

        if self.use_multibody:
            self.node_slice.update({'wing1_inner': np.arange(node_index_cum[0], node_index_cum[1] + 1),
                                    'wing1_main': np.arange(node_index_cum[1], node_index_cum[2] + 1),
                                    'wing1_tip': np.arange(node_index_cum[2] + 1, node_index_cum[3] + 2),
                                    'wing2_inner': np.array(
                                        (0, *np.arange(node_index_cum[3] + 2, node_index_cum[4] + 2))),
                                    'wing2_main': np.arange(node_index_cum[4] + 1, node_index_cum[5] + 2),
                                    'wing2_tip': np.arange(node_index_cum[5] + 2, node_index_cum[6] + 3),
                                    'wing1': np.arange(node_index_cum[0], node_index_cum[3] + 2),
                                    'wing2': np.array((0, *np.arange(node_index_cum[3] + 2, node_index_cum[6] + 3)))})
            i_end: int = 3
        else:
            self.node_slice.update({'wing1_inner': np.arange(node_index_cum[0], node_index_cum[1] + 1),
                                    'wing1_main': np.arange(node_index_cum[1], node_index_cum[2] + 1),
                                    'wing1_tip': np.arange(node_index_cum[2], node_index_cum[3] + 1),
                                    'wing2_inner': np.array(
                                        (0, *np.arange(node_index_cum[3] + 1, node_index_cum[4] + 1))),
                                    'wing2_main': np.arange(node_index_cum[4], node_index_cum[5] + 1),
                                    'wing2_tip': np.arange(node_index_cum[5], node_index_cum[6] + 1),
                                    'wing1': np.arange(node_index_cum[0], node_index_cum[3] + 1),
                                    'wing2': np.array((0, *np.arange(node_index_cum[3] + 1, node_index_cum[6] + 1)))})
            i_end: int = 1

        if self.include_tail:
            node_tail_base = node_index_cum[8] + i_end - 1
            self.node_slice.update({'fuselage_front': np.array((0, *np.arange(node_index_cum[6] + i_end,
                                                                              node_index_cum[7] + i_end))),
                                    'fuselage_main': np.array((0, *np.arange(node_index_cum[7] + i_end,
                                                                             node_index_cum[8] + i_end))),
                                    'fuselage_rear': np.arange(node_index_cum[8] + i_end - 1,
                                                               node_index_cum[9] + i_end),
                                    'tail1': np.array(
                                        (node_tail_base,
                                         *np.arange(node_index_cum[9] + i_end, node_index_cum[10] + i_end))),
                                    'tail2': np.array(
                                        (node_tail_base,
                                         *np.arange(node_index_cum[10] + i_end, node_index_cum[11] + i_end)))})

    def generate_right_wing_fem(self):
        # straight section
        self.y[self.node_slice['wing1_inner']] \
            = np.linspace(0., self.dimensions['kink_y'], self.num_node['wing_inner'])

        # main section
        self.x[self.node_slice['wing1_main']] \
            = np.linspace(0., self.dimensions['hinge_x'], self.num_node['wing_main'])
        self.y[self.node_slice['wing1_main']] \
            = np.linspace(self.dimensions['kink_y'], self.dimensions['hinge_y'], self.num_node['wing_main'])

        # tip section
        self.x[self.node_slice['wing1_tip']] \
            = np.linspace(self.dimensions['hinge_x'], self.dimensions['tip_x_qchord'], self.num_node['wing_tip'])
        self.y[self.node_slice['wing1_tip']] \
            = np.linspace(self.dimensions['hinge_y'], self.dimensions['semi_span'], self.num_node['wing_tip'])

        self.elem_mass[self.elem_slice['wing1']] = np.arange(self.num_elem['each_wing'])
        self.for_delta[self.elem_slice['wing1'], :, 0] = -1.
        self.boundary_conditions[self.node_slice['wing1'][-1]] = -1  # wing tip

        if self.use_multibody:
            self.boundary_conditions[self.node_slice['wing1_main'][-1]] = -1  # hinge tip
            self.boundary_conditions[self.node_slice['wing1_tip'][0]] = 1  # hinge root
            self.body_number[self.elem_slice['wing1_tip']] = 1
            self.beam_number[self.elem_slice['wing1_tip']] = 1
            self.body_quat[1, :] = self.in_quat

            for_pos_a = np.array((self.x[self.node_slice['wing1_tip'][0]],
                                  self.y[self.node_slice['wing1_tip'][0]],
                                  self.z[self.node_slice['wing1_tip'][0]]))
            for_pos_g = self.in_rot @ for_pos_a
            self.for_position[1, :3] = for_pos_g

        # connectivity
        if self.use_multibody:
            for i_elem in range(self.elem_slice['wing1_inner'][0], self.elem_slice['wing1_main'][-1] + 1):
                self.connectivity[i_elem, :] = np.ones(3) * i_elem * (self.num_node_elem - 1) + (0, 2, 1)

            for i_elem in self.elem_slice['wing1_tip']:
                self.connectivity[i_elem, :] = np.ones(3) * i_elem * (self.num_node_elem - 1) + (1, 3, 2)
        else:
            for i_elem in self.elem_slice['wing1']:
                self.connectivity[i_elem, :] = np.ones(3) * i_elem * (self.num_node_elem - 1) + (0, 2, 1)

        for i_elem in self.elem_slice['wing1']:
            i_base_node = i_elem * (self.num_node_elem - 1)
            i_pos = np.argmin(np.abs(self.y_cross_sections - self.y[i_base_node]))

            self.elem_stiffness[i_elem] = i_pos
            if self.y_cross_sections[i_pos] < self.y[i_base_node + 2]:
                self.elem_stiffness[i_elem] += 1
            self.elem_mass[i_elem] = i_elem

            self.elastic_axis[i_elem, :] = self.spanwise_shear_center[self.elem_stiffness[i_elem]]

        self.x_unswept[self.node_slice['wing1']] = self.x[self.node_slice['wing1']]
        self.y_unswept[self.node_slice['wing1']] = self.y[self.node_slice['wing1']]

        if self.use_rigid_sweep:
            tip_l = np.sqrt((self.dimensions['tip_x_qchord'] - self.dimensions['hinge_x']) ** 2
                            + (self.dimensions['semi_span'] - self.dimensions['hinge_y']) ** 2)
            sweep_tip_x = (self.dimensions['hinge_x']
                           + np.sin(self.rigid_sweep_ang + self.dimensions['sweep_wing_qchord']) * tip_l)
            sweep_tip_y = (self.dimensions['hinge_y']
                           + np.cos(self.rigid_sweep_ang + self.dimensions['sweep_wing_qchord']) * tip_l)
            self.x[self.node_slice['wing1_tip']] \
                = np.linspace(self.dimensions['hinge_x'], sweep_tip_x, self.num_node['wing_tip'])
            self.y[self.node_slice['wing1_tip']] \
                = np.linspace(self.dimensions['hinge_y'], sweep_tip_y, self.num_node['wing_tip'])
        pass

    def generate_left_wing_fem(self):
        # straight section
        self.x[self.node_slice['wing2']] = self.x[self.node_slice['wing1']]
        self.y[self.node_slice['wing2']] = -self.y[self.node_slice['wing1']]

        self.x_unswept[self.node_slice['wing2']] = self.x_unswept[self.node_slice['wing1']]
        self.y_unswept[self.node_slice['wing2']] = -self.x_unswept[self.node_slice['wing1']]

        self.elem_mass[self.elem_slice['wing2']] = self.elem_mass[self.elem_slice['wing1']]
        self.for_delta[self.elem_slice['wing2'], :, 0] = 1.
        self.boundary_conditions[self.node_slice['wing2']] = self.boundary_conditions[self.node_slice['wing1']]

        self.elem_stiffness[self.elem_slice['wing2']] = (self.elem_stiffness[self.elem_slice['wing1']]
                                                         + self.elem_stiffness[self.elem_slice['wing1'][-1]] + 1)
        self.elem_mass[self.elem_slice['wing2']] = (self.elem_mass[self.elem_slice['wing1']]
                                                    + self.elem_mass[self.elem_slice['wing1'][-1]] + 1)
        self.elastic_axis[self.elem_slice['wing2'], :] = self.elastic_axis[self.elem_slice['wing1'], :]
        self.connectivity[self.elem_slice['wing2'], :] = (self.connectivity[self.elem_slice['wing1'], :]
                                                          + self.node_slice['wing1'][-1])
        self.connectivity[self.elem_slice['wing2'][0], 0] = 0  # connect to junction

        if self.use_multibody:
            self.body_number[self.elem_slice['wing2_tip']] = 2
            self.beam_number[self.elem_slice['wing2']] = 2
            self.beam_number[self.elem_slice['wing2_tip']] = 3
            self.body_quat[2, :] = self.in_quat

            for_pos_a = np.array((self.x[self.node_slice['wing2_tip'][0]],
                                  self.y[self.node_slice['wing2_tip'][0]],
                                  self.z[self.node_slice['wing2_tip'][0]]))
            for_pos_g = self.in_rot @ for_pos_a
            self.for_position[2, :3] = for_pos_g
        else:
            self.beam_number[self.elem_slice['wing2']] = 1

    def generate_fuselage_fem(self):
        # front section
        self.x[self.node_slice['fuselage_front']] = np.linspace(0., -self.dimensions['length_fuselage_front'],
                                                                self.num_node['fuselage_front'])

        # middle section
        self.x[self.node_slice['fuselage_main']] = np.linspace(0., self.dimensions['length_fuselage_main'],
                                                               self.num_node['fuselage_main'])

        # tail section
        self.x[self.node_slice['fuselage_rear']] \
            = np.linspace(self.dimensions['length_fuselage_main'], self.dimensions['length_fuselage_main']
                          + self.dimensions['length_fuselage_rear'], self.num_node['fuselage_rear'])

        self.for_delta[self.elem_slice['fuselage_front'], :, 1] = -1.
        self.for_delta[self.elem_slice['fuselage_main'], :, 1] = 1.
        self.for_delta[self.elem_slice['fuselage_rear'], :, 1] = 1.

        self.elem_stiffness[self.elem_slice['fuselage']] = self.num_stiffness - 2
        self.elem_mass[self.elem_slice['fuselage']] = self.num_mass - 2
        self.boundary_conditions[self.node_slice['fuselage_front'][-1]] = -1
        self.boundary_conditions[self.node_slice['fuselage_rear'][-1]] = -1

        for i_elem in self.elem_slice['fuselage_front']:
            self.connectivity[i_elem, :] = np.ones(3) * i_elem * (self.num_node_elem - 1) + (0, 2, 1)

        for i_elem in range(self.elem_slice['fuselage_main'][0], self.elem_slice['fuselage_rear'][-1] + 1):
            self.connectivity[i_elem, :] = np.ones(3) * i_elem * (self.num_node_elem - 1) + (0, 2, 1)

        if self.use_multibody:
            self.connectivity[self.elem_slice['fuselage'], :] += 2
            self.beam_number[self.elem_slice['fuselage_front']] = 4
            self.beam_number[self.elem_slice['fuselage_main']] = 5
            self.beam_number[self.elem_slice['fuselage_rear']] = 5
        else:
            self.beam_number[self.elem_slice['fuselage_front']] = 2
            self.beam_number[self.elem_slice['fuselage_main']] = 3
            self.beam_number[self.elem_slice['fuselage_rear']] = 3

        self.connectivity[self.elem_slice['fuselage_front'][0], 0] = 0
        self.connectivity[self.elem_slice['fuselage_main'][0], 0] = 0

        x_payload = self.dx_payload + 0.2170
        self.lumped_mass[-1] = 42 - 10.799 - 0.35833756498172
        self.lumped_mass_nodes[-1] = self.node_slice['fuselage_main'][self.find_index_of_closest_entry(
            self.x[self.node_slice['fuselage_main']], x_payload)]
        self.lumped_mass_position[-1, :] = [x_payload - self.x[self.lumped_mass_nodes[-1]], 0., -0.25]

    def generate_right_tail_fem(self):
        self.x[self.node_slice['tail1']] = ((np.linspace(0., self.dimensions['tail_tip_x'],
                                                         self.num_node['each_tail']))
                                            + self.dimensions['length_fuselage_main'])

        self.y[self.node_slice['tail1']] \
            = np.linspace(0., self.dimensions['tail_semi_span'], self.num_node['each_tail'])

        self.z[self.node_slice['tail1']] \
            = np.linspace(0., self.dimensions['tail_tip_z'], self.num_node['each_tail'])

        self.for_delta[self.elem_slice['tail1'], :, 0] = -1.
        self.elem_stiffness[self.elem_slice['tail1']] = self.num_stiffness - 1
        self.elem_mass[self.elem_slice['tail1']] = self.num_mass - 1
        self.boundary_conditions[self.node_slice['tail1'][-1]] = -1

        for i_elem in self.elem_slice['tail1']:
            self.connectivity[i_elem, :] = np.ones(3) * i_elem * (self.num_node_elem - 1) + (0, 2, 1)

        if self.use_multibody:
            self.beam_number[self.elem_slice['tail1']] = 6
            self.connectivity[self.elem_slice['tail1'], :] += 2
        else:
            self.beam_number[self.elem_slice['tail1']] = 4

        self.connectivity[self.elem_slice['tail1'][0], 0] = self.node_slice['fuselage_main'][-1]

    def generate_left_tail_fem(self):
        self.x[self.node_slice['tail2']] = self.x[self.node_slice['tail1']]
        self.y[self.node_slice['tail2']] = -self.y[self.node_slice['tail1']]
        self.z[self.node_slice['tail2']] = self.z[self.node_slice['tail1']]

        self.for_delta[self.elem_slice['tail2'], :, 0] = 1.
        self.elem_stiffness[self.elem_slice['tail2']] = self.num_stiffness - 1
        self.elem_mass[self.elem_slice['tail2']] = self.num_mass - 1
        self.boundary_conditions[self.node_slice['tail2'][-1]] = -1

        for i_elem in self.elem_slice['tail2']:
            self.connectivity[i_elem, :] = np.ones(3) * i_elem * (self.num_node_elem - 1) + (0, 2, 1)

        if self.use_multibody:
            self.beam_number[self.elem_slice['tail2']] = 7
            self.connectivity[self.elem_slice['tail2'], :] += 2
        else:
            self.beam_number[self.elem_slice['tail2']] = 5

        self.connectivity[self.elem_slice['tail2'][0], 0] = self.node_slice['fuselage_main'][-1]

    def generate_h5_fem(self):
        file_route = self.case_route + '/' + self.case_name + '.fem.h5'
        try:
            os.remove(file_route)
            print("Deleted FEM H5 file")
        except OSError:
            pass

        with h5.File(file_route, 'a') as h5file:
            h5file.create_dataset('coordinates', data=np.column_stack((self.x, self.y, self.z)))
            h5file.create_dataset('connectivities', data=self.connectivity)
            h5file.create_dataset('num_node_elem', data=self.num_node_elem)
            h5file.create_dataset('num_node', data=self.num_node['total'])
            h5file.create_dataset('num_elem', data=self.num_elem['total'])
            h5file.create_dataset('stiffness_db', data=self.stiffness)
            h5file.create_dataset('elem_stiffness', data=self.elem_stiffness)
            h5file.create_dataset('mass_db', data=self.mass)
            h5file.create_dataset('elem_mass', data=self.elem_mass)
            h5file.create_dataset('frame_of_reference_delta', data=self.for_delta)
            h5file.create_dataset('structural_twist', data=self.struct_twist)
            h5file.create_dataset('boundary_conditions', data=self.boundary_conditions)
            h5file.create_dataset('beam_number', data=self.beam_number)
            h5file.create_dataset('body_number', data=self.body_number)
            h5file.create_dataset('app_forces', data=self.applied_forces)
            h5file.create_dataset('lumped_mass_nodes', data=self.lumped_mass_nodes)
            h5file.create_dataset('lumped_mass', data=self.lumped_mass)
            h5file.create_dataset('lumped_mass_inertia', data=self.lumped_mass_inertia)
            h5file.create_dataset('lumped_mass_position', data=self.lumped_mass_position)

    def generate_h5_mb(self):
        file_route = self.case_route + '/' + self.case_name + '.mb.h5'
        try:
            os.remove(file_route)
            print("Deleted MB H5 file")
        except OSError:
            pass

        with h5.File(file_route, 'a') as h5file:
            for i_body in range(self.num_bodies):
                body = h5file.create_group(f'body_{i_body:02d}')
                body.create_dataset('FoR_acceleration', data=self.for_acceleration[i_body, :])
                body.create_dataset('FoR_velocity', data=self.for_velocity[i_body, :])
                body.create_dataset('FoR_position', data=self.for_position[i_body, :])
                body.create_dataset('body_number', data=i_body)
                body.create_dataset('quat', data=self.body_quat[i_body, :])
                body.create_dataset('FoR_movement', data=self.for_movement[i_body])

            for i_cst in range(self.num_constraints):
                cst = h5file.create_group(f'constraint_{i_cst:02d}')
                for k, v in self.constraints[i_cst].items():
                    cst.create_dataset(k, data=v)

            h5file.create_dataset('num_bodies', data=self.num_bodies)
            h5file.create_dataset('num_constraints', data=self.num_constraints)

    # """
    # Copied functions :(
    # """

    def place_lumped_masses_wing(self, df_lumped_masses: pd.DataFrame):
        # dataframe has order: node mass; node xyz (?) and a bunch of zeros
        # find node closest to lumped mass (using y)
        # find position of mass relative to node in the node's coordinate system

        for imass in range(self.num_lumped_mass_each_wing):
            self.lumped_mass[imass] = df_lumped_masses.iloc[imass, 0]
            self.lumped_mass_nodes[imass] = self.find_index_of_closest_entry(self.y[:self.num_node['each_wing']],
                                                                             df_lumped_masses.iloc[imass, 1])
            self.set_lumped_mass_position_b_frame(imass, np.array(df_lumped_masses.iloc[imass, 1:4]),
                                                  self.lumped_mass_nodes[imass])
            # mirror lumped masses for left wing
            idx_symmetric = self.num_lumped_mass_each_wing + imass
            self.lumped_mass[idx_symmetric] = self.lumped_mass[imass]
            if self.lumped_mass_nodes[imass] == 0:
                self.lumped_mass_nodes[idx_symmetric] = 0
            else:
                self.lumped_mass_nodes[idx_symmetric] = self.num_node['each_wing'] - 1 + self.lumped_mass_nodes[imass]

            self.lumped_mass[idx_symmetric] = self.lumped_mass[imass]
            self.lumped_mass_position[idx_symmetric, 0] = self.lumped_mass_position[imass, 0]
            self.lumped_mass_position[idx_symmetric, 1] = -self.lumped_mass_position[imass, 1]
            self.lumped_mass_position[idx_symmetric, 2] = self.lumped_mass_position[imass, 2]

    def set_lumped_mass_position_b_frame(self, imass, position, inode):
        position[0] -= 0.22
        self.lumped_mass_position[imass, 2] = position[2]
        self.lumped_mass_position[imass, 0] = position[1] - self.y[inode]
        self.lumped_mass_position[imass, 1] = position[0] - self.x[inode]
        if self.y[inode] > self.dimensions['kink_y']:
            # local COS rotated around z-axis by beam sweep angle
            self.lumped_mass_position[imass, 0] /= np.cos(self.dimensions['sweep_wing_qchord'])
            self.lumped_mass_position[imass, 1] -= (self.lumped_mass_position[imass, 0]
                                                    * np.sin(self.dimensions['sweep_wing_qchord']))
            self.lumped_mass_position[imass, 1] *= np.cos(self.dimensions['sweep_wing_qchord'])

    @staticmethod
    def find_index_of_closest_entry(array_values, target_value):
        return np.argmin(np.abs(array_values - target_value))

    def read_spanwise_shear_center(self) -> np.ndarray:
        reference_shear_center = 0.71  # given by Jurij
        df = pd.read_csv(self.source_directory + '/shear_center.csv', sep=';')
        if self.material == "reference":
            return np.array(df.iloc[:, 1] + reference_shear_center)
        else:
            return np.array(df.iloc[:, 2] + reference_shear_center)

    def load_stiffness_and_mass_matrix(self):
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
            tmp_stiffness_matrix = np.array(matrices_cross_stiffness[counter:counter + 6, :])
            tmp_stiffness_matrix[5, 5] /= self.sigma / 2
            list_stiffness_matrix.append(tmp_stiffness_matrix)

            mass_matrix = np.zeros((6, 6))
            # mass distribution
            mass = float(matrices_cross_mass[row_counter])
            for i in range(3):
                mass_matrix[i, i] = mass
            mass_matrix[3:, 3:] = matrices_cross_moment_of_inertia[inertia_counter:inertia_counter + 3, :3]

            mass_matrix[3:, :3] = self.get_first_moment_matrix(0,
                                                               matrices_cross_first_moment[row_counter, 1],
                                                               - matrices_cross_first_moment[row_counter, 0])
            list_Jy.append(matrices_cross_first_moment[row_counter, 1])
            mass_matrix[:3, 3:] = -mass_matrix[3:, :3]

            # list_mass_matrix_data.append(np.diag(np.diagonal(mass_matrix)))
            list_mass_matrix_data.append(mass_matrix)
            counter += 6
            inertia_counter += 3
            row_counter += 1

        # # left wing
        for i_material in range(self.num_stiffness_per_wing):
            stiffness_matrix = list_stiffness_matrix[i_material].copy()
            stiffness_matrix[2, 3] *= -1
            stiffness_matrix[3, 2] *= -1

            stiffness_matrix[4, 5] *= -1
            stiffness_matrix[5, 4] *= -1

            stiffness_matrix[1, 2] *= -1
            stiffness_matrix[2, 1] *= -1

            stiffness_matrix[0, 5] *= -1
            stiffness_matrix[5, 0] *= -1

            list_stiffness_matrix.append(stiffness_matrix)

        for ielem in range(self.num_elem['each_wing']):
            mass_matrix = list_mass_matrix_data[self.elem_stiffness[ielem]].copy()
            mass_matrix[3:, :3] = self.correct_first_moment(mass_matrix[3:, :3],
                                                            self.get_chord(self.y[ielem * 2]),
                                                            mass_matrix[0, 0],
                                                            self.elastic_axis[ielem, 2])
            mass_matrix[:3, 3:] = -mass_matrix[3:, :3]
            list_mass_matrix.append(mass_matrix)

        for ielem in range(self.num_elem['each_wing']):
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

        return list_stiffness_matrix, list_mass_matrix

    @staticmethod
    def get_first_moment_matrix(jx, jy, jz):
        matrix = np.zeros((3, 3))
        matrix[0, 1] = -jz
        matrix[1, 0] = jz
        matrix[0, 2] = jy
        matrix[2, 0] = -jy
        matrix[1, 2] = -jx
        matrix[2, 1] = jx
        return matrix

    @staticmethod
    def correct_first_moment(j_matrix, chord, mass_elem, c_n):
        # Correct y coordinate for definition from reference frame
        chi_y = ((0.71 * chord) + j_matrix[0, 2] / mass_elem) - c_n * chord
        jy = chi_y * mass_elem
        j_matrix[0, 2] = jy
        j_matrix[2, 0] = -jy
        return j_matrix

    def get_chord(self, y: float) -> float:
        if y <= self.dimensions['kink_y']:
            return self.dimensions['chord_wing_root']
        else:
            y -= self.dimensions['kink_y']
            x_le = np.tan(self.dimensions['sweep_wing_le']) * y
            x_te = self.dimensions['chord_wing_root'] + np.tan(self.dimensions['sweep_wing_te']) * y
            return abs(x_le - x_te)

    def read_lumped_masses(self):
        file = self.source_directory + '/lumped_masses.csv'
        return pd.read_csv(file, sep=';', header=None)

    def load_y_cross_sections(self):
        # Load data from file
        if self.material == "reference":
            file = self.source_directory + '/dynamics_reference.mat'
        else:
            file = self.source_directory + '/dynamics_tailored.mat'

        matlab_data = load_mat(file)
        nodal_coordinates = matlab_data['dynamics']['str']['xyz']
        n_nodes = int(matlab_data['dynamics']['str']['Nnode'])

        # Transform data
        coords = np.zeros((n_nodes, 3))
        counter = 0
        for irow in range(n_nodes):
            # skip first row
            coords[irow, :] = np.transpose(nodal_coordinates[counter:counter + 3])
            counter += 3
        return coords[1:, 1]


class FlexopAeroelastic(FlexopStructure):
    def __init__(self, case_name, case_route, **kwargs):
        super().__init__(case_name, case_route, **kwargs)
        self.m_wing: int = kwargs.get('m_wing', 6)
        self.m_tail: int = kwargs.get('m_tail', 4)
        self.use_airfoil: bool = kwargs.get('use_airfoil', True)
        self.include_elevators: bool = kwargs.get('include_elevators', True)
        self.cfl1: bool = kwargs.get('cfl1', True)

        self.num_surfaces: int = 2
        if self.use_multibody:
            self.num_surfaces += 2
        if self.include_tail:
            self.num_surfaces += 2

        self.airfoil_distribution = np.zeros((self.num_elem['total'], self.num_node_elem), dtype=int)
        self.surface_distribution = np.zeros(self.num_elem['total'], dtype=int)
        self.surface_m = np.zeros(self.num_surfaces, dtype=int)
        self.aero_node = np.ones(self.num_node['total'], dtype=bool)
        self.aero_twist = np.zeros((self.num_elem['total'], self.num_node_elem))
        self.sweep = np.zeros((self.num_elem['total'], self.num_node_elem))
        self.chord = np.zeros((self.num_elem['total'], self.num_node_elem))
        self.elastic_axis = np.zeros((self.num_elem['total'], self.num_node_elem))

        self.generate_right_wing_aero()
        self.generate_left_wing_aero()
        if self.include_tail:
            self.generate_fuselage_aero()
            self.generate_right_tail_aero()
            self.generate_left_tail_aero()

        self.control_surface: Optional[np.ndarray] = None
        self.control_surface_deflection: Optional[np.ndarray] = None
        self.control_surface_chord: Optional[np.ndarray] = None
        self.control_surface_hinge_coord: Optional[np.ndarray] = None
        self.control_surface_type: Optional[np.ndarray] = None
        if self.include_elevators and self.include_tail:
            self.generate_elevators()

    def generate_right_wing_aero(self) -> None:
        self.surface_m[0] = self.m_wing
        if self.use_multibody:
            self.aero_node[self.node_slice['wing1_tip'][0]] = 0

        wing_chord_nodal = np.zeros((self.num_node['each_wing']))
        wing_chord_nodal[self.node_slice['wing1_inner']] = self.dimensions['chord_wing_root']
        wing_chord_nodal[self.node_slice['wing1_main'][0]:self.node_slice['wing1_tip'][-1] + 1] \
            = (self.dimensions['chord_wing_root'] -
               (((self.y[self.node_slice['wing1_main'][0]:self.node_slice['wing1_tip'][-1] + 1] -
                  self.dimensions['kink_y']) / (self.y[self.node_slice['wing1_tip'][-1]]
                                                - self.dimensions['kink_y'])) * (self.dimensions['chord_wing_root']
                                                                                 - self.dimensions['chord_wing_tip'])))

        wing_twist_nodal = [-self.get_jigtwist_from_y_coord(y) for y in self.y_unswept[self.node_slice['wing1']]]

        wing_ea_elem = np.array([self.spanwise_shear_center[self.elem_stiffness[i_elem]]
                                 for i_elem in range(self.num_elem['each_wing'])])

        self.chord[self.elem_slice['wing1'], 0] = wing_chord_nodal[:-2:2]
        self.chord[self.elem_slice['wing1'], 2] = wing_chord_nodal[1:-1:2]
        self.chord[self.elem_slice['wing1'], 1] = wing_chord_nodal[2::2]
        self.aero_twist[self.elem_slice['wing1'], 0] = wing_twist_nodal[:-2:2]
        self.aero_twist[self.elem_slice['wing1'], 2] = wing_twist_nodal[1:-1:2]
        self.aero_twist[self.elem_slice['wing1'], 1] = wing_twist_nodal[2::2]
        self.elastic_axis[self.elem_slice['wing1'], :] = np.tile(wing_ea_elem, (3, 1)).T

        if self.use_airfoil:
            hinge_node: int = int(self.node_slice['wing1_main'][-1])
            elem_sweep_abs = np.zeros((self.num_elem['each_wing'], 3))
            elem_sweep_abs[self.elem_slice['wing1_tip']] = -self.rigid_sweep_ang

            elem_sweep_rel = np.zeros((self.num_elem['each_wing'], 3))

            num_node_warp_main = self.num_elem_warp_main * 2 + 1
            num_node_warp_tip = self.num_elem_warp_tip * 2 + 1

            rel_sweep_main = np.zeros(self.num_node['wing_main'])
            rel_sweep_main[-num_node_warp_main:] = np.linspace(0, -0.5 * self.rigid_sweep_ang, num_node_warp_main)

            rel_sweep_tip = np.zeros(self.num_node['wing_tip'])
            rel_sweep_tip[:num_node_warp_tip] = np.linspace(0.5 * self.rigid_sweep_ang, 0, num_node_warp_tip)

            elem_sweep_rel[self.elem_slice['wing1_main'], 0] = rel_sweep_main[:-2:2]
            elem_sweep_rel[self.elem_slice['wing1_main'], 2] = rel_sweep_main[1:-1:2]
            elem_sweep_rel[self.elem_slice['wing1_main'], 1] = rel_sweep_main[2::2]

            elem_sweep_rel[self.elem_slice['wing1_tip'], 0] = rel_sweep_tip[:-2:2]
            elem_sweep_rel[self.elem_slice['wing1_tip'], 2] = rel_sweep_tip[1:-1:2]
            elem_sweep_rel[self.elem_slice['wing1_tip'], 1] = rel_sweep_tip[2::2]

            self.sweep[self.elem_slice['wing1'], :] = elem_sweep_abs + elem_sweep_rel
            self.chord[self.elem_slice['wing1'], :] /= np.cos(elem_sweep_rel)

    def generate_left_wing_aero(self) -> None:
        if self.use_multibody:
            self.surface_m[1] = self.m_wing
            self.surface_distribution[self.elem_slice['wing2']] = 1
            self.aero_node[self.node_slice['wing2_tip'][0]] = 0
        else:
            self.surface_m[1] = self.m_wing
            self.surface_distribution[self.elem_slice['wing2']] = 1

        self.chord[self.elem_slice['wing2'], :] = self.chord[self.elem_slice['wing1'], :]
        self.aero_twist[self.elem_slice['wing2'], :] = self.aero_twist[self.elem_slice['wing1'], :]
        self.elastic_axis[self.elem_slice['wing2'], :] = self.elastic_axis[self.elem_slice['wing1'], :]
        self.sweep[self.elem_slice['wing2'], :] = -self.sweep[self.elem_slice['wing1'], :]

    def generate_fuselage_aero(self) -> None:
        self.aero_node[self.node_slice['fuselage_front'][1]:self.node_slice['fuselage_rear'][-1] + 1] = 0
        self.aero_node[self.node_slice['fuselage_main'][-1]] = 1
        self.chord[self.elem_slice['fuselage_main'][-1]][1] = self.dimensions['chord_tail_root']
        self.chord[self.elem_slice['fuselage_rear'][0]][0] = self.dimensions['chord_tail_root']
        self.elastic_axis[self.elem_slice['fuselage_main'][-1]][1] = self.dimensions['ea_tail']
        self.elastic_axis[self.elem_slice['fuselage_rear'][0]][0] = self.dimensions['ea_tail']
        self.airfoil_distribution[self.elem_slice['fuselage_main'][-1]][1] = 1
        self.airfoil_distribution[self.elem_slice['fuselage_rear'][0]][0] = 1
        self.surface_distribution[self.elem_slice['fuselage']] = -1

    def generate_right_tail_aero(self) -> None:
        if self.use_multibody:
            self.surface_distribution[self.elem_slice['tail1']] = 2
            self.surface_m[2] = self.m_tail
        else:
            self.surface_distribution[self.elem_slice['tail1']] = 2
            self.surface_m[2] = self.m_tail

        tail_chord_nodal = np.linspace(self.dimensions['chord_tail_root'],
                                       self.dimensions['chord_tail_tip'],
                                       self.num_node['each_tail'])

        self.chord[self.elem_slice['tail1'], 0] = tail_chord_nodal[:-2:2]
        self.chord[self.elem_slice['tail1'], 2] = tail_chord_nodal[1:-1:2]
        self.chord[self.elem_slice['tail1'], 1] = tail_chord_nodal[2::2]
        self.elastic_axis[self.elem_slice['tail1'], :] = self.dimensions['ea_tail']
        self.airfoil_distribution[self.elem_slice['tail1'], :] = 1

    def generate_left_tail_aero(self) -> None:
        if self.use_multibody:
            self.surface_distribution[self.elem_slice['tail2']] = 3
            self.surface_m[3] = self.m_tail
        else:
            self.surface_distribution[self.elem_slice['tail2']] = 3
            self.surface_m[3] = self.m_tail

        self.chord[self.elem_slice['tail2'], :] = self.chord[self.elem_slice['tail1'], :]
        self.elastic_axis[self.elem_slice['tail2'], :] = self.dimensions['ea_tail']
        self.airfoil_distribution[self.elem_slice['tail2'], :] = 1

    def generate_elevators(self) -> None:
        self.control_surface = np.full((self.num_elem['total'], self.num_node_elem), -1, dtype=int)
        self.control_surface_deflection = np.full(2, self.input_settings.get('elevator_angle', 0.), dtype=float)
        self.control_surface_chord = np.full(2, self.m_tail // 2, dtype=int)
        self.control_surface_hinge_coord = np.zeros(2)
        self.control_surface_type = np.zeros(2, dtype=int)

        self.control_surface[self.elem_slice['tail1'], :] = 0
        self.control_surface[self.elem_slice['tail2'], :] = 0

    def get_jigtwist_from_y_coord(self, y_coord):
        y_coord = abs(y_coord)
        df_jig_twist = pd.read_csv(self.source_directory + '/jig_twist.csv', sep=';')
        idx_closest_value = self.find_index_of_closest_entry(df_jig_twist.iloc[:, 0], y_coord)
        if self.material == "reference":
            column = 1
        else:
            column = 2
        if idx_closest_value == df_jig_twist.shape[0]:
            idx_adjacent = idx_closest_value - 1
        elif idx_closest_value == 0 or df_jig_twist.iloc[idx_closest_value, 0] < y_coord:
            idx_adjacent = idx_closest_value + 1
        else:
            idx_adjacent = idx_closest_value - 1

        jig_twist_interp = df_jig_twist.iloc[idx_closest_value, column] + (
                (y_coord - df_jig_twist.iloc[idx_closest_value, 0])
                / (df_jig_twist.iloc[idx_adjacent, 0] - df_jig_twist.iloc[idx_closest_value, 0])
                * (df_jig_twist.iloc[idx_adjacent, column] - df_jig_twist.iloc[idx_closest_value, column]))
        # when the denominator of the interpolation is zero
        if np.isnan(jig_twist_interp):
            jig_twist_interp = df_jig_twist.iloc[idx_closest_value, 1]
        return np.deg2rad(jig_twist_interp)

    def generate_h5_aero(self):
        file_route = self.case_route + '/' + self.case_name + '.aero.h5'

        try:
            os.remove(file_route)
            print("Deleted AERO H5 file")
        except OSError:
            pass

        with h5.File(file_route, 'a') as h5file:
            airfoils_group = h5file.create_group('airfoils')
            if self.use_airfoil:
                airfoils_group.create_dataset('0', data=np.column_stack(self.load_airfoil_data_from_file()))
            else:
                airfoils_group.create_dataset('0', data=np.column_stack(self.generate_naca_camber(p=0, m=0)))
            airfoils_group.create_dataset('1', data=np.column_stack(self.generate_naca_camber(p=0, m=0)))
            chord_input = h5file.create_dataset('chord', data=self.chord)
            chord_input.attrs['units'] = 'm'
            twist_input = h5file.create_dataset('twist', data=self.aero_twist)
            twist_input.attrs['units'] = 'rad'
            sweep_input = h5file.create_dataset('sweep', data=self.sweep)
            sweep_input.attrs['units'] = 'rad'
            h5file.create_dataset('airfoil_distribution', data=self.airfoil_distribution)
            h5file.create_dataset('surface_distribution', data=self.surface_distribution)
            h5file.create_dataset('surface_m', data=self.surface_m)
            h5file.create_dataset('aero_node', data=self.aero_node)
            h5file.create_dataset('elastic_axis', data=self.elastic_axis)
            h5file.create_dataset('m_distribution', data="uniform".encode('ascii', 'ignore'))

            if self.include_elevators and self.include_tail:
                h5file.create_dataset('control_surface', data=self.control_surface)
                h5file.create_dataset('control_surface_deflection', data=self.control_surface_deflection)
                h5file.create_dataset('control_surface_chord', data=self.control_surface_chord)
                h5file.create_dataset('control_surface_hinge_coord', data=self.control_surface_hinge_coord)
                h5file.create_dataset('control_surface_type', data=self.control_surface_type)

    def generate_h5(self):
        self.generate_h5_fem()
        self.generate_h5_aero()
        if self.use_multibody:
            self.generate_h5_mb()

    @staticmethod
    def generate_naca_camber(m=0, p=0):
        mm = m * 1e-2
        p = p * 1e-1

        def naca(x, mm_, p_):
            if x < 1e-6:
                return 0.0
            elif x < p_:
                return mm_ / (p_ * p_) * (2 * p_ * x - x * x)
            elif p_ < x < 1 + 1e-6:
                return mm_ / ((1 - p_) * (1 - p_)) * (1 - 2 * p_ + 2 * p_ * x - x * x)

        x_vec = np.linspace(0, 1, 1000)
        y_vec = np.array([naca(x, mm, p) for x in x_vec])
        return x_vec, y_vec

    def load_airfoil_data_from_file(self):
        file = self.source_directory + "/camber_line_airfoils.csv"
        camber_line = pd.read_csv(file, sep=";")
        return np.array(camber_line.iloc[:, 0]), np.array(camber_line.iloc[:, 1])

    def add_constraint(self, constraint_name: str, **constraint_args: dict[str, Any]) -> None:
        if not self.use_multibody:
            print("Skipping constraint as multibody is not enabled")
            return

        self.constraint_settings.update(constraint_args)

        match constraint_name:
            case 'free':
                pass
            case 'free_hinge':
                flare_ang: float = float(constraint_args.get('flare_angle', 0.))
                sweep_ang: float = self.dimensions['sweep_wing_qchord']
                hinge1_node = int(self.node_slice['wing1_main'][-1])
                hinge2_node = hinge1_node * 2

                e1 = np.array((1., 0., 0.))
                angle_b = np.pi / 2. - sweep_ang
                rmat_b = np.array(((np.cos(angle_b), -np.sin(angle_b), 0.),
                                 (np.sin(angle_b), np.cos(angle_b), 0.),
                                 (0., 0., 1.)))

                rmat_f = np.array(((np.cos(flare_ang), -np.sin(flare_ang), 0.),
                                   (np.sin(flare_ang), np.cos(flare_ang), 0.),
                                   (0., 0., 1.)))

                b1_axis = rmat_b.T @ e1
                b2_axis = rmat_b @ e1

                self.constraints.append({'behaviour': 'hinge_node_FoR',
                                         'body_FoR': 1,
                                         'body': 0,
                                         'node_in_body': hinge1_node,
                                         'rot_axisA2': rmat_f.T @ e1,
                                         'rot_axisB': rmat_f.T @ b1_axis,
                                         })
                self.constraints.append({'behaviour': 'hinge_node_FoR',
                                         'body_FoR': 2,
                                         'body': 0,
                                         'node_in_body': hinge2_node,
                                         'rot_axisA2': rmat_f @ e1,
                                         'rot_axisB': rmat_f @ b2_axis,
                                         })

                hinge_x = self.x[self.node_slice['wing1_main'][-1]]
                hinge_y = self.y[self.node_slice['wing1_main'][-1]]

                # equation of line of leading edge in form y = mx + c
                test1 = np.pi / 2. - self.dimensions['sweep_wing_le']
                le_grad = np.tan(np.pi / 2. - self.dimensions['sweep_wing_le'])
                le_kink_x = -self.chord[0, 0] * self.elastic_axis[0, 0]
                le_c = self.dimensions['kink_y'] - le_kink_x * le_grad

                # equation of line of trailing edge in form y = mx + c
                te_grad = np.tan(np.pi / 2. - self.dimensions['sweep_wing_te'])
                te_kink_x = self.chord[0, 0] * (1. - self.elastic_axis[0, 0])
                te_c = self.dimensions['kink_y'] - te_kink_x * te_grad

                # equation of line of hinge in form y = mx + c
                h_grad = np.tan(-flare_ang)
                h_c = hinge_y - hinge_x * h_grad

                # solve for hinge leading and trailing edge coordinates
                leh_x = (h_c - le_c) / (le_grad - h_grad)
                leh_y = h_grad * leh_x + h_c

                teh_x = (h_c - te_c) / (te_grad - h_grad)
                teh_y = h_grad * teh_x + h_c

                hinge_chord = np.sqrt((leh_y - teh_y) ** 2 + (leh_x - teh_x) ** 2)
                hinge_chord_le = np.sqrt((leh_y - hinge_y) ** 2 + (leh_x - hinge_x) ** 2)
                hinge_elastic_axis = hinge_chord_le / hinge_chord

                self.chord[self.elem_slice['wing1_main'][-1], 1] = hinge_chord
                self.chord[self.elem_slice['wing1_tip'][0], 0] = hinge_chord
                self.chord[self.elem_slice['wing2_main'][-1], 1] = hinge_chord
                self.chord[self.elem_slice['wing2_tip'][0], 0] = hinge_chord

                self.elastic_axis[self.elem_slice['wing1_main'][-1], 1] = hinge_elastic_axis
                self.elastic_axis[self.elem_slice['wing1_tip'][0], 0] = hinge_elastic_axis
                self.elastic_axis[self.elem_slice['wing2_main'][-1], 1] = hinge_elastic_axis
                self.elastic_axis[self.elem_slice['wing2_tip'][0], 0] = hinge_elastic_axis

                self.sweep[self.elem_slice['wing1_main'][-1], 1] = -flare_ang
                self.sweep[self.elem_slice['wing1_tip'][0], 0] = -flare_ang
                self.sweep[self.elem_slice['wing2_main'][-1], 1] = flare_ang
                self.sweep[self.elem_slice['wing2_tip'][0], 0] = flare_ang

                self.airfoil_distribution[self.elem_slice['wing1_main'][-1], 1] = 1
                self.airfoil_distribution[self.elem_slice['wing1_tip'][0], 0] = 1
                self.airfoil_distribution[self.elem_slice['wing2_main'][-1], 1] = 1
                self.airfoil_distribution[self.elem_slice['wing2_tip'][0], 0] = 1

                # self.aero_node[self.node_slice['wing1_tip'][0]] = 0
                # self.aero_node[self.node_slice['wing2_tip'][0]] = 0

            case 'prescribed_hinge':
                hinge_node: int = int(self.node_slice['wing1_main'][-1])
                hinge_main_elem = self.elem_slice['wing1_main'][-1]
                hinge_tip_elem = self.elem_slice['wing1_tip'][0]

                rhs_warp_factor = np.zeros((self.num_elem['total'], 3))
                lhs_warp_factor = np.zeros((self.num_elem['total'], 3))
                num_node_warp_main = self.num_elem_warp_main * (self.num_node_elem - 1) + 1
                num_node_warp_tip = self.num_elem_warp_tip * (self.num_node_elem - 1) + 1

                nodal_warp_main = np.linspace(0., 0.5, num_node_warp_main)

                rhs_warp_factor[hinge_main_elem - self.num_elem_warp_main + 1:hinge_main_elem + 1, 0] \
                    = nodal_warp_main[:-2:2]
                rhs_warp_factor[hinge_main_elem - self.num_elem_warp_main + 1:hinge_main_elem + 1, 2] \
                    = nodal_warp_main[1:-1:2]
                rhs_warp_factor[hinge_main_elem - self.num_elem_warp_main + 1:hinge_main_elem + 1, 1] \
                    = nodal_warp_main[2::2]

                nodal_warp_tip = np.linspace(-0.5, 0., num_node_warp_tip)

                rhs_warp_factor[hinge_tip_elem:hinge_tip_elem + self.num_elem_warp_tip, 0] \
                    = nodal_warp_tip[:-2:2]
                rhs_warp_factor[hinge_tip_elem:hinge_tip_elem + self.num_elem_warp_tip, 2] \
                    = nodal_warp_tip[1:-1:2]
                rhs_warp_factor[hinge_tip_elem:hinge_tip_elem + self.num_elem_warp_tip, 1] \
                    = nodal_warp_tip[2::2]

                lhs_warp_factor[self.elem_slice['wing2'], :] = rhs_warp_factor[self.elem_slice['wing1'], :]

                self.constraints.append({'behaviour': 'control_node_FoR_rot_vel',
                                         'controller_id': 'controller_rhs',
                                         'body_FoR': 1,
                                         'body': 0,
                                         'node_in_body': hinge_node,
                                         'aerogrid_warp_factor': rhs_warp_factor
                                         })
                self.constraints.append({'behaviour': 'control_node_FoR_rot_vel',
                                         'controller_id': 'controller_lhs',
                                         'body_FoR': 2,
                                         'body': 0,
                                         'node_in_body': 2 * hinge_node,
                                         'aerogrid_warp_factor': lhs_warp_factor
                                         })
                u_rhs = self.constraint_settings['u_rhs']
                u_dot_rhs = self.constraint_settings['u_dot_rhs']
                u_lhs = self.constraint_settings['u_lhs']
                u_dot_lhs = self.constraint_settings['u_dot_lhs']

                z_ang_offset = np.pi / 2. - self.dimensions['sweep_wing_qchord']

                u_rhs[:, 2] += z_ang_offset
                u_lhs[:, 2] += z_ang_offset


                np.save(self.constraint_settings['input_angle_rhs_dir'], u_rhs)
                np.save(self.constraint_settings['input_velocity_rhs_dir'], u_dot_rhs)
                np.save(self.constraint_settings['input_angle_lhs_dir'], u_lhs)
                np.save(self.constraint_settings['input_velocity_lhs_dir'], u_dot_lhs)

            case 'fully_constrained':
                hinge_node: int = int(self.node_slice['wing1_main'][-1])
                self.constraints.append({'behaviour': 'fully_constrained_node_FoR',
                                         'body_FoR': 1,
                                         'body': 0,
                                         'node_in_body': hinge_node,
                                         })
                self.constraints.append({'behaviour': 'fully_constrained_node_FoR',
                                         'body_FoR': 2,
                                         'body': 0,
                                         'node_in_body': 2 * hinge_node,
                                         })
            case 'clamped':
                self.constraints.append({'behaviour': 'fully_constrained_FoR',
                                         'body_FoR': 0})
            case _:
                raise KeyError(f'Unknown constraint {constraint_name}')
        self.num_constraints = len(self.constraints)

    def generate_settings(self):
        u_inf = self.input_settings.get('u_inf', 40.)
        u_inf_dir = self.input_settings.get('u_inf_dir', np.array((1., 0., 0.)))
        rho = self.input_settings.get('rho', 1.225)
        m_star_fact = self.input_settings.get('m_star_fact', 3)
        n_tstep = self.input_settings['n_tstep']
        dt = self.input_settings['dt']
        use_aero = self.input_settings.get('use_aero', True)
        steady_aero = self.input_settings.get('steady_aero', False)

        self.settings['SHARPy'] = {
            'flow': self.input_settings['flow'],
            'case': self.case_name,
            'route': self.case_route,
            'write_screen': True,
            'write_log': True
        }

        self.settings['BeamLoader'] = {
            'unsteady': True,
            'orientation': self.in_quat}

        self.settings['AerogridLoader'] = {
            'unsteady': True,
            'aligned_grid': False,
            'mstar': int(m_star_fact * self.m_wing),
            'freestream_dir': u_inf_dir,
            'wake_shape_generator': 'StraightWake',
            'wake_shape_generator_input': {'u_inf': u_inf,
                                           'u_inf_direction': u_inf_dir,
                                           'dt': dt}
        }

        self.settings['AerogridPlot'] = {'include_rbm': True,
                                         'include_applied_forces': True,
                                         'minus_m_star': 0}

        self.settings['BeamPlot'] = {'include_rbm': True,
                                     'include_applied_forces': True}

        if not self.cfl1:
            self.settings['AerogridLoader']['wake_shape_generator_input'].update({'dx1': u_inf * dt,
                                                                                  'ndx1': 6,
                                                                                  'r': 1.2,
                                                                                  'dxmax': 20 * u_inf * dt})

        self.settings['StaticCoupled'] = {
            'print_info': True,
            'max_iter': 200,
            'n_load_steps': 1,
            'tolerance': 1e-10,
            'relaxation_factor': 0,
            'aero_solver': 'StaticUvlm',
            'aero_solver_settings': {
                'rho': rho,
                'print_info': False,
                'horseshoe': False,
                'num_cores': 8,
                'n_rollup': 0,
                'cfl1': self.cfl1,
                'rollup_dt': dt,
                'rollup_aic_refresh': 1,
                'rollup_tolerance': 1e-4,
                'velocity_field_generator': 'SteadyVelocityField',
                'velocity_field_input': {
                    'u_inf': u_inf,
                    'u_inf_direction': u_inf_dir}},
            'structural_solver': 'NonLinearStatic',
            'structural_solver_settings': {'print_info': False,
                                           'max_iterations': 150,
                                           'num_load_steps': 4,
                                           'delta_curved': 1e-1,
                                           'min_delta': 1e-10,
                                           'gravity_on': self.gravity_on,
                                           'gravity': 9.81}}

        self.settings['StaticTrim'] = {'solver': 'StaticCoupled',
                                       'solver_settings': self.settings['StaticCoupled'],
                                       'thrust_nodes': [0],
                                       'save_info': True}

        self.settings['NonLinearDynamicPrescribedStep'] = {'print_info': False,
                                                           'max_iterations': 950,
                                                           'delta_curved': 1e-1,
                                                           'min_delta': 1e3,
                                                           'newmark_damp': 5e-3,
                                                           'gravity_on': True,
                                                           'gravity': 9.81,
                                                           'num_steps': n_tstep,
                                                           'dt': dt}

        self.settings['NonLinearDynamicCoupledStep'] = {'print_info': False,
                                                           'max_iterations': 950,
                                                           'delta_curved': 1e-1,
                                                           'min_delta': 1e3,
                                                           'newmark_damp': 5e-3,
                                                           'gravity_on': True,
                                                           'gravity': 9.81,
                                                           'num_steps': n_tstep,
                                                           'dt': dt,
                                                            'initial_velocity': -u_inf,
                                                            'initial_velocity_direction': u_inf_dir}

        self.settings['NonLinearDynamicMultibodyJAX'] = {'gravity_on': True,
                                                         'gravity': 9.81,
                                                         'num_steps': n_tstep,
                                                         'initial_velocity': u_inf if self.free else 0.,
                                                         'time_integrator': 'NewmarkBetaJAX',
                                                         'time_integrator_settings': {'newmark_damp': 0.0,
                                                                                      'dt': dt}}

        self.settings['NonLinearDynamicMultibody'] = {'gravity_on': True,
                                                    'gravity': 9.81,
                                                      'initial_velocity': u_inf if self.free else 0.,
                                                         'time_integrator': 'NewmarkBeta',
                                                         'time_integrator_settings': {'newmark_damp': 0.0,
                                                                                      'dt': dt}}

        self.settings['StepUvlm'] = {'print_info': True,
                                     'num_cores': 8,
                                     'cfl1': self.cfl1,
                                     'convection_scheme': 2,
                                     'velocity_field_generator': 'GustVelocityField',
                                     'velocity_field_input':
                                         {'u_inf': u_inf * int(not self.free),
                                          'u_inf_direction': u_inf_dir,
                                          'gust_shape': '1-cos',
                                          'gust_parameters':
                                              {'gust_length': self.input_settings['gust_length'],
                                               'gust_intensity': self.input_settings['gust_intensity'] * u_inf},
                                          'offset': self.input_settings.get('gust_offset', 0.),
                                          'relative_motion': not self.free},
                                     'rho': rho,
                                     'n_time_steps': n_tstep,
                                     'dt': dt,
                                     'gamma_dot_filtering': 3}

        self.settings['StaticUvlm'] = {'print_info': True,
                                       'horseshoe': True,
                                       'num_cores': 8,
                                       'velocity_field_generator': 'SteadyVelocityField',
                                       'velocity_field_input':
                                           {'u_inf': u_inf * int(not self.free),
                                            'u_inf_direction': u_inf_dir},
                                       'rollup_dt': dt,
                                       'rho': rho,
                                       'map_forces_on_struct': True
                                       }

        self.settings['AeroForcesCalculator'] = {'write_text_file': True}

        self.settings['DynamicCoupled'] = {'print_info': True,
                                           'structural_substeps': 0,
                                           'dynamic_relaxation': True,
                                           'cleanup_previous_solution': True,
                                           'fsi_substeps': 100,
                                           'minimum_steps': 1,
                                           'relaxation_steps': 150,
                                           'final_relaxation_factor': 0.,
                                           'n_time_steps': n_tstep,
                                           'dt': dt,
                                           'include_unsteady_force_contribution': True,
                                           'postprocessors': ['BeamLoads'],
                                           'postprocessors_settings': {'BeamLoads': {}}}

        if use_aero:
            if steady_aero:
                self.settings['DynamicCoupled']['aero_solver'] = 'StaticUvlm'
                self.settings['DynamicCoupled']['aero_solver_settings'] = self.settings['StaticUvlm']
            else:
                self.settings['DynamicCoupled']['aero_solver'] = 'StepUvlm'
                self.settings['DynamicCoupled']['aero_solver_settings'] = self.settings['StepUvlm']
        else:
            self.settings['DynamicCoupled']['aero_solver'] = 'NoAero'
            self.settings['DynamicCoupled']['aero_solver_settings'] = {}


        if self.input_settings.get('save_vtu', True):
            stride = self.input_settings.get('plot_stride', 1)
            self.settings['DynamicCoupled']['postprocessors'].extend(['AerogridPlot', 'BeamPlot'])
            self.settings['DynamicCoupled']['postprocessors_settings'].update({'BeamPlot': {'include_rbm': True,
                                                                                    'include_applied_forces': True,
                                                                                            'stride': stride},
                                                                       'AerogridPlot': {
                                                                           'u_inf': u_inf,
                                                                           'include_rbm': True,
                                                                           'include_applied_forces': True,
                                                                           'minus_m_star': 0,
                                                                       'stride': stride}})

        if self.use_multibody and self.use_jax:
            self.settings['DynamicCoupled'].update({'structural_solver': 'NonLinearDynamicMultibodyJAX'})
            self.settings['DynamicCoupled'].update(
                {'structural_solver_settings': self.settings['NonLinearDynamicMultibodyJAX']})
        elif self.use_multibody and not self.use_jax:
            self.settings['DynamicCoupled'].update({'structural_solver': 'NonLinearDynamicMultibody'})
            self.settings['DynamicCoupled'].update(
                {'structural_solver_settings': self.settings['NonLinearDynamicMultibody']})
        elif self.free:
            self.settings['DynamicCoupled'].update({'structural_solver': 'NonLinearDynamicCoupledStep'})
            self.settings['DynamicCoupled'].update(
                {'structural_solver_settings': self.settings['NonLinearDynamicCoupledStep']})
        else:
            self.settings['DynamicCoupled'].update({'structural_solver': 'NonLinearDynamicPrescribedStep'})
            self.settings['DynamicCoupled'].update(
                {'structural_solver_settings': self.settings['NonLinearDynamicPrescribedStep']})

        if self.constraint_settings.get('use_control', False):
            self.settings['DynamicCoupled']['controller_id'] = {'controller_rhs': 'MultibodyController',
                                                                'controller_lhs': 'MultibodyController'}
            self.settings['DynamicCoupled']['controller_settings'] \
                = {'controller_rhs': {'ang_history_input_file': self.constraint_settings['input_angle_rhs_dir'],
                                      'ang_vel_history_input_file': self.constraint_settings['input_velocity_rhs_dir'],
                                      'dt': dt},
                   'controller_lhs': {'ang_history_input_file': self.constraint_settings['input_angle_lhs_dir'],
                                      'ang_vel_history_input_file': self.constraint_settings['input_velocity_lhs_dir'],
                                      'dt': dt}}

        self.settings['BeamLoads'] = {}

        self.settings['Modal'] = {'NumLambda': 20,
                                  'rigid_body_modes': False,
                                  'print_matrices': False,
                                  'save_data': False,
                                  'continuous_eigenvalues': False,
                                  'dt': 0,
                                  'plot_eigenvalues': False,
                                  'max_rotation_deg': 15.,
                                  'max_displacement': 0.15,
                                  'write_modes_vtk': True,
                                  'use_undamped_modes': True}
        self.settings['DynamicTrim'] = {'solver': 'DynamicCoupled',
                                        'solver_settings': self.settings['DynamicCoupled'],
                                        'speed_up_factor': 1.
                                        }
        if 'DynamicTrim' in self.input_settings['flow']:
            self.settings['DynamicTrim']['solver_settings']['structural_solver_settings'].update({'dyn_trim': True})

        config = configobj.ConfigObj()
        config.filename = self.case_route + '/' + self.case_name + '.sharpy'

        for k, v in self.settings.items():
            config[k] = v
        config.write()


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
