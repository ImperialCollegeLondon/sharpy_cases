from structure import PazyStructure
import sharpy.utils.settings as settings
import aero
import configobj
import sharpy.utils.cout_utils as cout


class PazyWing:

    model_settings_default = dict()
    model_settings_types = dict()
    model_settings_options = dict()
    model_settings_description = dict()

    model_settings_default['skin_on'] = False
    model_settings_types['skin_on'] = 'bool'

    model_settings_default['discretisation_method'] = 'michigan'
    model_settings_types['discretisation_method'] = 'str'
    model_settings_options['discretisation_method'] = ['michigan', 'even', 'fine_root_tip']

    model_settings_types['num_elem'] = 'int'
    model_settings_default['num_elem'] = 2
    model_settings_description['num_elem'] = ('If discretisation is ``michigan`` then it corresponds to how many times '
                                              'the original number of elements is replicated, else is the number of '
                                              'elements')

    model_settings_types['surface_m'] = 'int'
    model_settings_default['surface_m'] = 4

    model_settings_types['num_surfaces'] = 'int'
    model_settings_default['num_surfaces'] = 2

    model_settings_types['sweep_angle'] = 'float'
    model_settings_default['sweep_angle'] = 0.0

    model_settings_types['sigma'] = 'float'
    model_settings_default['sigma'] = 1.0

    def __init__(self, case_name, case_route='./', in_settings=None):
        self.case_name = case_name
        self.case_route = case_route

        cout.start_writer()

        self.settings = in_settings if in_settings is not None else dict()

        settings.to_custom_types(self.settings, self.model_settings_types, self.model_settings_default,
                                 self.model_settings_options, no_ctype=True)

        self.structure = None
        self.aero = None

        self.config = configobj.ConfigObj()
        self.config.filename = self.case_route + '/' + self.case_name + '.sharpy'

    def save_config(self):
        self.config.write()

    @staticmethod
    def get_ea_reference_line():
        return 0.4410

    def generate_structure(self):
        pazy = PazyStructure(**self.settings)
        pazy.generate()

        self.structure = pazy

    def generate_aero(self):
        pazy_aero = aero.PazyAero(main_ea=self.get_ea_reference_line(),
                                  pazy_structure=self.structure,
                                  **self.settings)

        pazy_aero.generate_aero()

        self.aero = pazy_aero

    def create_aeroelastic_model(self):
        self.generate_structure()
        self.structure.mirror_wing()
        self.generate_aero()

    def save_files(self):
        self.structure.save_fem_file(case_name=self.case_name, case_route=self.case_route)
        if self.aero is not None:
            self.aero.save_files(case_name=self.case_name, case_route=self.case_route)
