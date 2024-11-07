#! /usr/bin/env python3
import h5py as h5
import configobj
import numpy as np
from structure import FLEXOPStructure
from fuselage import FLEXOPFuselage
from aero import FLEXOPAero
from aero import area_ref
import os
import sharpy.sharpy_main
import subprocess

FLEXOP_DIRECTORY = os.path.dirname(os.path.realpath(__file__)) + '/'


class FLEXOP:

    def __init__(self, case_name, case_route, output_route):
        self.case_name = case_name
        self.case_route = case_route
        self.output_route = output_route

        self.structure = None
        self.aero = None
        self.fuselage = None

        self.settings = None

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.source_directory = os.path.join(dir_path, 'aeroelastic_properties')
        
        print(f'Looking for source files in {self.source_directory}')

    def init_aeroelastic(self,**kwargs):
        m = kwargs.get('m', 4)
        self.clean()
        self.init_structure()
        self.init_aero(m=m, **kwargs)
        if kwargs.get('nonlifting_interactions', False):
            self.init_fuselage(**kwargs)

    def init_structure(self, **kwargs):
        self.structure = FLEXOPStructure(self.case_name, self.case_route, self.source_directory, **kwargs)

    def init_aero(self, m, **kwargs):
        self.aero = FLEXOPAero(m, self.structure, self.case_name, self.case_route, self.source_directory,**kwargs)

    def init_fuselage(self, m, **kwargs):
        self.fuselage = FLEXOPFuselage(m, self.structure, self.case_name, self.case_route, self.source_directory, **kwargs)

    def set_flight_controls(self, thrust=0., elevator=0.):
        self.structure.set_thrust(thrust)

        if self.aero is not None:
            self.aero.cs_deflection = elevator
    @property
    def reference_area(self):
        return area_ref

    def generate(self):

        if not os.path.isdir(self.case_route):
            os.makedirs(self.case_route)

        self.structure.generate()

        if self.aero is not None:
            self.aero.generate()
        if self.fuselage is not None:
            self.fuselage.generate()

    def create_settings(self, settings):
        file_name = self.case_route + '/' + self.case_name + '.sharpy'
        config = configobj.ConfigObj()
        config.filename = file_name
        for k, v in settings.items():
            config[k] = v
        config.write()
        self.settings = settings

        # git model filename 
        git_file_name = self.case_route + '/' + 'flexop_model_info' + self.case_name + '.txt'
        with open(git_file_name, 'w') as f:
            f.write(print_git_status())

    def clean(self):
        list_files = ['.fem.h5', '.aero.h5', '.nonlifting_body.h5', '.dyn.h5', '.mb.h5', '.sharpy', '.flightcon.txt']
        for file in list_files:
            path_file = self.case_route + '/' + self.case_name + file
            if os.path.isfile(path_file):
                os.remove(path_file)

    def run(self):
        sharpy.sharpy_main.main(['', self.case_route + '/' + self.case_name + '.sharpy'])


# version tracker and output
def get_git_revision_hash(di=FLEXOP_DIRECTORY):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=di).strip().decode('utf-8')


def get_git_revision_short_hash(di=FLEXOP_DIRECTORY):
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=di).strip().decode('utf-8')


def get_git_revision_branch(di=FLEXOP_DIRECTORY):
    return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=di).strip().decode('utf-8')


def get_git_tag(di=FLEXOP_DIRECTORY):
    return subprocess.check_output(['git', 'describe'], cwd=di).strip().decode('utf-8')


def print_git_status():
    try:
        version_msg = get_git_tag()
    except subprocess.CalledProcessError:
        version_msg = 'unreleased'

    return ('FLEXOP Model Git info:'
            '\tThe branch being run is ' + get_git_revision_branch() +
            '\n' +
            '\tVersion: ' + version_msg + '\n' +
            '\tCommit hash: ' + get_git_revision_short_hash() + '\n'
            '\tFull hash: ' + get_git_revision_hash())

if __name__ == '__main__':
    print(FLEXOP_DIRECTORY)
    print(print_git_status())
