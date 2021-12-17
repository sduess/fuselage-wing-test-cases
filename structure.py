#! /usr/bin/env python3
import h5py as h5
import pandas as pd
import numpy as np
from scipy.io import loadmat, matlab

# Define all main parameters in the aircraft 
# MODEL GEOMETRY

# Fuselage information
offset_fuselage_vertical = 0
offset_wing_nose = 4.5
sigma_fuselage = 10
m_bar_fuselage = 0.3
j_bar_fuselage = 0.1

thickness_ratio_ellipse = 7.5
length_fuselage = 10
radius_fuselage = length_fuselage/thickness_ratio_ellipse/2
offset_wing_fuselage_vertical = 0

aspect_ratio = 6.
chord_main =radius_fuselage*2/1.328
span_main = chord_main*aspect_ratio/2 #8.8/2

# Fuselage = 
y_coord_junction = radius_fuselage



class Structure:

    def __init__(self, case_name, case_route, **kwargs):
        self.sigma = kwargs.get('sigma', 1)
        self.n_elem_multiplier = kwargs.get('n_elem_multiplier', 8)
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
        self.n_elem_fuselage = None

        self.n_node_main = None
        self.n_node_fuselage = None

        self.span_main = span_main
        self.chord_main = chord_main
        self.y_coord_junction = y_coord_junction # Radius fuselage at wing
        self.length_fuselage = length_fuselage

        self.lifting_only = kwargs.get('lifting_only', True)

        self.sweep_quarter_chord = sweep_quarter_chord

    def generate(self):
        # Set Elements

        self.n_elem_main =  int(2*self.n_elem_multiplier) 
        self.n_elem_fuselage = int(10*self.n_elem_multiplier_fuselage) + 1

        # lumped masses
        n_lumped_mass =  1
        lumped_mass_nodes = np.zeros((n_lumped_mass, ), dtype=int)
        lumped_mass = np.zeros((n_lumped_mass, ))
        lumped_mass[0] = 0 # mass_take_off
        lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
        lumped_mass_position = np.zeros((n_lumped_mass, 3))
        
        # total number of elements
        self.n_elem = self.n_elem_main + self.n_elem_main
        if not self.wing_only:
            self.n_elem += self.n_elem_fuselage

        # number of nodes per part
        self.n_node_main = self.n_elem_main*(self.n_node_elem - 1) + 1
        self.n_node_fuselage = (self.n_elem_fuselage+1)*(self.n_node_elem - 1) -1

        # total number of nodes
        self.n_node = self.n_node_main + self.n_node_main - 1
        if not self.wing_only:
            self.n_node += self.n_node_fuselage - 1


        # Aeroelastic properties
        n_stiffness = self.n_stiffness_per_wing * 2
        n_mass = n_stiffness
        
        m_bar_fuselage = 0.3*1.5
        j_bar_fuselage = 0.08


        # beam
        self.x = np.zeros((self.n_node, ))
        self.y = np.zeros((self.n_node, ))
        self.z = np.zeros((self.n_node, ))
        structural_twist = np.zeros((self.n_elem, self.n_node_elem))
        beam_number = np.zeros((self.n_elem, ), dtype=int)
        frame_of_reference_delta = np.zeros((self.n_elem, self.n_node_elem, 3))
        conn = np.zeros((self.n_elem, self.n_node_elem), dtype=int)
        stiffness = np.zeros((n_stiffness, 6, 6))
        self.elem_stiffness = np.zeros((self.n_elem, ), dtype=int)
        mass = np.zeros((n_mass, 6, 6))
        elem_mass = np.zeros((self.n_elem, ), dtype=int)
        boundary_conditions = np.zeros((self.n_node, ), dtype=int)
        app_forces = np.zeros((self.n_node, 6))


        # Define aeroelastic properties
        ea = 1e7
        ga = 1e5
        gj = 1e4
        eiy = 2e4
        eiz = 4e6
        m_bar_main = 0.75
        j_bar_main = 0.075
        base_stiffness_main = self.sigma*np.diag([ea, ga, ga, gj, eiy, eiz])
        base_stiffness_fuselage = base_stiffness_main.copy()*sigma_fuselage
        base_stiffness_fuselage[4, 4] = base_stiffness_fuselage[5, 5]

        stiffness[0, ...] = base_stiffness_main
        stiffness[1, ...] = base_stiffness_fuselage

        mass[0, ...] = self.generate_mass_matrix(m_bar_main, j_bar_main)
        mass[1, ...] = self.generate_mass_matrix(m_bar_fuselage, j_bar_fuselage)


        ###############
        # right wing
        ###############
        we = 0
        wn = 0
        beam_number[we:we + self.n_elem_main] = 0
        # junction (part without ailerons)
        n_node_junctions = int(3 + 2*(self.n_elem_junction_main-1))
        self.y[wn:wn + self.n_node_main] = np.linspace(0, span_main, self.n_node_main)
        if self.sweep_quarter_chord != 0.0:
            self.x[wn+n_node_junctions:wn + self.n_node_main] += (abs(self.y[wn+n_node_junctions:wn + self.n_node_main])-y_coord_junction) * np.tan(self.sweep_quarter_chord)
    
        self.elem_stiffness[we:we + self.n_elem_main] = 0
        elem_mass[we:we + self.n_elem_main] = 0
        for ielem in range(self.n_elem_main):
            conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(self.n_node_elem - 1)) +
                                [0, 2, 1])          
            for inode in range(self.n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]  

        app_forces[wn] = [0, self.thrust, 0, 0, 0, 0]
        boundary_conditions[0] = 1

        ###############
        # left wing
        ###############
        we += self.n_elem_main
        wn += self.n_node_main

        # outer right wing

        beam_number[we:we + self.n_elem_main - 1] = 1
        # Mirror coordinates from left wing
        self.y[wn:wn + self.n_node_main - 1] = -self.y[1:self.n_node_main]
        self.x[wn:wn + self.n_node_main - 1] = self.x[1:self.n_node_main]
        self.z[wn:wn + self.n_node_main - 1] = self.z[1:self.n_node_main]


        self.elem_stiffness[we:we + self.n_elem_main] = 0
        elem_mass[we:we + self.n_elem_main] = 0


        for ielem in range(self.n_elem_main):
            conn[we + ielem, :] = ((np.ones((3, ))*(we+ielem)*(self.n_node_elem - 1)) +
                                [0, 2, 1])        
            for inode in range(self.n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0] 

        conn[we, 0] = 0
        boundary_conditions[wn-1] = -1 # tip right wing
        we += self.n_elem_main
        wn += self.n_node_main - 1
        boundary_conditions[wn-1] = -1 # tip left wing

        if not self.wing_only:
            ###############
            # fuselage
            ###############
            beam_number[we:we + self.n_elem_fuselage] = 2
            x_fuselage = np.linspace(0.0, self.length_fuselage, self.n_node_fuselage) - offset_wing_nose
            z_fuselage = np.linspace(0.0, offset_fuselage_vertical, self.n_node_fuselage)
            idx_junction = self.find_index_of_closest_entry(x_fuselage, self.x[0])
            x_fuselage = np.delete(x_fuselage, idx_junction)
            z_fuselage = np.delete(z_fuselage, idx_junction)
            self.x[wn:wn + self.n_node_fuselage-1] = x_fuselage 
            self.z[wn:wn + self.n_node_fuselage-1] = z_fuselage

            for ielem in range(self.n_elem_fuselage):
                conn[we + ielem, :] = ((np.ones((3,))*(we + ielem)*(self.n_node_elem - 1)) +
                                    2 + [0, 2, 1]) - 1

                for inode in range(self.n_node_elem):
                    frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
            for ielem in range(self.n_elem_fuselage):
                if (conn[we + ielem, :] ==  wn+idx_junction).any():
                    if (conn[we + ielem, 0] == wn+idx_junction):
                        # junction at nose
                        conn[:,:] -= 1
                        conn[we,0 ]= 0
                        break
                    elif (conn[we + ielem, 2] == wn+idx_junction):
                        # junction at center of an element
                        conn[we + ielem, 2] = 0
                        conn[we + ielem, 1] -= 1 
                        conn[we + ielem + 1:we + self.n_elem_fuselage, :] -= 1 
                    elif  (conn[we + ielem, 1] == wn+idx_junction):
                        # junction at last node of an element and first of the second one
                        conn[we + ielem, 1] = 0
                        conn[we + ielem + 1:we + self.n_elem_fuselage, :] -= 1 
                        conn[we + ielem + 1, 0] = 0
                    break
            boundary_conditions[wn] = - 1
            self.elem_stiffness[we:we + self.n_elem_fuselage] = 1
            elem_mass[we:we + self.n_elem_fuselage] = 1
            we += self.n_elem_fuselage
            wn += self.n_node_fuselage - 1
            boundary_conditions[wn - 1] = -1
            
        with h5.File(self.route + '/' + self.case_name + '.fem.h5', 'a') as h5file:
            h5file.create_dataset('coordinates', data=np.column_stack((self.x, self.y, self.z)))
            h5file.create_dataset('connectivities', data=conn)
            h5file.create_dataset('num_node_elem', data=self.n_node_elem)
            h5file.create_dataset('num_node', data=self.n_node)
            h5file.create_dataset('num_elem', data=self.n_elem)
            h5file.create_dataset('stiffness_db', data=stiffness)
            h5file.create_dataset('elem_stiffness', data=self.elem_stiffness)
            h5file.create_dataset('mass_db', data=mass)
            h5file.create_dataset('elem_mass', data=elem_mass)
            h5file.create_dataset('frame_of_reference_delta', data=frame_of_reference_delta)
            h5file.create_dataset('structural_twist', data=structural_twist)
            h5file.create_dataset('boundary_conditions', data=boundary_conditions)
            h5file.create_dataset('beam_number', data=beam_number)
            h5file.create_dataset('app_forces', data=app_forces)
            h5file.create_dataset('lumped_mass_nodes', data=lumped_mass_nodes)
            h5file.create_dataset('lumped_mass', data=lumped_mass)
            h5file.create_dataset('lumped_mass_inertia', data=lumped_mass_inertia)
            h5file.create_dataset('lumped_mass_position', data=lumped_mass_position)

    def generate_mass_matrix(self, m_bar, j_bar):
        return np.diag([m_bar, m_bar, m_bar, 
                j_bar, 0.5*j_bar, 0.5*j_bar])

    def find_index_of_closest_entry(self, array_values, target_value):
        return np.argmin(np.abs(array_values - target_value))

    def set_thrust(self, value):
        self.thrust = value
