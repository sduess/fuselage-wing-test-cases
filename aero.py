#! /usr/bin/env python3
import h5py as h5
import numpy as np
import pandas as pd
from structure import span_main


# TODO:
# - solution for all the global parameters
# - set elastic axis
span_main = 7.07
half_wing_span = span_main*0.5
sweep_LE_main = np.deg2rad(20.)
chord_main_root = 0.471 
chord_main_tip = 0.236

# calculated inputs
x_tip = half_wing_span*np.tan(sweep_LE_main)
sweep_quarter_chord = np.arctan((x_tip+chord_main_tip/4-chord_main_root/4)/(half_wing_span))
sweep_TE_main= np.arctan((x_tip + chord_main_tip - chord_main_root)/(half_wing_span))


# Geometry parameters tail
chord_tail_root = 0.21717159844088685
chord_tail_tip = 0.180325




# Ailerons
numb_ailerons = 4
y_coord_ailerons= np.array([0.862823, 2.820273, 4.301239, 5.653424, 6.928342])/2.

# Elevators
numb_elevators = 2
y_coord_elevators = np.array([0.258501, 0.788428, 1.318355])/2.

# TODO: Adjust
chord_fin = 0.5
ea_main = 0.3
ea_fin = 0.5
ea_tail = 0.5

# reference area
area_ref = 2.54

y_coord_junction = 0.144


class Aero:
    def __init__(self, m, structure, case_name, case_route, source_directory, **kwargs):
        """
        
        Key-Word Arguments:
            - cs_deflection (float): Elevator control surface deflection
            - rudder_deflection (float): rudder deflection
            - polars (list(np.array)): 4-column array for AoA (rad), Cl, Cd, Cm of each airfoil polar
        """
        self.m = m
        self.structure = structure

        self.route = case_route
        self.case_name = case_name

        self.cs_deflection = kwargs.get('cs_deflection', 0.)
        self.rudder_deflection = kwargs.get('rudder_deflection', 0.)

        self.chord_main = chord_main
        self.sweep_main = kwargs.get('sweep', 0.)

        self.wing_only = self.structure.wing_only
        self.lifting_only = self.structure.lifting_only

        self.polars = kwargs.get('polars', None)
        self.source_directory = source_directory

    def generate(self):
        n_surfaces = 2

        # aero
        airfoil_distribution = np.zeros((self.structure.n_elem, self.structure.n_node_elem), dtype=int)
        surface_distribution = np.zeros((self.structure.n_elem,), dtype=int) - 1
        surface_m = np.zeros((n_surfaces, ), dtype=int)
        m_distribution = 'uniform'
        aero_node = np.zeros((self.structure.n_node,), dtype=bool)
        twist = np.zeros((self.structure.n_elem, self.structure.n_node_elem))
        sweep = np.zeros((self.structure.n_elem, self.structure.n_node_elem))
        chord = np.zeros((self.structure.n_elem, self.structure.n_node_elem,))
        elastic_axis = np.zeros((self.structure.n_elem, self.structure.n_node_elem,))

        junction_boundary_condition_aero = np.zeros((1, n_surfaces), dtype=int) - 1

        ###############
        # right wing
        ###############
        we = 0
        wn = 0
        # right wing (surface 0, beam 0)
        i_surf = 0
        airfoil_distribution[we:we + self.structure.n_elem_main, :] = 0
        surface_distribution[we:we + self.structure.n_elem_main] = i_surf
        surface_m[i_surf] = self.m

        if self.lifting_only:
            aero_node[wn:wn + self.structure.n_node_main] = True
        else:
            aero_node[wn:wn + self.structure.n_node_main] = abs(self.structure.y[wn:wn + self.structure.n_node_main]) >= y_coord_junction  
 
        junction_boundary_condition_aero[0, i_surf] = 1 # BC at fuselage junction
        temp_chord = np.zeros((self.structure.n_node_main)) + self.chord_main
        node_counter = 0
        for i_elem in range(we, we + self.structure.n_elem_main):
            for i_local_node in range(self.structure.n_node_elem):
                if not i_local_node == 0:
                    node_counter += 1
                inode = node_counter
                if i_local_node == 1:                
                    inode += 1
                elif i_local_node == 2:
                    inode -= 1
                chord[i_elem, i_local_node] = temp_chord[inode]
                elastic_axis[i_elem, i_local_node] = ea_main 


        we += self.structure.n_elem_main
        wn += self.structure.n_node_main
        ###############
        # left wing
        ###############
        i_surf = 1
        airfoil_distribution[we:we + self.structure.n_elem_main] = 0
        surface_distribution[we:we + self.structure.n_elem_main] = i_surf
        surface_m[i_surf] = self.m

        if self.lifting_only:
            aero_node[wn:wn + self.structure.n_node_main] = True
        else:
            aero_node[wn:wn + self.structure.n_node_main] = self.structure.y[wn:wn + self.structure.n_node_main] <= -y_coord_junction

        junction_boundary_condition_aero[0, i_surf] = 0 # BC at fuselage junction
        node_counter = 0
        for i_elem in range(we, we + self.structure.n_elem_main):
            for i_local_node in range(self.structure.n_node_elem): 
                twist[i_elem, i_local_node] = twist[i_elem - we, i_local_node] 
                chord[i_elem, i_local_node] = chord[i_elem-we, i_local_node]
                elastic_axis[i_elem, i_local_node] = elastic_axis[i_elem - we, i_local_node]
                sweep[i_elem, i_local_node] = sweep[i_elem-we, i_local_node] 

        with h5.File(self.route + '/' + self.case_name + '.aero.h5', 'a') as h5file:
            airfoils_group = h5file.create_group('airfoils')
            # add one airfoil
            NACA_airfoil = airfoils_group.create_dataset('0', data=np.column_stack(
                self.generate_naca_camber(P=0, M=0)))

            # chord
            chord_input = h5file.create_dataset('chord', data=chord)
            chord_input.attrs['units'] = 'm'

            # twist
            twist_input = h5file.create_dataset('twist', data=twist)
            twist_input.attrs['units'] = 'rad'

            # sweep
            sweep_input = h5file.create_dataset('sweep', data=sweep)
            sweep_input.attrs['units'] = 'rad'

            # airfoil distribution
            h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)
            h5file.create_dataset('surface_distribution', data=surface_distribution)
            h5file.create_dataset('surface_m', data=surface_m)
            h5file.create_dataset('m_distribution', data=m_distribution.encode('ascii', 'ignore'))
            h5file.create_dataset('aero_node', data=aero_node)
            h5file.create_dataset('elastic_axis', data=elastic_axis)
            h5file.create_dataset('junction_boundary_condition', data=junction_boundary_condition_aero)
            h5file.create_dataset('control_surface', data=control_surface)
            h5file.create_dataset('control_surface_deflection', data=control_surface_deflection)
            h5file.create_dataset('control_surface_chord', data=control_surface_chord)
            h5file.create_dataset('control_surface_hinge_coord', data=control_surface_hinge_coord)
            h5file.create_dataset('control_surface_type', data=control_surface_type)

            if self.polars is not None:
                polars_group = h5file.create_group('polars')
                for i_airfoil in range(3):  # there are three airfoils
                    polars_group.create_dataset('{:g}'.format(i_airfoil), data=self.polars[i_airfoil])

    def get_jigtwist_from_y_coord(self, y_coord):
        y_coord = abs(y_coord)
        # TODO: Find function for the interpolation (there must be one out there)
        df_jig_twist = pd.read_csv(self.source_directory + '/jig_twist.csv',
                                sep=';')
        idx_closest_value = self.find_index_of_closest_entry(df_jig_twist.iloc[:,0], y_coord)
        if self.structure.material == "reference":
            column = 1
        else:
            column = 2
        if idx_closest_value == df_jig_twist.shape[0]:
            idx_adjacent = idx_closest_value - 1 
        elif idx_closest_value == 0 or df_jig_twist.iloc[idx_closest_value,0] < y_coord:
            idx_adjacent = idx_closest_value + 1
        else:
            idx_adjacent = idx_closest_value - 1  
        
        
        jig_twist_interp = df_jig_twist.iloc[idx_closest_value,column] + ((y_coord - df_jig_twist.iloc[idx_closest_value, 0]) 
                                                    / (df_jig_twist.iloc[idx_adjacent, 0] - df_jig_twist.iloc[idx_closest_value,0])
                                                    *(df_jig_twist.iloc[idx_adjacent, column] - df_jig_twist.iloc[idx_closest_value,column]))
        # when the denominator of the interpolation is zero
        if np.isnan(jig_twist_interp):
            jig_twist_interp = df_jig_twist.iloc[idx_closest_value, 1]
        return np.deg2rad(jig_twist_interp)


    def generate_naca_camber(self,M=0, P=0):
        mm = M*1e-2
        p = P*1e-1

        def naca(x, mm, p):
            if x < 1e-6:
                return 0.0
            elif x < p:
                return mm/(p*p)*(2*p*x - x*x)
            elif x > p and x < 1+1e-6:
                return mm/((1-p)*(1-p))*(1 - 2*p + 2*p*x - x*x)

        x_vec = np.linspace(0, 1, 1000)
        y_vec = np.array([naca(x, mm, p) for x in x_vec])
        return x_vec, y_vec

    def load_airfoil_data_from_file(self):
        file = self.source_directory + "/camber_line_airfoils.csv"
        camber_line = pd.read_csv(file, sep = ";")
        return np.array(camber_line.iloc[:,0]), np.array(camber_line.iloc[:,1])

    def find_index_of_closest_entry(self, array_values, target_value):
        return np.argmin(np.abs(array_values - target_value))

    def read_spanwise_shear_center(self):
        reference_shear_center = 0.71 # given by Jurij
        df = pd.read_csv(self.source_directory + '/shear_center.csv',
                                sep=';')
        if self.structure.material == "reference":
            column = 1
        else:
            column = 2
        return (reference_shear_center + df.iloc[:,column]).to_list()