#! /usr/bin/env python3
import h5py as h5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
   
n_nonlifting_bodies = 1

class Fuselage:
    def __init__(self, m, structure, case_name, case_route, **kwargs):
        """
        
        Key-Word Arguments:
        """
        self.m = m
        self.structure = structure

        self.route = case_route
        self.case_name = case_name
        self.n_nonlifting_bodies = n_nonlifting_bodies

    def generate(self): 

        nonlifting_body_node = np.zeros((self.structure.n_node,), dtype=bool)
        nonlifting_body_distribution = np.zeros((self.structure.n_elem,), dtype=int) - 1
        nonlifting_body_m = np.zeros((self.n_nonlifting_bodies, ), dtype=int)

        radius = np.zeros((self.structure.n_node,))
        we = 0
        wn = 0

        # right wing
        nonlifting_body_node[wn:wn + self.structure.n_node_main] = False
        we += self.structure.n_elem_main
        wn += self.structure.n_node_main

        # left wing
        nonlifting_body_node[wn:wn + self.structure.n_node_main] = False
        we += self.structure.n_elem_main
        wn += self.structure.n_node_main -1

        #fuselage (beam?, body ID = 0)
        i_body = 0
        
        nonlifting_body_node[0] = True
        nonlifting_body_node[wn:wn + self.structure.n_node_fuselage-1] = True
        nonlifting_body_distribution[we:we + self.structure.n_elem_fuselage] = i_body
        nonlifting_body_m[i_body] = self.m
        #radius[wn:wn + self.structure.n_node_fuselage] = get_ellipsoidal_geometry(x[wn:wn + self.structure.n_node_fuselage], thickness_ratio_ellipse,0) #np.genfromtxt('radius_wanted.csv',delimiter=',')
        # radius_fuselage = create_fuselage_geometry()
        x_coord_fuselage = np.sort(self.structure.x[nonlifting_body_node])
        x_coord_fuselage += abs(min(x_coord_fuselage))

        radius_fuselage = self.create_fuselage_geometry(x_coord_fuselage, self.structure.y_coord_junction, 0.2*self.structure.length_fuselage, 0.8*self.structure.length_fuselage) 
        idx_junction = self.find_index_of_closest_entry(x_coord_fuselage, self.structure.x[0])
        radius[0] = max(radius_fuselage)
        radius_fuselage = np.delete(radius_fuselage,idx_junction)
        
        # print("len after = ", len(radius_fuselage))
        radius[wn:wn + self.structure.n_node_fuselage] = radius_fuselage #create_fuselage_geometry()
        # np.savetxt("nonlifting_body.csv", np.transpose(np.array([x, radius, nonlifting_body_node])))
        # np.savetxt("radius_fuselage_uncorrected.csv",radius[wn:wn + n_node_fuselage], delimiter = ",")
        # #radius[wn:wn + n_node_fuselage] = adjust_curve_tangency(x[wn:wn + n_node_fuselage], radius[wn:wn + n_node_fuselage], list_cylinder_position_fuselage[0]*length_fuselage, radius_fuselage, 0.3)
        plt.plot(self.structure.x[wn:wn + self.structure.n_node_fuselage], radius[wn:wn + self.structure.n_node_fuselage], "-", color = "k")
        plt.grid()
        plt.xlabel("x [m]")
        plt.ylabel("r [m]")
        plt.ylim([0,1])
        plt.gca().set_aspect('equal')
        plt.savefig("./radius.eps")
        plt.show()
        with h5.File(self.route + '/' + self.case_name + '.nonlifting_body.h5', 'a') as h5file:
            h5file.create_dataset('shape', data='cylindrical')
            h5file.create_dataset('surface_m', data=nonlifting_body_m)
            h5file.create_dataset('nonlifting_body_node', data=nonlifting_body_node)

            h5file.create_dataset('surface_distribution', data=nonlifting_body_distribution)
            
            # radius
            radius_input = h5file.create_dataset('radius', data=radius)
            radius_input.attrs['units'] = 'm'

    def find_index_of_closest_entry(self, array_values, target_value):
        return np.argmin(np.abs(array_values - target_value))


    def create_fuselage_geometry(self, x_coord_fuselage, radius_fuselage, x_nose_end, x_tail_start):
        array_radius = np.zeros((self.structure.n_node_fuselage))
        idx_cylinder_start = self.find_index_of_closest_entry(x_coord_fuselage, x_nose_end)

        idx_cylinder_end = self.find_index_of_closest_entry(x_coord_fuselage, x_tail_start)
 
        # set constant radius of cylinder
        array_radius[idx_cylinder_start:idx_cylinder_end] = radius_fuselage
        # set r(x) for nose and tail region
        array_radius[:idx_cylinder_start] = self.add_nose_or_tail_shape(idx_cylinder_start, x_coord_fuselage, radius_fuselage, nose = True)

        array_radius[idx_cylinder_end:] = self.add_nose_or_tail_shape(idx_cylinder_end, x_coord_fuselage, radius_fuselage, nose = False)
        if array_radius[0] != 0.0:
            array_radius[1:idx_cylinder_start+1] = array_radius[:idx_cylinder_start]
            array_radius[0] = 0.0
        if array_radius[-2] == 0.0:
            array_radius[idx_cylinder_end:] =  array_radius[idx_cylinder_end-1:-1]
        return array_radius

    def add_nose_or_tail_shape(self, idx, array_x, x_transition, radius_fuselage, nose = True):
        if nose:
            x_nose = np.append(array_x[:idx],x_transition)
            shape = self.create_ellipsoid(x_nose, x_nose[-1] - x_nose[0], radius_fuselage, True)
            shape = shape[:-1]
        if not nose:
            #TO-DO: Add paraboloid shaped tail
            x_tail = np.insert(array_x[idx:],0,x_transition)
            shape = self.create_ellipsoid(x_tail, x_tail[-1]-x_tail[0], radius_fuselage, False)
            shape = shape[1:]
        return shape

    def create_ellipsoid(self, x_geom, a, b, flip):
        len_initial = len(x_geom)
        x_geom -= x_geom.max()
        if not flip:
            x_geom = np.flip(x_geom)
        np.append(x_geom,np.flip(-x_geom))
        y = b*np.sqrt(1-(x_geom/a)**2)
        if not flip:
            return y[:len_initial]
        else:
            return y[:len_initial]