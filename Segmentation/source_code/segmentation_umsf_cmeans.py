import numpy
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import math
import time
from plotly import express as xp


class UMSFCM:
    def __init__(self, configuration, logger=None) -> None:
        self.configuration = configuration
        self.logger = logger
        self.mri_data = None
        self.clusters_data = np.empty(configuration.nb_clusters)
        self.segmentation = None

    @staticmethod
    def remove_empty_areas(array: np.ndarray) -> np.ndarray:
        """
        Remove empty slices inside an array
        :param array: the array to clean
            should be defined like : ( x_00  x_01  ... x_0nc
                                       x_10  x_11  ... x_1nc
                                       ...   ...   ... ...
                                       x_nr0 x_nr1 ... x_nrnc )
            where nc is the number of columns, and nr the number of rows
        :return: a 3d numpy.ndarray without any row of zeros in any axis
        """
        # Get the number of dimensions, and prepare the transpose target
        nb_dims = len(array.shape)
        np.roll(np.array(range(nb_dims)), 1)
        for d in range(nb_dims):
            indexes = []
            tmp = np.transpose(array, (2, 0, 1))
            for i in range(tuple(tmp.shape)[0]):
                if not np.any(tmp[i, ...]):
                    indexes.append(i)
            array = np.delete(tmp, indexes, 0)
        return array

    def import_mri_data(self) -> None:
        """
        Import a NIfTI file from its path set in the object's configuration, remove the empty boundaries,
        and set the result to the object's "mri_data" attribute.
        """

        # Try to load the NIfTI file from the path set in the configuration
        try:
            img = nib.load(self.configuration.mri_path)
        except Exception:
            print(f"Error: The file {self.configuration.mri_path} could not be read.")
            raise
        original_array = np.array(img.dataobj)

        # Normalize the data of the MRI and convert it to integers in [0, 255]
        normalized_array = (original_array / np.max(original_array) * 255).astype(int)

        # Remove empty slices
        cleaned_array = UMSFCM.remove_empty_areas(normalized_array)

        # Set the result to the object's attribute mri_data
        self.mri_data = cleaned_array

    def show_mri(self, axis: int, use_slider: bool = True) -> None:
        """
        Display the MRI
        :param axis: ID of the axis to iterate on (0: X, 1: Y, 2: Z)
        :param use_slider: True (default): display layers one by one, using a slider to switch between them,
                           False: display all layers at once
        """
        # Transpose the data to the axis passed in parameter
        final_array = np.transpose(self.mri_data, (axis % 3, (1 + axis) % 3, (2 + axis) % 3))

        # Show the MRI slices using the plotly module
        if use_slider:
            final_array = np.transpose(self.mri_data, (axis % 3, (1 + axis) % 3, (2 + axis) % 3))
            fig = xp.imshow(final_array, animation_frame=0, color_continuous_scale="gray")
            fig.show()

        else:
            # Prepare the figures and sub-figures to display the MRI
            fig = plt.figure(figsize=(10, 10), layout='constrained')
            subfigures = fig.subfigures(1, 1)

            # Compute the subplot dimensions according to the ratio defined right below
            ratio = (2, 1)  # You can change the display ratio by modifying these values
            img_ratio = final_array.shape[1] / final_array.shape[2]
            ratio = (ratio[0] * img_ratio, ratio[1] / img_ratio)
            height = math.sqrt(final_array.shape[0] * ratio[1] / ratio[0])
            width = height * ratio[0] / ratio[1]

            # Create the subplots, and display every slices on the figure
            subplots = subfigures.subplots(math.ceil(height), math.ceil(width))
            for i, sp in enumerate(subplots.flat):
                sp.set_axis_off()
                sp.set_aspect('equal')
                if i >= final_array.shape[0]:
                    continue
                sp.imshow(final_array[i, ...], cmap='gray')
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()

    def local_membership(self, mask_data: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Compute the local membership values of a voxel using its neighbours and the cluster values,
        and compute the weights of the local membership values.
        :param mask_data: flatten numpy array of the voxel analysed, and its neighbours
            should be defined like : ( x_00  x_01  ... x_0mc
                                       x_10  x_11  ... x_1mc
                                       ...   ...   ... ...
                                       x_mr0 x_mr1 ... x_mrmc )
                   where mc is the number of columns, and mr the number of rows
        :return:    - a 1d numpy.ndarray of the local membership to each cluster
                    - a 1d numpy.ndarray of the weights
        """

        # Initialize the variables
        nb_data = len(mask_data)
        nb_clusters = self.clusters_data.size
        distances = np.empty((nb_clusters, nb_data))
        mins = np.empty((nb_clusters, nb_data))
        local_memberships = np.empty(nb_clusters)
        weights = np.empty(nb_clusters)

        # Compute the intensity distance between each neighbour and each cluster
        # distances = {D_(iM_j) for all i in {0, ...nb_clusters)} | where D_(iM_j) = {abs(x_k - V_i)} for all k in M_j}
        for i, v_i in enumerate(self.clusters_data):
            for k, x_k in enumerate(mask_data):
                distances[i, k] = abs(x_k - v_i)

        # Search for each neighbour, the cluster it is closest to
        for i in range(nb_clusters):
            for k in range(nb_data):
                mins[i, k] = 1 if distances[i, k] == np.min(distances[:, k]) else 0

        # Compute, for each cluster, the ratio between the sum of its minimal distances
        # and the sum of the minimal distances of all clusters
        for i, v_i in enumerate(self.clusters_data):
            local_memberships[i] = np.sum(v_i * mins[i]) / np.sum(distances * mins)
            weights = np.sum(distances[i] * mins[i]) / nb_data

        return local_memberships, weights

    def global_membership(self) -> (np.ndarray, np.ndarray):
        """
        Compute the global membership for every voxel of the MRI
        :return:    - a 2d numpy.ndarray of the global membership values for each voxel to each cluster
                        format : ( µ_00 µ_10 ... µ_0C
                                   µ_10 µ_11 ... µ_1C
                                   ...  ...  ... ...
                                   µ_X0 µ_X1 ... µ_XC )
                        where C is the number of clusters, and X the number of voxels.
                    - a 2d numpy.ndarray of the distances computed
                        format : ( d_00 d_10 ... d_0C
                                   d_10 d_11 ... d_1C
                                   ...  ...  ... ...
                                   d_X0 d_X1 ... d_XC )
        """

        # Initialize the variables
        voxels = self.mri_data.flatten()
        global_memberships = np.empty((voxels.size, self.clusters_data.size))
        distances = np.empty((voxels.size, self.clusters_data.size))

        # Compute the global memberships
        for j, x_j in enumerate(voxels):
            # First, for each voxel, compute the distance to each cluster
            distances[j] = np.array(abs(x_j - self.clusters_data))
            sum_distances = np.sum(distances)
            for i, v_i in enumerate(self.clusters_data):
                # Then, calculate the global membership of the voxel x_j for each cluster
                global_memberships[j, i] = (1 / ((distances[j, i] / sum_distances) ** (
                                            2 / (self.configuration.fuzzifier - 1))))
        return global_memberships, distances

    def combined_membership(self, global_memberships: np.ndarray, local_memberships: np.ndarray) -> np.ndarray:
        """
        Compute a combination of the global and local membership values of each voxel
        :param global_memberships: a 2d numpy.ndarray of the global memberships
                should be defined like : ( µ_00 µ_10 ... µ_0C
                                           µ_10 µ_11 ... µ_1C
                                           ...  ...  ... ...
                                           µ_X0 µ_X1 ... µ_XC )
                where C is the number of clusters, and X the number of voxels.
        :param local_memberships: a 2d numpy.ndarray of the local memberships
                should be defined like : ( l_00 l_10 ... l_0C
                                           l_10 l_11 ... l_1C
                                           ...  ...  ... ...
                                           l_X0 l_X1 ... l_XC )
        :return: a 2d numpy.ndarray of the combined memberships
                format : ( u_00 u_10 ... u_0C
                           u_10 u_11 ... u_1C
                           ...  ...  ... ...
                           u_X0 u_X1 ... u_XC )
        """

        # Calculate a product between the global and local memberships, and apply a modifier to them
        membership_products = np.array(global_memberships ** self.configuration.global_modifier
                                       * local_memberships ** self.configuration.local_modifier)
        # Compute the sum of the products
        sum_products = np.sum(membership_products)
        # Compute the combined membership by dividing each product by the sum of all products
        return np.array(membership_products / sum_products)

    def objective_function(self, global_memberships: np.ndarray, distances: np.ndarray,
                           local_memberships: np.ndarray, weighted_data: np.ndarray) -> float:
        """
        Compute the objective function
        :param global_memberships: a 2d numpy.ndarray of the global memberships
                should be defined like : ( µ_00 µ_10 ... µ_0C
                                           µ_10 µ_11 ... µ_1C
                                           ...  ...  ... ...
                                           µ_X0 µ_X1 ... µ_XC )
                where C is the number of clusters, and X the number of voxels.
        :param distances: a 2d numpy.ndarray of the distances between voxel and clusters
                should be defined like : ( d_00 d_10 ... d_0C
                                           d_10 d_11 ... d_1C
                                           ...  ...  ... ...
                                           d_X0 d_X1 ... d_XC )
        :param local_memberships: a 2d numpy.ndarray of the local memberships
                should be defined like : ( l_00 l_10 ... l_0C
                                           l_10 l_11 ... l_1C
                                           ...  ...  ... ...
                                           l_X0 l_X1 ... l_XC )
        :param weighted_data: a 2d numpy.ndarray of the weights of the local memberships
                should be defined like : ( w_00 w_10 ... w_0C
                                           w_10 w_11 ... w_1C
                                           ...  ...  ... ...
                                           w_X0 w_X1 ... w_XC )
        :return: the result of the objective function
        """

        # J = sum( µ_(ij)**m * d_(ij)**2 ) + τ * sum( l_(iM_j)**m * w_(iM_j)**2) )
        return (np.sum(global_memberships ** self.configuration.fuzzifier * distances ** 2) +
                self.configuration.spatial_rate *
                np.sum(local_memberships ** self.configuration.fuzzifier * weighted_data ** 2))

    def compute_new_clusters(self, combined_memberships: np.ndarray):
        """
        Compute new clusters from the combined memberships
        :param combined_memberships:
                should be defined like : ( u_00 u_10 ... u_0C
                                           u_10 u_11 ... u_1C
                                           ...  ...  ... ...
                                           u_X0 u_X1 ... u_XC )
                where C is the number of clusters, and X the number of voxels.
        :return: a 1d numpy.ndarray with the new value of each cluster
        """

        # Initialize variables
        new_clusters = numpy.empty_like(self.clusters_data)

        # V_i = sum_(j=1)^(nr*nc*ns)(u_(ij) ** m * x_j) / sum_(j=1)^(nr*nc*ns)(u_(ij) ** m)
        for i in range(new_clusters.size):
            new_clusters[i] = sum(combined_memberships[:, i] * self.mri_data.flatten()) / sum(combined_memberships[:, i])
        return new_clusters

    def start_process(self):
        """

        :return:
        """
        print("Starting segmentation process...")
        max_iter_debug = 4
        current_iter = 1

        # Initialize variables
        segmentation = np.empty_like(self.mri_data)
        local_memberships = np.empty((self.mri_data.size, self.clusters_data.size))
        weights = np.empty((self.mri_data.size, self.clusters_data.size))
        global_memberships = np.empty((self.mri_data.size, self.clusters_data.size))
        distances = np.empty((self.mri_data.size, self.clusters_data.size))

        # Get clusters
        self.clusters_data = np.random.randint(255, size=self.configuration.nb_clusters)
        # Compute the local memberships
        voxels = self.mri_data.flatten()

        while True:
            print(f"Iteration {current_iter}...", end='\r')
            for j in range(voxels.size):
                mask = self.mri_data[j-1:j+1, j-1:j+1, j-1:j+1]
                local_memberships[j], weights[j] = self.local_membership(mask)

            # Compute the global memberships
            global_memberships, distances = self.global_membership()

            # Compute the combined memberships
            combined_memberships = self.combined_membership(global_memberships, local_memberships)

            # Calculate the objective function
            objective = self.objective_function(global_memberships, distances, local_memberships, weights)

            # If objective function is optimal, exit the loop
            if objective <= 1:
                print(f"Iteration {current_iter} done."
                      + "Optimal segmentation found.")
                break

            # If not optimal, compute new clusters
            new_clusters = self.compute_new_clusters(combined_memberships)

            # If the difference between old and new clusters is below the threshold, exit the loop
            if abs(np.mean(self.clusters_data) - np.mean(new_clusters)) <= self.configuration.threshold:
                print(f"Iteration {current_iter} done."
                      + "New clusters below threshold, "
                      + "found segmentation should be optimal.")
                break

            if current_iter+1 > max_iter_debug:
                print(f"Iteration {current_iter} done."
                      + "Max iteration reached..")
                break

            self.clusters_data = new_clusters
            print(f"Iteration {current_iter} done.")
            current_iter += 1

        # Compute the segmentation from the combined memberships
        self.segmentation = combined_memberships.min(axis=1).reshape(self.mri_data.shape)

        return self.segmentation, self.clusters_data

