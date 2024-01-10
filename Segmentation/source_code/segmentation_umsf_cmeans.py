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

    @staticmethod
    def remove_empty_areas(array: np.ndarray) -> np.ndarray:
        """
        Remove empty slices inside an array
        :param array: the array to clean
        :return:
        """

        nb_dims = len(array.shape)
        for d in range(nb_dims):
            indexes = []
            tmp = np.transpose(array, (2, 0, 1))
            for i in range(tmp.shape[0]):
                if not np.any(tmp[i, ...]):
                    indexes.append(i)
            array = np.delete(tmp, indexes, 0)
        return np.transpose(array, (2, 0, 1))

    def import_mri_data(self) -> None:
        """
        Import a NIfTI file from its path set in the configuration, and remove the empty boundaries.
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
        :param axis: ID of the axis to iterate on
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

            # Compute the subplot dimensions according to the ratio right below
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
                    - a 2d numpy.ndarray of the distances computed
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
        :param local_memberships: a 2d numpy.ndarray of the local memberships
        :return: a 2d numpy.ndarray of the combined memberships
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
        :param distances: a 2d numpy.ndarray of the distances between voxel and clusters
        :param local_memberships: a 2d numpy.ndarray of the local memberships
        :param weighted_data: a 2d numpy.ndarray of the weights of the local memberships
        :return: the value of the objective function
        """

        # J = sum( µ_(ij)**m * d_(ij)**2 ) + τ * sum( l_(iM_j)**m * w_(iM_j)**2) )
        return (np.sum(global_memberships ** self.configuration.fuzzifier * distances ** 2) +
                self.configuration.spatial_rate *
                np.sum(local_memberships ** self.configuration.fuzzifier * weighted_data ** 2))

    def compute_new_clusters(self, combined_memberships: np.ndarray):
        new_clusters = numpy.empty_like(self.clusters_data)
        for i in range(new_clusters):
            new_clusters[i] = sum(combined_memberships * self.mri_data)

        return new_clusters

