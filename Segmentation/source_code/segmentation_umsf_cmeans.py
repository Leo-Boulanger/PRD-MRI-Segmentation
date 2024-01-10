import numpy as np
import nibabel as nib
import plotly.io
from matplotlib import pyplot as plt
import math
import time
import ipywidgets
from plotly import graph_objects as go, express as xp


class UMSFCM:
    def __init__(self, configuration, logger=None) -> None:
        self.configuration = configuration
        # self.logger = logger
        self.mri_data = self.import_mri_data()
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
        return array

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

        self.mri_data = cleaned_array

    def show_mri(self, axis: int, use_slider: bool = True) -> None:
        final_array = np.transpose(self.mri_data, (axis % 3, (1 + axis) % 3, (2 + axis) % 3))
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

    def local_membership(self, mask_data: np.ndarray, cluster_values: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Compute the local membership values of a voxel using its neighbours and the cluster values,
        and compute a weight .
        :param mask_data: flatten numpy array of the voxel analysed, and its neighbours
        :param cluster_values: 1D numpy array of the
        :return:
        """
        nb_data = len(mask_data)
        nb_clusters = len(cluster_values)
        distances = np.zeros((nb_clusters, nb_data))
        mins = np.zeros((nb_clusters, nb_data))
        local_memberships = np.zeros(nb_clusters)
        weights = np.zeros(nb_clusters)

        # Compute the intensity distance between each neighbour and each cluster
        # distances = {D_(iM_j) for all i in {0, ...nb_clusters)} | where D_(iM_j) = {abs(x_k - V_i)} for all k in M_j}
        for i in range(nb_clusters):
            for k in range(nb_data):
                distances[i, k] = abs(mask_data[k] - cluster_values[i])

        # Search for each neighbour, the cluster it is closest to
        for i in range(nb_clusters):
            for k in range(nb_data):
                mins[i, k] = 1 if distances[i, k] == min(distances[:, k]) else 0

        # Compute, for each cluster, the ratio between the sum of its minimal distances
        # and the sum of the minimal distances of all clusters
        for i in range(nb_clusters):
            local_memberships[i] = sum(distances[i] * mins[i]) / sum(distances * mins)
            weights = sum(distances[i] * mins[i]) / nb_data

        return local_memberships, weights
