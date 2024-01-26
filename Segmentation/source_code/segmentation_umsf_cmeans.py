import numpy
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import math
import time
from plotly import express as xp
from plotly import graph_objects as go
import sys


class UMSFCM:
    def __init__(self, configuration, logger=None, _debug: bool = False) -> None:
        self.configuration = configuration
        self.logger = logger
        self.clusters_data = np.empty(configuration.nb_clusters)
        self.mri_data = None
        self.segmentation = None
        self._debugging = _debug

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

        # Debug mode: use a smaller subset of the MRI data
        if self._debugging:
            print('## Debugging: downscaling array to 20%')
            new_shape = [int(d*0.25) for d in cleaned_array.shape]
            self.mri_data = cleaned_array[:new_shape[0], :new_shape[1], :new_shape[2]]
            print(f'## Debugging: final array shape: {self.mri_data.shape}')

        # Set the result to the object's attribute mri_data
        else:
            self.mri_data = cleaned_array

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
        try:
            # Initialize the variables
            nb_data = mask_data.size
            nb_clusters = self.clusters_data.size
            distances = np.empty((nb_clusters, nb_data))
            is_min_distance = np.empty((nb_clusters, nb_data))

            # Compute the intensity distance between each neighbour and each cluster
            # distances = {D_(iM_j) for all i in {0, ...nb_clusters)}
            #             where D_(iM_j) = {abs(x_k - V_i)} for all k in M_j}
            for i, v_i in enumerate(self.clusters_data):
                for k, x_k in enumerate(mask_data):
                    distances[i, k] = abs(x_k - v_i)

            # If every voxel are equal to a cluster, then both local membership and
            # weights are set to 1 at that cluster's ID:
            for i, v_i in enumerate(self.clusters_data):
                if np.all(distances[i] == 0):
                    local_membership = np.zeros(nb_clusters)
                    local_membership[i] = 1.0
                    weights = np.zeros(nb_clusters)
                    weights[i] = 1.0
                    return local_membership, weights

            # Search, for each neighbour, the cluster it is closest to
            for i in range(nb_clusters):
                for k in range(nb_data):
                    is_min_distance[i, k] = 1 if distances[i, k] == np.min(distances[:, k]) else 0

            # Compute, for each cluster, the ratio between the sum of its minimal distances
            # and the sum of the minimal distances of all clusters
            sum_minimal_distances = np.sum(distances * is_min_distance, axis=1)
            total_sum_minimal_distances = np.sum(sum_minimal_distances)
            local_memberships = sum_minimal_distances / total_sum_minimal_distances

            deltas_to_center = sum_minimal_distances / nb_data

            total_deltas_to_center = np.sum(deltas_to_center)
            weights = deltas_to_center / total_deltas_to_center

            return local_memberships, weights

        except ZeroDivisionError:
            print('Division by zero encountered.')
            raise

        except Exception:
            print('Unhandled exception encountered. This should not happen.')
            raise

    def global_membership(self) -> np.ndarray:
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

        # Compute the distances
        process_start_time = time.time()
        distances = np.empty((self.mri_data.size, self.clusters_data.size))
        for j, x_j in enumerate(self.mri_data.flatten()):
            # First, for each voxel, compute the distance to each cluster
            distances[j] = np.array(abs(x_j - self.clusters_data))
        print(f'Distances processing time = {(time.time() - process_start_time):.2f}s')

        # Compute the global memberships
        fuzzifier = 2 if self.configuration.fuzzifier == 1 else self.configuration.fuzzifier
        sum_distances = np.sum(distances)

        process_start_time = time.time()
        nb_clusters = len(self.clusters_data)
        for i, v_i in enumerate(self.clusters_data):
            tic = time.time()
            for j in range(len(voxels)):
                # Then, calculate the global membership of the voxel x_j for each cluster
                if distances[j, i] == 0:
                    global_memberships[j, i] = 1
                else:
                    global_memberships[j, i] = 1 / (distances[j, i] /
                                                    sum_distances ** (2 / fuzzifier - 1)
                                                    )
            toc = time.time()
            remaining_time = (len(voxels) - i) * (toc - tic)
            print(f"computing cluster {i}, time left: {remaining_time:0.2f}s ", end='\r')
        print(f'Global membership processing time = {(time.time() - process_start_time):.2f}s')
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
        sum_products = np.sum(membership_products, axis=1)

        # Compute the combined membership by dividing each product by the sum of all products
        return np.array(membership_products / sum_products[:, None])

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

    def compute_new_clusters(self, combined_memberships: np.ndarray) -> np.ndarray:
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
            sum_combined_memberships = np.sum(combined_memberships[:, i])
            if sum_combined_memberships == 0:
                new_clusters[i] = self.clusters_data[i]
            else:
                new_clusters[i] = (np.sum(self.mri_data.flatten() * combined_memberships[:, i]) /
                                   sum_combined_memberships)
        return new_clusters

    def histogram_peak_analysis(self, threshold: int = 25) -> np.ndarray:
        histogram = np.histogram(self.mri_data, bins=int(self.mri_data.max() - self.mri_data.min()))[0]
        clusters = np.zeros(self.configuration.nb_clusters)

        # If nb_clusters is set to "-1", search automatically a number of cluster values
        if self.configuration.nb_clusters == -1:
            max_count = np.max(histogram)

        # Else, search for a number of clusters set
        # This is just a self-made cluster-finding algorithm, probably unoptimized
        else:
            values = np.copy(histogram)
            for i in range(self.configuration.nb_clusters):
                # Get the peak
                max_count = np.max(values)
                clusters[i] = np.argwhere(values == max_count)[0]
                max_id = np.argmax(values)

                # Remove the values on the left side of the max count
                previous_values = [max_count, max_count]
                del_ids = [max_id]
                for n in range(max_id, max(max_id-threshold, 0), -1):
                    if np.all(previous_values < [values[n]]): break
                    else:
                        del_ids.append(n)
                        previous_values[0] = previous_values[1]
                        previous_values[1] = values[n]

                # Remove the values on the right side of the max count
                previous_values = [max_count, max_count]
                for n in range(max_id, min(max_id+threshold, len(values))):
                    if np.all(previous_values < [values[n]]): break
                    else:
                        del_ids.append(n)
                        previous_values[0] = previous_values[1]
                        previous_values[1] = values[n]
                values = np.delete(values, del_ids)
        return clusters

    def start_process(self) -> (np.ndarray, np.ndarray):
        """

        :return:
        """
        print("Verifying data...", end='\r')
        # Assert MRI file content
        try:
            assert np.all(self.mri_data >= 0)
            assert np.all(self.mri_data <= 255)
        except AssertionError:
            print("!> Some values in the NIfTI file exceed the boundaries [0, 255].")
            # print(np.where(self.mri_data < 0))
            print(self.mri_data[np.where(self.mri_data < 0)])
            raise
        print("Verifying data. Done: Data should be compatible.")

        print("Starting segmentation process...")
        # Timer to compute the total time of the process
        segmentation_start_time = time.time()

        max_iter_debug = 20
        current_iter = 1

        # Get clusters
        # self.clusters_data = np.sort(np.random.default_rng().choice(255, size=self.configuration.nb_clusters, replace=False))
        self.clusters_data = self.histogram_peak_analysis()
        print(f'Starting clusters: {self.clusters_data}')

        # Initialize variables
        local_memberships = np.empty((self.mri_data.size, self.clusters_data.size))
        weights = np.empty((self.mri_data.size, self.clusters_data.size))
        mri_shape = self.mri_data.shape
        print(f"MRI shape: {mri_shape}")

        while True:
            print(f"\n## Iteration {current_iter} ##")

            # Compute the local memberships
            loop_id = 0
            process_start_time = time.time()
            for x in range(mri_shape[0]):
                tic = time.time()
                print(f"computing [{x},:,:], time left: ", end='\r')
                for y in range(mri_shape[1]):
                    for z in range(mri_shape[2]):
                        mask = self.mri_data[x, y, z].flatten()
                        local_memberships[loop_id], weights[loop_id] = self.local_membership(mask)
                        loop_id += 1
                toc = time.time()
                remaining_time = (mri_shape[0] - (x + 1)) * (toc - tic)
                print(f"computing [{x}, :, :], time left: {remaining_time:0.2f}s ", end='\r')
            print(f'Local membership processing time = {(time.time() - process_start_time):.2f}s')

            # Compute the global memberships
            global_memberships, distances = self.global_membership()

            # Compute the combined memberships
            process_start_time = time.time()
            combined_memberships = self.combined_membership(global_memberships, local_memberships)
            print(f'Combined membership processing time = {(time.time() - process_start_time):.2f}s')

            # Calculate the objective function
            process_start_time = time.time()
            objective = self.objective_function(global_memberships, distances, local_memberships, weights)
            print(f'Objective function processing time = {(time.time() - process_start_time):.2f}s')

            # If objective function is optimal, exit the loop
            if objective <= 1:
                print(f"Iteration {current_iter} done.\n"
                      + "Optimal segmentation found.")
                break

            # If not optimal, compute new clusters
            new_clusters = self.compute_new_clusters(combined_memberships)

            # If the threshold is set as float, convert it to an integer in [0, 255]
            threshold = self.configuration.threshold if self.configuration.threshold > 0 else 1

            # If the difference between old and new clusters is below the threshold, exit the loop
            if np.sum(np.abs(self.clusters_data - new_clusters)) <= threshold:
                print(f"Iteration {current_iter} done.\n"
                      + "New clusters below threshold,\n"
                      + "The segmentation found should be optimal.")
                break

            if current_iter + 1 > max_iter_debug:
                print(f"Iteration {current_iter} done."
                      + "Max iteration reached..")
                break

            self.clusters_data = new_clusters
            print(f'New clusters found: {new_clusters}')
            print(f"Iteration {current_iter} completed.")
            current_iter += 1

        # Compute the segmentation from the combined memberships
        self.segmentation = combined_memberships.argmax(axis=1).reshape(self.mri_data.shape)

        # Print the total process time
        print(f'Total segmentation processing time = {(time.time() - segmentation_start_time):.2f}s')

        return self.segmentation, self.clusters_data

    def show_mri(self, axis: int = 0, nb_rot90: int = 0,
                 volume: bool = False, volume_slice: int = 0, volume_opacity: float = 1.0,
                 slider: bool = False, all_slices: bool = False, histogram: bool = False) -> None:
        """
        Display the MRI
        :param axis: ID of the axis to iterate on (0: X, 1: Y, 2: Z)
        :param nb_rot90: number of times the MRI is rotated by 90 degrees
        :param volume: if true, display the MRI in a 3D space
        :param volume_slice: will display the 3D MRI until the slice at this number
        :param volume_opacity: the amount of opacity of the 3D representation
        :param slider: if true, display layers one by one, using a slider to switch between them,
        :param all_slices: if true, display all slices next
        :param histogram: if true, display a histogram of the MRI
        """
        if histogram:
            hist_data = self.mri_data.flatten()
            fig = xp.histogram(hist_data[hist_data > 0])
            fig.show()

        # Transpose the data to the axis passed in parameter
        final_array = np.transpose(self.mri_data, (axis % 3, (1 + axis) % 3, (2 + axis) % 3))
        final_array = np.rot90(final_array, nb_rot90, (1, 2))

        # Show the MRI in a 3D space using the plotly module
        if volume:
            if volume_slice > 0:
                final_array = final_array[:volume_slice, :, :]
            coordinates = np.array([(i, j, k) for i in range(final_array.shape[0])
                                    for j in range(final_array.shape[1])
                                    for k in range(final_array.shape[2])])
            mri_x = coordinates[..., 0]
            mri_y = coordinates[..., 1]
            mri_z = coordinates[..., 2]
            fig = go.Figure(
                go.Volume(
                    x=mri_x,
                    y=mri_y,
                    z=mri_z,
                    value=final_array.flatten(),
                    isomin=1,
                    opacity=volume_opacity,
                    colorscale=[[i / 255, f'rgb({i},{i},{i})'] for i in range(256)]
                )
            )
            fig.show()

        # Show the MRI slices with a slider using the plotly module
        if slider:
            fig = xp.imshow(final_array, animation_frame=0, color_continuous_scale="gray")
            fig.show()

        if all_slices:
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

    def __str__(self):
        return (f"Configuration : \n{self.configuration}\n"
                f"Logger : \n{self.logger}\n"
                f"MRI data size : \n{self.mri_data.size}\n"
                f"Clusters : \n{self.clusters_data}\n"
                f"Segmentation: \n{self.segmentation}\n")
