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
        self.mri_data = np.array

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
            # Prepare the figures and subfigures to display the MRI
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
