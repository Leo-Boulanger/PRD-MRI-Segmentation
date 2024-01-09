import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import math
import time

class UMSFCM:
    def __init__(self, configuration, logger=None):
        self.configuration = configuration
        #self.logger = logger
        self.mri_data = np.array

    @staticmethod
    def remove_empty_areas(array):
        nb_dims = len(array.shape)
        for d in range(nb_dims):
            indexes = []
            tmp = np.transpose(array, (2, 0, 1))
            for i in range(tmp.shape[0]):
                if not np.any(tmp[i, ...]):
                    indexes.append(i)
            array = np.delete(tmp, indexes, 0)
        return array

    def import_mri_data(self):
        """
        Import the NIfTI file
        """
        try:
            img = nib.load(self.configuration.mri_path)
        except Exception:
            print(f"Error: The file {self.configuration.mri_path} could not be read.")
            raise
        original_array = np.array(img.dataobj)
        normalized_array = (original_array/np.max(original_array)*255).astype(int)

        print(normalized_array.shape)

        tic = time.process_time()
        data_area = tuple(
            slice(np.min(idx), np.max(idx) + 1)
            for idx in np.where(normalized_array != 0))
        cleaned_array = normalized_array[data_area]
        toc = time.process_time()
        print(toc - tic)

        print(cleaned_array.shape)

        tic = time.process_time()
        # Remove empty slices
        cleaned_array = UMSFCM.remove_empty_areas(normalized_array)
        toc = time.process_time()
        print(toc-tic)



        print(cleaned_array.shape)

        fig = plt.figure(figsize=(10, 10), layout='constrained')
        subfigs = fig.subfigures(1, 1)
        ratio = (16, 10)
        y = math.sqrt(cleaned_array.shape[2] * ratio[1] / ratio[0])
        x = y * ratio[0]/ratio[1]
        subplots = subfigs.subplots(math.ceil(y), math.ceil(x))
        for i, sp in enumerate(subplots.flat):
            sp.set_axis_off()
            sp.set_aspect('equal')
            if i >= cleaned_array.shape[2]:
                continue
            sp.imshow(cleaned_array[:, :, i], cmap='gray')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()



