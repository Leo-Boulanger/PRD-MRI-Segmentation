from source_code import segmentation_umsf_cmeans, configuration
import nibabel
import unittest
import numpy as np


class Test_UMSFCM:
    def __init__(self):
        self.nb_passed = 0
        self.nb_failures = 0

        self.config = configuration.Configuration(nb_c=3, sp_rate=0.5, q=2.0, p=2.0,
                                                  fuzz=2.0, thresh=0.5)
        self.umsfcm = segmentation_umsf_cmeans.UMSFCM(self.config)

    def test_import_mri_data(self):
        try:
            # Creating a temporary "MRI" array
            tmp_mri_data = np.zeros((5, 5, 5), dtype=int)
            tmp_mri_data[1:2, :2, :2] = 10
            tmp_mri_data[2:, :2, :3] = 255
            tmp_mri_data[1:, :, 3] = 120
            tmp_mri_data[2:, 3:, :3] = 20
            # Nibabel normalize the data between 0 and 255, so the saved values will change
            # if the min and max values of tmp_mri_data are not 0 and 255 respectively.
            # So, if you are modifying tmp_mri_data, make sure the values are set as such.

            # Saving the temporary image as a file
            nibabel_image = nibabel.Nifti1Image(tmp_mri_data, np.eye(4))
            nibabel.save(nibabel_image, 'tmp_mri_validation_data.nii')

            # Executing the function to test
            self.config.mri_path = 'tmp_mri_validation_data.nii'
            self.umsfcm.import_mri_data()
            # import_mri_data() will remove any empty slices, in any direction.
            # So, the first layer and the last column should be removed after the import.

            # Modifying tmp_mri_data without the empty slices
            tmp_mri_data = np.delete(tmp_mri_data, 0, axis=0)
            tmp_mri_data = np.delete(tmp_mri_data, 4, axis=2)

            # Enable this code to display both arrays:
            # print('####### tmp_mri_data:')
            # print(tmp_mri_data)
            # print('####### mri_data:')
            # print(self.umsfcm.mri_data)
            # print('####### array of differences')
            # print(tmp_mri_data-self.umsfcm.mri_data)

            # assert if both arrays are equal
            assert np.sum(tmp_mri_data - self.umsfcm.mri_data) == 0

            # if the assertion passed:
            self.nb_passed += 1

        except AssertionError:
            self.nb_failures += 1
            print('> test_import_mri_data(): The data imported does not have the right values.')

        except Exception:
            print('> test_import_mri_data(): Unhandled exception. This should not happen.')

    def test_local_membership(self):
        try:
            self.umsfcm.clusters_data = np.array([1, 154, 254])
            self.umsfcm.mri_data = np.ones((3, 3, 3))
            self.umsfcm.mri_data[:, :, 1] = 50
            self.umsfcm.mri_data[:, :, 2] = 255
            mri_shape = self.umsfcm.mri_data.shape

            local_memberships = np.zeros((self.umsfcm.mri_data.size, self.umsfcm.clusters_data.size))
            weights = np.zeros((self.umsfcm.mri_data.size, self.umsfcm.clusters_data.size))

            loop_id = 0
            for x in range(mri_shape[0]):
                for y in range(mri_shape[1]):
                    for z in range(mri_shape[2]):
                        mask = self.umsfcm.mri_data[max(x-1, 0):x+2, max(y-1, 0):y+2, max(z-1, 0):z+2].flatten()
                        local_memberships[loop_id], weights[loop_id] = self.umsfcm.local_membership(mask)
                        loop_id += 1

            # Validate the results
            for voxel_local_membership in local_memberships:
                assert np.sum(voxel_local_membership) == 1

            # if the assertion passed:
            self.nb_passed += 1

        except AssertionError:
            self.nb_failures += 1
            print('> test_local_membership(): The local membership values are not correctly computed.')

        except Exception:
            print('> test_local_membership(): Unhandled exception. This should not happen.')


