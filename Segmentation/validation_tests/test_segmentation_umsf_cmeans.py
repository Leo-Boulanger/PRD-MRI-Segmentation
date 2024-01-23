from source_code import segmentation_umsf_cmeans, configuration
import nibabel
import numpy as np
import os


class Test_UMSFCM:
    def __init__(self):
        # Variables used in validation.py
        self.nb_passed = 0
        self.nb_failures = 0
        self.nb_skipped = 0

        # Status of the tests. To check if the ones after have to be skipped
        self.test_local_passed = False
        self.test_global_passed = False
        self.test_combined_passed = False

        # Data shared between the tests
        self.saved_test_data = {}

        # Definition of the objects needed to execute the tests
        self.config = configuration.Configuration(nb_c=3, sp_rate=0.5, q=2.0, p=2.0,
                                                  fuzz=2.0, thresh=0.2)
        self.umsfcm = segmentation_umsf_cmeans.UMSFCM(self.config)

    def test_import_mri_data(self):
        """

        """
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
            self.config.mri_path = 'tmp_mri_validation_data.nii'
            nibabel_image = nibabel.Nifti1Image(tmp_mri_data, np.eye(4))
            nibabel.save(nibabel_image, self.config.mri_path)

            # Executing the function to test

            self.umsfcm.import_mri_data()
            # import_mri_data() will remove any empty slices, in any direction.
            # So, the first layer and the last column should be removed after the import.

            # Delete the tempory NIfTI file saved previously
            os.remove(self.config.mri_path)

            # Modifying tmp_mri_data without the empty slices
            tmp_mri_data = np.delete(tmp_mri_data, 0, axis=0)
            tmp_mri_data = np.delete(tmp_mri_data, 4, axis=2)

            # Enable this code to display both arrays and their difference:
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
            print('!> test_import_mri_data(): The data imported does not have the right values.')

        except Exception as e:
            self.nb_failures += 1
            print(e)
            print('!> test_import_mri_data(): Unhandled exception. This should not happen.')

    def test_local_membership(self):
        """

        """
        try:
            self.umsfcm.clusters_data = np.array([1, 154, 254])
            self.umsfcm.mri_data = np.ones((3, 3, 3))
            self.umsfcm.mri_data[:, :, 1] = 50
            self.umsfcm.mri_data[:, :, 2] = 255
            self.umsfcm.mri_data[:, 0, :] = 150
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
            self.test_local_passed = True
            self.saved_test_data['local_membership'] = local_memberships
            self.saved_test_data['weights'] = weights

        except AssertionError:
            self.nb_failures += 1
            print('!> test_local_membership(): The local membership values are not correctly computed.')

        except Exception as e:
            self.nb_failures += 1
            print(e)
            print('!> test_local_membership(): Unhandled exception. This should not happen.')

    def test_global_membership(self):
        """

        """
        try:
            # Initialize de variables
            self.umsfcm.clusters_data = np.array([1, 154, 254])
            self.umsfcm.mri_data = np.ones((3, 3, 3))
            self.umsfcm.mri_data[:, :, 1] = 50
            self.umsfcm.mri_data[:, :, 2] = 255
            self.umsfcm.mri_data[:, 0, :] = 150

            # Compute the global memberships
            global_memberships, distances = self.umsfcm.global_membership()

            # Validate the results
            assert np.all(global_memberships <= 1)
            assert np.all(global_memberships >= 0)
            assert np.all(distances <= 255)
            assert np.all(distances >= 10)

            # if the assertion passed:
            self.nb_passed += 1
            self.test_global_passed = True
            self.saved_test_data['global_membership'] = global_memberships
            self.saved_test_data['distances'] = distances

        except AssertionError:
            self.nb_failures += 1
            print('!> test_global_membership(): The global membership values are over their boundaries.')

        except Exception as e:
            self.nb_failures += 1
            print(e)
            print('!> test_global_membership(): Unhandled exception. This should not happen.')

    def test_combined_membership(self):
        """

        """
        if not (self.test_local_passed and self.test_global_passed):
            self.nb_skipped += 1
            print('?> test_combined_membership() skipped.')
        else:
            try:
                # Compute the combined memberships
                combined_memberships = self.umsfcm.combined_membership(self.saved_test_data['global_membership'],
                                                                       self.saved_test_data['local_membership'])

                # Reproduce the combination using only the first row of both arrays
                membership_product = np.array(self.saved_test_data['global_membership'][0]
                                              ** self.umsfcm.configuration.global_modifier
                                              * self.saved_test_data['local_membership'][0]
                                              ** self.umsfcm.configuration.local_modifier)
                sum_product = np.sum(membership_product)
                expected_result = membership_product / sum_product

                # Validate the results
                assert np.all(combined_memberships[0] == expected_result)
                assert np.all(combined_memberships >= 0)

                # if the assertion passed:
                self.nb_passed += 1
                self.test_combined_passed = True
                self.saved_test_data['combined_membership'] = combined_memberships

            except AssertionError:
                self.nb_failures += 1
                print('!> test_combined_membership(): The memberships arrays do not have the same shapes')

            except Exception as e:
                self.nb_failures += 1
                print(e)
                print('!> test_combined_membership(): Unhandled exception. This should not happen.')

    def test_objective_function(self):
        """

        """
        if not self.test_combined_passed:
            self.nb_skipped += 1
            print('?> test_objective_function() skipped.')
        else:
            try:
                # Compute the combined memberships
                objective = self.umsfcm.objective_function(self.saved_test_data['global_membership'],
                                                           self.saved_test_data['distances'],
                                                           self.saved_test_data['local_membership'],
                                                           self.saved_test_data['weights'])
                # Validate the results
                assert objective >= 0

                # if the assertion passed:
                self.nb_passed += 1

            except AssertionError:
                self.nb_failures += 1
                print('!> test_objective_function(): The result is negative.')

            except Exception as e:
                self.nb_failures += 1
                print(e)
                print('!> test_objective_function(): Unhandled exception. This should not happen.')
