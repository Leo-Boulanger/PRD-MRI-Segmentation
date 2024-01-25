# from test_configuration import Test_configuration
# from test_logger import Test_logger
from validation_tests.test_segmentation_umsf_cmeans import TestUMSFCM


def validate_tests():
    print('------------------')
    print('Validation started')

    # Initialize variables
    total_nb_passed = 0
    failures_list = []

    # Check every test, and display any error encountered
    print('Errors:', end='\r')

    # Check the tests from test_segmentation_umsf_cmeans
    test_umsfcm = TestUMSFCM()
    test_umsfcm.test_import_mri_data()
    test_umsfcm.test_local_membership()
    test_umsfcm.test_global_membership()
    test_umsfcm.test_combined_membership()
    test_umsfcm.test_objective_function()
    test_umsfcm.test_compute_new_clusters()
    test_umsfcm.test_start_process()

    # Get the results from test_segmentation_umsf_cmeans
    total_nb_passed += test_umsfcm.nb_passed
    failures_list.append(('test_segmentation_umsf_cmeans', test_umsfcm.nb_failures, test_umsfcm.nb_skipped))

    total_nb_failed = sum([f[1] for f in failures_list])
    total_nb_skipped = sum([f[2] for f in failures_list])
    if total_nb_failed == 0:
        print('No error encountered')
    print('Validation completed')

    # Print the validation results in the CLI:
    print('------------------')
    print('### Validation results:\n'
          f' {total_nb_passed}/{total_nb_failed + total_nb_skipped + total_nb_passed} tests passed')
    if total_nb_failed > 0:
        for failure in failures_list:
            print(f'-> {failure[0]}:\n'
                  f'   {failure[1]} error{"s" if failure[1] > 1 else ""}\n'
                  f'   {failure[2]} skipped')
