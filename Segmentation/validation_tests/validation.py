# from test_configuration import Test_configuration
# from test_logger import Test_logger
from validation_tests.test_segmentation_umsf_cmeans import Test_UMSFCM


def validate_tests():
    print('Validation started.')

    # Initialize variables
    total_nb_passed = 0
    failures_list = []

    # Check the tests from test_segmentation_umsf_cmeans
    test_umsfcm = Test_UMSFCM()
    test_umsfcm.test_import_mri_data()
    test_umsfcm.test_local_membership()

    # Get the results from test_segmentation_umsf_cmeans
    total_nb_passed += test_umsfcm.nb_passed
    failures_list.append(('test_segmentation_umsf_cmeans', test_umsfcm.nb_failures))




    print('Validation completed.')

    # Print the validation results in the CLI:
    total_nb_failed = sum([f[1] for f in failures_list])
    print('\n### Validation results:\n'
          f'>> {total_nb_passed}/{total_nb_failed + total_nb_passed} tests passed.')
    if total_nb_failed > 0:
        for failure in failures_list:
            print(f'> {failure[0]}: {failure[1]} errors.')

