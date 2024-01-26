from validation_tests.validation import validate_tests
from source_code import configuration as cfg, logger as log
from source_code.segmentation_umsf_cmeans import UMSFCM
import argparse
import nibabel as nib
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI segmentation tool using a modified spatial fuzzy c-means method.")
    parser.add_argument("-i", "--input-file", dest="mri_path", type=str,
                        help="Path to the MRI file. (should be .nii or .nii.gz)")
    parser.add_argument("-o", "--output-dir", dest="output_directory", type=str,
                        help="Path where to write the output of the program.")
    parser.add_argument("-n", "--nb-clusters", dest="nb_clusters", type=int,
                        help="The number of clusters to search.")
    parser.add_argument("-q", dest="local_modifier", type=float,
                        help="The value of the local modifier.")
    parser.add_argument("-p", dest="global_modifier", type=float,
                        help="The value of the global modifier.")
    parser.add_argument("-f", "--fuzzifier", dest="fuzzifier", type=float,
                        help="The value of the fuzzifier.")
    parser.add_argument("-t", "--threshold", dest="threshold", type=float,
                        help="The value of the threshold.")
    parser.add_argument("-s", "--spatial-rate", dest="spatial_rate", type=float,
                        help="The value of the spatial rate.")
    parser.add_argument("-v", "--validation", dest="validation", type=bool,
                        action=argparse.BooleanOptionalAction,
                        help="Used to check if the functions are operating as they should.")

    args = parser.parse_args()
    if args.validation:
        validate_tests()

    else:
        config = cfg.Configuration(mri=args.mri_path,
                                   out_dir=args.output_directory,
                                   nb_c=args.nb_clusters,
                                   q=args.local_modifier,
                                   p=args.global_modifier,
                                   fuzz=args.fuzzifier,
                                   thresh=args.threshold,
                                   sp_rate=args.spatial_rate)
        config.setup()
        print("### Starting the program with these settings ###" + '\n'
              + str(config) + '\n')
        segmentation = UMSFCM(configuration=config)
        segmentation.import_mri_data()
        # segmentation.show_mri(axis=0, volume=False, volume_slice=0, volume_opacity=0.8,
        #                       slider=False, all_slices=False, nb_rot90=0, histogram=True)
        seg_result, clusters = segmentation.start_process()

        try:
            image_segmentation = nib.Nifti1Image(seg_result, affine=segmentation.mri_affine, header=segmentation.mri_header)
            nib.save(image_segmentation, "res_result_test.nii.gz")
        except:
            print('################ Error ')

        print(clusters)
