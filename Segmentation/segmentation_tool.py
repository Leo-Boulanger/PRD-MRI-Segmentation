import os.path

from validation_tests.validation import validate_tests
from source_code import configuration as cfg, logger as log
from source_code.segmentation_umsf_cmeans import UMSFCM
import argparse
import nibabel as nib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI segmentation tool using a modified spatial fuzzy c-means method.")
    parser.add_argument("-i", "--input-file", dest="mri_path", type=str,
                        help="Path to the MRI file. (Should be .nii or .nii.gz)")
    parser.add_argument("-o", "--output-dir", dest="output_directory", type=str,
                        help="Path where to write the output of the program. (Must be a folder)")
    parser.add_argument("-n", "--nb-clusters", dest="nb_clusters", type=int,
                        help="The number of clusters to search. Integer in [3, 254]")
    parser.add_argument("-q", dest="local_modifier", type=float,
                        help="Control the relative importance of the local membership when combining the "
                             "global and local values.\nRecommended value: 2")
    parser.add_argument("-p", dest="global_modifier", type=float,
                        help="Control the relative importance of the global membership when combining the "
                             "global and local values.\nRecommended value: 2")
    parser.add_argument("-f", "--fuzzifier", dest="fuzzifier", type=float,
                        help="The fuzzy weighting exponent, m in [1, +inf): impact the performance, suppress noise "
                             "and smooth the membership functions."
                             "\nRecommended value: 2 if 3~4 clusters, up to 5 if 20+ clusters")
    parser.add_argument("-t", "--threshold", dest="threshold", type=float,
                        help="When the difference between the newly computed clusters and the previous ones are below "
                             "this THRESHOLD, the segmentation will be supposed optimal and the process will end."
                             "\nRecommended value: half the number of clusters (ex: 4 clusters => threshold set to 2)")
    parser.add_argument("-s", "--spatial-rate", dest="spatial_rate", type=float,
                        help="The value of the spatial rate (T): when calculating the objective function J, influence "
                             "the spatial information."
                             "\n'J = sum(global memberships * distances) + T * sum(local memberships * weights)'")
    parser.add_argument("-v", "--validation", dest="validation", type=bool,
                        action=argparse.BooleanOptionalAction,
                        help="Used to check if the functions are operating as they should. "
                             "Will overwrite any other argument.")
    parser.add_argument("--get-cropped-image", dest="get_cropped_image", type=bool,
                        action=argparse.BooleanOptionalAction,
                        help="Export a NIfTI file of the cropped image, which can be used in ITK_SNAP as a main image.")
    # Parse the arguments
    args = parser.parse_args()

    if args.validation:  # If the program is launched for validation, run the tests validation only
        validate_tests()
    else:                # Otherwise, initialize the variables for the segmentation
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
        if args.get_cropped_image:
            image_cropped = nib.Nifti1Image(segmentation.mri_data, affine=segmentation.mri_affine,
                                            header=segmentation.mri_header)
            mri_filename = os.path.basename(config.mri_path).split('.')[0]
            output_filename = f"main_{mri_filename}.nii.gz"
            output_path = os.path.join(config.output_directory, output_filename)
            nib.save(image_cropped, output_path)

        seg_result, clusters = segmentation.start_process()

        try:
            image_segmentation = nib.Nifti1Image(seg_result, affine=segmentation.mri_affine,
                                                 header=segmentation.mri_header)
            mri_filename = os.path.basename(config.mri_path).split('.')[0]
            output_filename = f"segmentation_{mri_filename}_{config.nb_clusters}-clusters.nii.gz"
            output_path = os.path.join(config.output_directory, output_filename)
            nib.save(image_segmentation, output_path)
        except Exception as e:
            print('################ Error ')
            print(e)

        print(clusters)
