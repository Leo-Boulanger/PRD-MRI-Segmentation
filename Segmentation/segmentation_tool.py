import validation_tests.validation
from source_code import configuration as cfg, logger as log
from source_code.segmentation_umsf_cmeans import UMSFCM
import argparse
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

    args = parser.parse_args()
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
    segmentation = UMSFCM(config)
    segmentation.import_mri_data()
    segmentation.show_mri(axis=0, volume=True, volume_slice=46, volume_opacity=0.8,
                          slider=False, all_slices=False, nb_rot90=0)
    #segmentation.start_process()
