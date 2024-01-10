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
    parser.add_argument("-n", "--nb-clusters", dest="nb_clusters", type=str,
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
    np.random.random_sample()
    mask = segmentation.mri_data[60:63, 60:63, 60:63].flatten()
    clusters = np.random.randint(255, size=(4))
    segmentation.local_membership(mask, clusters)