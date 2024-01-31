# PRD MRI Segmentation
 
## Installation

### Python
Download and install [Python](https://www.python.org/downloads/) version 3.12.0 or newer.
You will maybe have to restart your device.

### Virtual environment
You may want to create a virtual environment to prevent version conflicts:<br>
`python -m venv <venv directory>`

To enable a venv, execute:
`<path_to_venv>/Scripts/activate.bat`

To disable a venv, execute:
`<path_to_venv>/Scripts/deactivate.bat`

### Packages installation
pip should be installed by default. Otherwise, install pip using:<br>
`python -m ensurepip --upgrade`

Install all the required packages:<br>
`pip install -r requirements.txt `

## Usage

```
python segmentation_tool.py [-h] [-i MRI_PATH] [-o OUTPUT_DIRECTORY] [-n NB_CLUSTERS]
                            [-q LOCAL_MODIFIER] [-p GLOBAL_MODIFIER][-f FUZZIFIER]
                            [-t THRESHOLD] [-s SPATIAL_RATE]
                            [-v | --validation] [--get-cropped-image]
```

Options:

  -h, --help            show this help message and exit
  
  -i MRI_PATH, --input-file MRI_PATH
  <br>Path to the MRI file. (Should be .nii or .nii.gz)
                        
  -o OUTPUT_DIRECTORY, --output-dir OUTPUT_DIRECTORY
  <br>Path where to write the output of the program. (Must be a folder)
                        
  -n NB_CLUSTERS, --nb-clusters NB_CLUSTERS
  <br>The number of clusters to search. Integer in [3, 254]

  -q LOCAL_MODIFIER
  <br>Control the relative importance of the local membership when combining the global and local values.
  <br>Recommended value: 2

  -p GLOBAL_MODIFIER
  <br>Control the relative importance of the global membership when combining the global and local values.
  <br>Recommended value: 2

  -f FUZZIFIER, --fuzzifier FUZZIFIER
  <br>The fuzzy weighting exponent, "m" in [1, +inf): impact the performance, suppress noise and smooth the membership functions.
  <br>Recommended value: 2 if 3~4 clusters, up to 5 if 20+ clusters
                        
  -t THRESHOLD, --threshold THRESHOLD
  <br>When the difference between the newly computed clusters and the previous ones are below this THRESHOLD, the segmentation will be supposed optimal and the processus will end.
  <br>Recommended value: half the number of clusters (ex: 4 clusters => threshold set to 2)
                        
  -s SPATIAL_RATE, --spatial-rate SPATIAL_RATE
  <br>The value of the spatial rate (T): when calculating the objective function J, influence the spatial information. "J = sum(global memberships * distances) + T * sum(local memberships * weights)"
  <br>Recommended value: 0.5
                        
  -v, --validation, --no-validation
  <br>Used to check if the functions are operating as they should. Will overwrite any other argument.
                        
  --get-cropped-image, --no-get-cropped-image
  <br>Export a NIfTI file of the cropped image, which can be used in ITK_SNAP as a main image.