# Ice Floe Tracker Calibration and Validation, Part One
This project has three aims: (1) creation of a manually labeled validation dataset, (2) implementation of appropriate metrics for segmentation evaluation, and (3) application of the dataset and the metrics for creation of a classifier to remove false positives from ice floe segmentation results.

## Software setup
The calibration and validation code includes both Python and Julia scripts. The file `cal-val.yml` contains a list of the package dependencies. Using `conda`, you can create an environment with the command 

```> conda env create -f cal-val.yml```

The package `proplot` is used for figures. There are issues with `proplot` for newer versions of `matplotlib`, hence the version is set as 0.9.7 for `proplot` and 3.4.3 for `matplotlib`, and Python is set to version 3.9.13. If a newer version of `scikit-image` is needed, we may need to choose a different tool for figure creation.

For Julia, `IceFloeTracker.jl` requires at least Julia 1.9. Instructions for installing IFT are on the IFT github repository.
TBD: Check whether we need to explicitly include `iJulia` in the `conda` yaml file in order to run Julia notebooks. What are the minimal setup steps to get a Julia notebook up and running?

The Ice Floe Tracker algorithm is encoded in the ice-floe-tracker-pipeline repository for use in the Brown University computing cluster, Oscar. Scripts here are used to prepare specification files for running the algorithm, and a portion of the output is stored locally.

## Creating the validation dataset
The goal of the calibration and validation process is to identify the best choice of parameters for the algorithm, and to quantify uncertainty and biases in the data. A crucial first step is the set of "truth values" for 




In order to do this, we need to develop a set of metrics that measure the performance of the algorithm. The set of parameters that minimizes the error in the set of metrics is considered the best choice. It is not expected that the optimal parameter set reduces all the uncertainty to zero, of course, so an additional step is to report the error metrics with the optimal paramter set. For that to be a valid measure of uncertainty, though, we need to maintain separation between testing and training data. This may require increasing the sample size -- time will tell.

TBD: Overview of IFT algorithm. Stages of the algorithm that need calibrating, and what knobs we can turn.  
TBD: Overview of the IFT output. What comes out of the algorithm, and what do we need to know to use the data?


## Sample selection
Sample selection is described and carried out in the Jupyter notebook `sample_selection.ipynb`. We sample scenes from 9 regions spanning the circumpolar Arctic, as seen in the figure below.
<!-- ![North polar stereographic map of the Arctic showing the 9 study regions. Regions are marked with color and pattern-coded boxes.](/figures/fig01_region_map.png?raw=true "Map of the sample locations") -->
<img align="right" src="/figures/fig01_region_map.png" width="300">

The notebook produces the following:
* Table 1: Study regions (text in Jupyter notebook)
* Figure 1: Map of study regions
* Table 2: Number of images of each size
The notebook `IFT case specifications.ipynb` produces CSV files to be fed into the IFT pipeline with the parameters of scenes to download. The specification files have a "location" column with entries in the format `<region_name>_<case_number>`.

## Calibration
The goal of the calibration is to identify an "optimal" set of parameters for the IFT algorithm. Optimal is in quotes because optimization requires a set of metrics, and there isn't just one set of ways that something can be optimized. Hence, we need to describe a set of metrics, which will be used both for calibration and for reporting the uncertainties in the validation section.

### Data
Results from the IFT-pipeline runs are placed in the data folder, with `data/ift_results` holding the algorithm output and the error logs being placed in `data/ift_error_logs`. The bounding boxes for each individual case are stored in `data/ift_case_definitions`. IFT-pipeline was run with the default settings first (minimum area = 300), then with the minimum area changed to 100 pixels. The file `data/ift_case_definitions/ift_runs_key.csv` contains the information on which folder in each region correspond to the min=300 and min=100 runs. A summary CSV file showing which stages of the aglorithm were completed successfully is included, so for example for the 20240124T1406Z run of baffin bay, that file is stored in `data/ift_results/baffin_bay/20240124T1406Z/baffin_bay_evaluation_table.csv`.

## Validation
Key properties that we need to account for, in no particular order
- Location of floe centroids, uncertainty in centroid
- Location of floe boundary, uncertainty in boundary
- Sensitivity to cloud cover
- Confusion matrix: false positive rate, true positive rate, false negative rate, true negative rate
- Dependence of confusion matrix on other variables: are all parts of the FSD recovered well?
- Time uncertainty
- Sensitivity to scene size, complexity
- Tracking (similar set of tests, but testing whether a floe can be tracked, rather than whether a floe can be identified)
- Is there a limit on shapes and sizes of floes for them to be trackable?

## Validation data
The validation data consists of a set of 100 km by 100 km images, randomly sampled across spring and summer months, within 9 regions of the Arctic Ocean and its marginal seas. The folder `data/validation_tables` contains two folders. In the folder `data/validation_tables/quantitative_assessment_tables/`, there are CSV files for each region as well as a file `all_100km_cases.csv` that is a simple concatenation of the other files. The CSV files include case metadata, including a case number, the file name (`long_name`), the region name, start and end dates, satellite name, and image size. The quantitative assessment results are "yes/no" data for `visible_sea_ice`, `visible_landfast_ice`, `visible_floes`, and  `artifacts` (errors in the image, missing data, or obvious overlap of different images), manual assessment of cloud fraction (0 to 1, to the nearest 0.1), and cloud category (none, thin, scattered, opaque). These values were first estimated by `qa_analyst`, then checked by `qa_reviewer`. Adjustments to the values in the first assessment are noted in `notes`. The columns `fl_analyst` and `fl_reviewer` indicate the analysts who manually labeled the images and who reviewed and/or corrected the manual labeling. 

The floe labeling task was carried out by first selecting all the images where the quantitative assessment indicated visible ice floes, then randomly dividing the images between the 5 analysts. The images in Baffin Bay were each labeled twice to provide a measure of the subjectivity in floe labeling. Floe labeling assignments


# Next steps
- Calculate fraction of land pixels using the actual land mask
- Calculate cloud fraction using the false color images, save the cloud mask
- Download additional MASIE images based on the updated sample selection
- Calculate ice fraction from MASIE image, save MASIE subset with the validation images
- Update the image download section and access the additional imagery
- Updating the data tables for Google Drive with the new samples
- Make list of new samples that need to have the quantitative assessment done on them
- Make a list of all the remaining images for floe labeling
- Set up CSV files to run the extended cases on Oscar
- Set up script to copy and rename the falsecolor and truecolor images from the Oscar runs into the validation imagery folders


# Data file structure (adjust with correct structure)
Small images are saved in the github repository currently for convenience. During project development, we are storing the full set of images and output on Google Drive and on files.brown.edu, and will eventually place the data in a repository (probably the Brown Data Repository).  

```
<data_loc>
├── validation_images
│   ├── <region_name>
│   │   ├── <dimensions>
│   │   │   ├── <case_number>_<region_name>_<startdate>
│   │   │   │   ├── landmask.tiff
│   │   │   │   ├── truecolor
│   │   │   │   │   ├── YYYYMMDD.<satellite>.truecolor.250m.tiff
│   │   │   │   ├── falsecolor
│   │   │   │   │   ├── YYYYMMDD.<satellite>.falsecolor.250m.tiff
│   │   │   │   ├── labeled
│   │   │   │   │   ├── <case_number>_<region_name>_<dimensions>_<satellite>_labeled_floes.png
│   │   │   │   │   ├── <case_number>_<region_name>_<dimensions>_<satellite>_labeled_floes.tiff
│   │   │   │   │   ├── <case_number>_<region_name>_<dimensions>_<satellite>_labeled_landfast.png
│   │   │   │   │   ├── <case_number>_<region_name>_<dimensions>_<satellite>_labeled_landfast.tiff
│   │   │   │   │   ├── <case_number>_<region_name>_<dimensions>_<satellite>.psd
│   │   │   │   ├── ift_results
│   │   │   │   │   ├── landmasks
│   │   │   │   │   ├── preprocess
│   │   │   │   │   │   ├── hdf5-files
│   │   │   │   │   │   ├── filenames.jls
│   │   │   │   │   │   ├── floe_props.jls
│   │   │   │   │   │   ├── passtimes.jls
│   │   │   │   │   │   ├── segmented_floes.jls
│   │   │   │   │   │   ├── timedeltas.jls
│   │   │   │   │   ├── soit
│   │   │   │   │   ├── tracker
```