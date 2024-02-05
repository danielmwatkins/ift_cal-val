# Ice Floe Tracker Calibration and Validation
Code and data supporting the calibration and validation of the Ice Floe Tracker algorithm. 

The goal of the calibration and validation process is to identify the best choice of parameters for the algorithm, and to quantify uncertainty and biases in the data. In order to do this, we need to develop a set of metrics that measure the performance of the algorithm. The set of parameters that minimizes the error in the set of metrics is considered the best choice. It is not expected that the optimal parameter set reduces all the uncertainty to zero, of course, so an additional step is to report the error metrics with the optimal paramter set. For that to be a valid measure of uncertainty, though, we need to maintain separation between testing and training data. This may require increasing the sample size -- time will tell.

TBD: Overview of IFT algorithm. Stages of the algorithm that need calibrating, and what knobs we can turn.  
TBD: Overview of the IFT output. What comes out of the algorithm, and what do we need to know to use the data?

## Software setup
The calibration and validation code includes both Python and Julia scripts. The file `cal-val.yml` contains a list of the package dependencies. Using `micromamba`, you can create an environment with the command 

```> micromamba create -f cal-val.yml```

The package `proplot` is used for figures. There are issues with `proplot` for newer versions of `matplotlib`, hence the version is set as 0.9.7 for `proplot` and 3.4.3 for `matplotlib`. If a newer version of `scikit-image` is needed, we may need to choose a different tool for figure creation.

For Julia, `IceFloeTracker.jl` requires at least Julia 1.9. Instructions for installing IFT are on the IFT github repository.
TBD: Check whether we need to explicitly include `iJulia` in the `micromamba` yaml file in order to run Julia notebooks.

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
- Manually identified floes
- Manually verified tracking? How should that work? Look into automated methods 



