# Performance Validation Using Reference Turbines
The purpose of this project is to facilitate the optimisation of wind turbine
efficiency. To do this, we have explored and developed methods to predict the
power output of a target turbine, given the measurements of reference
turbines. Such a tool may be used to determine if a wind turbine is producing
more or less power than expected over a given time, which can be used to
justify or validate modifications made to a turbine.

## Contributing Group Members
- [Mary Eby](https://github.com/maryeby)
- [Alex Richardson](https://github.com/AlexDR1998)
- [Billy Sumners](https://github.com/wbsumners96)

### Industry Partner
- Ventient Energy
- https://www.ventientenergy.com/
- Industry Supervisor: Oliver Warlow

### Academic Supervisors
- Cathal Cummins, _Heriot-Watt University_
- Abdul-Lateef Haji-Ali, _Heriot-Watt University_
- Moritz Linkmann, _University of Edinburgh_
- Aretha Teckentrup, _University of Edinburgh_

### Methods Used
- Predictive Modelling
- Data Visualization
- Inferential Statistics
- Machine Learning

### Technologies
- Python
- Pandas
- Numpy
- Pyplot
- Sci-Kit Learn

## Project Description
Over the course of the past few decades, on-shore wind farms have been
well-established as a viable option for renewable energy production in the
United Kingdom and elsewhere. With the escalation of the climate crisis, the
demand for clean energy has only grown, and thus the need to improve the
power output of existing wind farms has grown with it. One approach is to
modify individual wind turbines to increase their efficiency and overall power
output. Since there are various ways in which a wind turbine may be modified,
it is crucial to have a metric with which to determine if a modification is an
upgrade. In order to compare the power output of a wind turbine before and
after a change has been made, it is necessary to be able to predict the
expected power output of the turbine, to be used as a reference by which to
compare the observed output. Intuition may suggest using the wind speed to
predict the power output of a given turbine, but this parameter is not a
reliable predictor of power output due to the uncertainty and variability
present in the measurement of the wind speed. Throughout this project, we
explored various models to predict the power output of existing on-shore wind
turbines independently of the wind speed. We employed a weighted average model
and some pairwise regression models, and explored metrics by which to compare
the predictive ability of these models.

## Provided Data
The wind turbine data provided for this project by Ventient Energy is neither
publicly accessible nor particularly small (specifically, the provided zip is
on the order of 250 MB, which grows to close to 1 GB when unzipped). For that
reason, the data is not in this repository. Instead, the data should be
requested from Ventient Energy directly, and any script which acts directly on
the data should take the path where the data is located as a command line
argument.

### Interpreting the data
There are two sets of `.csv` files provided - one prefixed with ARD, and the
other prefixed with CAU, where the prefixes correspond to the wind farm's
identifying code. For each set, there is a data file containing a table with a
row for each data point (where the primary key is the timestamp and turbine
ID), and columns including wind speed, power, ambient temperature, and wind
direction. There is also an additional file containing a "normal operation
flag" for each wind turbine, denoting whether the wind turbine is flagged as
operating abnormally at the specified timestamp. The details of each file and
its contents are outlined below. 

`ARD_Data.csv` and `CAU_Data.csv` 
- `ts` timestamp, formatted `DD-MMM-YYYY hh:mm:ss`
- `instanceID` the ID of the wind turbine, prefixed by `ARD` or `CAU`
   respectively
- `TI` turbulence intensity, non-dimensional
- `Wind_speed` wind speed measured in m/s
- `Power` power output of the turbine measured in kilowatts
- `Ambient_temperature` the ambient air temperature measured in degrees
   Celsius
- `Wind_direction_calibrated` calibrated wind direction relative to true north

`ARD_Flag.csv` and `CAU_Flag.csv`
- `ts` timestamp, formatted `DD-MMM-YYYY hh:mm:ss`
- `instanceID` the ID of the wind turbine, prefixed by `ARD`
- `value` a binary value indicating whether the turbine is flagged with
   abnormal behaviour

Furthermore, for each prefix, there is a spreadsheet containing turbine
positions in meters from turbine 1, the southwestern-most turbine in the farm.
It should be noted that prefix CAU has turbines in neighboring farms which do
not belong to Ventient.

`ARD Turbine Positions.xlsx` and `CAU Turbine Positions.xlsx`
- `Obstical` the ID of the wind turbine, prefixed by `ARD` or `CAU`
   respectively
- `LocType` the relative location of the remaining turbines in reference to
   turbine 01
- `Easting` the distance East from turbine 01, measured in meters
- `Northing` the distance North from turbine 01, measured in meters
- `HubHeight` the height of the turbine hub, measured in meters
- `Diameter` the diameter of the rotation path of the rotor blades, measured
   in meters
- `Altitude` the altitude at which the turbines are located, measured in
   meters

Finally, there is an additional file containing information on when
configuration changes were made to turbines which may potentially alter
their performance. The baseline configuration date refers to the date at which
it is confirmed that all turbines were configured up to some baseline state, and
the upgrade dates denote dates after which it is guaranteed that a configuration
change was implemented. We refer to the period of time between these two dates
as "intermediate time".

`Config Changes.xlsx`
- `InstanceID` the ID of the wind turbine, prefixed by `ARD` or `CAU`
- `Baseline Config up to` the date of the baseline configuration, formatted
  `DD/MM/YYYY`
- `Phase 1 upgrade from` the date of the latest upgrade, if applicable,
   formatted `DD/MM/YYYY` (if not applicable, marked as `N/A`)

## Our Source Code
The source code written for this project is contained in the `src` directory. We
structured our code in a Model-View-Controller (MVC) format, where the View and
Controller portions are combined.

`load_data.py`  
This program contains the TurbineData class, is responsible for loading the
data files, and houses various data cleaning functions.
- `__init__(path, farm)`  
   Initializes the TurbineData object using data from the specified data path
   for the specified farm.
- `to_tensor()`  
   Converts pd.dataframe to 2 rank 3 tensors.
- `to_dataframe()`  
   Converts tensor form to dataframe.
- `select_baseline(inplace=False)`  
   Selects only data before the configuration changes.
- `select_new_phase()`  
   Selects only data after the configuration changes.
- `select_time(time, verbose=False)`  
   Return the data for all wind turbines at the specified time.
- `select_turbine(turbine, inplace=False, verbose=False)`  
   Return the data for one wind turbine (or a set of turbines) across all times.
- `select_wind_direction(direction, width, verbose=False)`  
   Selects only times when the average wind direction is within the specified
   width of the specified direction
- `select_normal_operation_times(verbose=False)`  
   Removes times where any turbine is not functioning normally.
- `select_unsaturated_times(cutoff=1900, verbose=False)`  
   Removes times where any turbine is obviously saturated in the power curve.
- `select_power_min(cutoff=10, verbose=False)`  
   Removes times where any turbine has very low power.
- `nan_to_zero()`  
   Sets NaNs in the data to 0.
- `clear_wake_affected_turbines()`  
   Remove all data points of turbines lying in the wake of non-operational
   turbines.
- `merge_wake_affected_data()`  
   Combines multiple data files containing information on wake effects.
- `sample(frac=0.1, inplace=False)`  
   Selects a random subset of the data.  
The following functions are defined in `load_data.py`, but are not part of the
`TurbineData` class.
- `load_data(path: str, data_type: str, flag: bool = False)`  
   Loads wind turbine data.
- `load_positions(path, data_type, flag=False)`  
   Loads wind turbine relative position data.
- `load_data_positions(path, data_type, flag=False)`  
   Loads wind turbine data and relative position data combined.

`model/predictor.py`  
The `Predictor` class is an instance of a model, and is an object which is capable of
predicting the power output given target turbines, reference turbines, and any
other necessary parameters the model requires.
- `predict(data, targets, references, times)`  
   Predict the output power of a given set of target turbines given a separate
   set of reference turbines at a collection of times.
- `fit(data)`  
   Fit a model against the given data.
- `predict_abs_error(data, targets, references, times=None)`  
   Run the predict() function, and return its results alongside information
   about the error between prediction and target.

`model/weighted_average.py`  
This program contains definitions for both `WeightedAverage` and its subclass
`GaussianWeightedAverage`. The `WeightedAverage` class inherits the
`Predictor` class. The non-inherited functions in the `WeightedAverage` class
are listed below.
- `predict_tensor(data, tar_mask, ref_mask, verbose=False)`  
   Predict the power of the specified wind turbines.
- `predict_pd(data, targets, references, times)`  
   Predict the power of the target turbine at the specified time.

`model/kernel_ridge_regressors.py`  
This program contains definitions for the `KernelRidgeRegressor` class, and its
subclasses `LaplacianKRR`, `PowerLaplacianKRR`, `RadialBasisKRR`, and
`PeriodicLaplacianKRR`. The `KernelRidgeRegressor` class inherits the `Predictor`
class. The non-inherited functions in the `KernelRidgeRegressor` class are
listed below.
- `kernel(x_i, x_j)`  
   Compute kernel matrix given two lists of features.
- `scores(data)`  
   Returns kernel ridge scores for the specified data.

`model/__init__.py`  
This program initializes the models so that they may be easily indexed by the
app and UI.

`turbine_app.py`  
The `TurbineApp` class loads the data and runs predictions for specified models.
- `__init__(data_path, farm)`  
   Initializes a TurbineApp object with the specified data path and farm.
- `load_data()`  
   Loads the data and returns a TurbineData object.
- `clean_data()`  
   Performs pre-processing operations on the data.
- `create_predictors()`  
   Creates a predictor object for each model.
- `run_predictions(filepath, filename)`  
   Returns a list of predictions (one for each model).
- `run()`  
   Cleans the data, creates predictor objects, and runs predictions.

`turbine_ui.py`  
This program incorporates all the previously described files, and handles all
user input. This file is the only file the user needs to run in order to produce
predictions.
- `select_farm(farms)`  
   Returns the user's choice for which farm to work on.
- `select_models()`  
   Returns a list of the user's choices of models to run.
- `select_wake_effects()`  
   Returns the user's choice for whether to remove the wake affected turbines.
- `select_sample_fraction()`  
   Returns a float representing the user's choice for the fraction of data to
   consider.
- `select_turbines()`  
   Returns two int lists of turbine IDs, for targets and references.
- `select_predictor_parameters(app)`  
   Returns the parameters the user specifies for each selected model.
