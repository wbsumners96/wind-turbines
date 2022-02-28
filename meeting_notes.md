# Quick Links
- [zoom meeting](https://ed-ac-uk.zoom.us/j/87477169710)  passcode: wind2022
- [PEP8](https://www.python.org/dev/peps/pep-0008/) coding conventions
- [numpy](https://numpydoc.readthedocs.io/en/latest/format.html) style guide
- [reports & presentation](https://www.overleaf.com/2638667994ssjctncpsmvh)

# Important Dates
- **18 February** interim presentation
- **16 March 17:00** project report submission & executive summary deadline

# Meeting Notes
## 13 January 2022
- **project goal**: predict the power of a target wind turbine based on
  reference turbines
- wind speed is not always a useful predictor since there is so much variability
  and uncertainty involved in its measurement
- standard minimum is 4 target wind turbines and 4 reference wind turbines
- data is recorded with 10 minute resolution (the average of each 10 minute
  interval)

## 20 January 2022
- ARD and CAU are the wind farm codes
- Sometimes validity of the prediction may be affected if most of the operating
  predictors are far away
- `TI` is turbulence intensity and is non-dimensional
- wind speed is measured in meters per second
- power is measured in kilowatts
- calibrated wind direction is relative to true north
- altitude is measured in meters
- since you cannot accurately measure the resource that produces the power (the
  wind speed), the goal is to predict the power output following a configuration
  change/upgrade which you would expect to change the performance of the wind
  turbine
- CAU has no 11, 12, 13 turbines
- we can take the turbine angle to be the same as the wind direction
- look at different wind turbulence intensities

## 27 January 2022
- method for testing the accuracy of the models: graph power bins vs delta,
  where delta is the difference between the modeled power output and the
  measured power output and has uncertainty
- simplify the problem by looking only at data where the wind is coming from the
  dominant direction and the group of target turbines is not blocked by other
  turbines
 
## 03 February 2022
- main interest is in the ability to predict the power over long periods of time
  (i.e. many months), rather than the ability to predict the power for a
  specific instance of time
- things to add to future plots & figures:
  - plot power vs error
  - express error as a percentage of the measured power
- develop models for each wind direction, where wind direction is split into ten
  degree "bands"
- create correlation models for each target-reference pair of turbines
  - potentially create a correlation model that doesn't depend on distance, and
    include the distance weighting factor (and other factors such as wind
    direction) later
  - correlation should be some function representing the relationship between
    the power output of the two turbines
- introduce a weighting for each target-reference pair based on a quality metric
  rather than based purely on distance
- consider using regression to incorporate multiple parameters such as distance,
  turbine angles, etc
- consider including a confidence measure of how accurate the prediction is
- currently we are assuming that the power output of the target is a linear
  combination of the power output of the reference turbines (this is the least
  sophisticated method)
- try maximizing the likelihood of observing the data we predict
- initially, for 1 target and _n_ references:
  - find the distribution of the target turbine
  - plot the density of the power output
  - fit the distribution
  - find the mean and variance, assuming they are a linear model/weighted
    average of the power outputs of the reference turbines
- begin weekly reporting

## 10 February 2022
- try tests with many inputs in addition to pairwise correlations
- look at performance results for many inputs (i.e. the general trend of all
  turbines) from before, during, and after upgrades
- project presentation should include
  - description of the problem
  - information about the data
  - how we are processing, cleaning, & using the data
  - methods/approaches we have tried thus far and their results
  - approaches we intend to try in the future

## 17 February 2022
preparing for the interim presentation
- include a clear explanation of the motivation for the project
- consider clearer ways to express and visualize the data

## 24 February 2022
starting to wrap up and prepare the report
- the main idea of the report is a discussion of good and bad ways to predict power output
- discuss each model and how well the model predicts power
- future work: applying these prediction methods on turbines where configuration changes have occurred
- for applications which consider configuration changes, use turbines without configuration change as references and targets with configuration changes as targets
- plot the power vs power delta over all time (gain curve)
  - represents error in predictions
  - can be used as a metric for how good or bad a model is at predicting the power
  - line should be as close to zero as possible (negative is an underprediction, positive is an overprediction)
  - can be distilled down to single number using a probability distribution
- formalize and send advisors an outline of the report

## Upcoming Meeting: Thursday 03 March 09:00
