# Quick Links
- [zoom meeting](https://ed-ac-uk.zoom.us/j/87477169710)  
  passcode: wind2022
- [PEP8](https://www.python.org/dev/peps/pep-0008/) coding conventions
- [numpy](https://numpydoc.readthedocs.io/en/latest/format.html) style guide

# Important Dates
- **18 February** interim presentation
- **16 March 17:00** project report submission deadline

# Tasks
## Alex
- [x] determine and develop an approach to testing method accuracy
## Billy
- [x] create ui (for the purpose of only having to load the data once)
- [x] develop a predictor class to provide a common interface for different
	  models
- [ ] incorporate wind direction

## Mary
- [x] create UML diagram
- [ ] implement proposed MVC structure

incorporate modification dates
- [ ] create mathematical representation of incorporation
- [ ] design/write pseudocode for implementation
- [ ] implement the incorporation of modification dates

## TBD

- [ ] tweak weighting function
- [ ] tweak nonlinearity function $f(power)$
- [ ] incorporate neighboring times
- [ ] incorporate turbulence intensity (TI)
- [-] learn how to determine reference turbines
- [ ] add file info to README
- [ ] from the measurements given by the turbines at a given time, what's the
  best way to get an "agreed" measurement for e.g. wind speed, wind direction.
  mode/mean/hypermean/stock prices?

# Meeting Notes
## 13 January 2022
- **project goal**: predict the power of a target wind turbine based on
  reference turbines
- wind speed is not always a useful predictor since there is so much
  variability and uncertainty involved in its measurement
- standard minimum is 4 target wind turbines and 4 reference wind turbines
- data is recorded with 10 minute resolution (the average of each 10 minute
  interval)

## 20 January 2022
- ARD and CAU are the wind farm codes
- Sometimes validity of the prediction may be affected if most of the
  operating predictors are far away
- `TI` is turbulence intensity and is non-dimensional
- wind speed is measured in meters per second
- power is measured in kilowatts
- calibrated wind direction is relative to true north
- altitude is measured in meters
- since you cannot accurately measure the resource that produces the power
  (the wind speed), the goal is to predict the power output following a
  configuration change/upgrade which you would expect to change the
  performance of the wind turbine
- CAU has no 11, 12, 13 turbines
- we can take the turbine angle to be the same as the wind direction
- look at different wind turbulence intensities

## 27 January 2022
- method for testing the accuracy of the models: graph power bins vs delta,
  where delta is the difference between the modeled power output and the
  measured power output and has uncertainty
- simplify the problem by looking only at data where the wind is coming
  from the dominant direction and the group of target turbines is not
  blocked by other turbines

## Upcoming Meeting: Thursday 3 February 09:00
### Questions
- In the config changes file, it specifies there's a "Baseline config up to"
  some date, and upgrades from/to some other date. Problem is, these dates never
  match, so there's an interval of time where the turbine doesn't seem to have
  any configuration at all. Is this just a period where upgrades are being
  applied, so should we completely ignore the turbine during these periods?
    - Follow up: if so, does N/A mean the turbine is completely useless after
      the baseline config finishing day?
- Related to that, what times in the day should we take the configuration being
  available from/to? Say, if it says the baseline config is up to August 1 2019,
  should we interpret that as meaning the baseline config is up until midnight
  the following day?
