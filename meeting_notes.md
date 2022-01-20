# Quick Links
- [zoom meeting](https://ed-ac-uk.zoom.us/j/87477169710)  
  passcode: wind2022

# Important Dates
- **18 February** interim presentation
- **16 March 17:00** project report submission deadline

# Tasks
## Alex
- [x] fill out the corrected [doodle poll](https://doodle.com/poll/envaniqyhsn8crah?utm_source=poll&utm_medium=link)
- [x] implement most basic model of predicting wind turbine power. Take weighted average of all other wind turbines at given timestep. Weight according to relative distance. General and efficient code, can be easily extended

## Billy
- [x] fill out the corrected [doodle poll](https://doodle.com/poll/envaniqyhsn8crah?utm_source=poll&utm_medium=link)

## Mary
- [x] create doodle poll
- [x] fix zoom meeting issues
- [x] correct time zone on doodle poll
- [x] fill out the corrected [doodle poll](https://doodle.com/poll/envaniqyhsn8crah?utm_source=poll&utm_medium=link)

## TBD
- [ ] determine and develop an approach to testing method accuracy
- [ ] tweak weighting function
- [ ] tweak nonlinearity function $f(power)$
- [ ] incorporate wind direction
- [ ] incorporate neighboring times
- [ ] learn how to determine reference turbines

# Meeting Notes
## 13 January 2022
- **project goal**: predict the power of a target wind turbine based on reference turbines
- wind speed is not always a useful predictor since there is so much variability and uncertainty involved in its measurement
- standard minimum is 4 target wind turbines and 4 reference wind turbines
- data is recorded with 10 minute resolution (the average of each 10 minute interval)

## 20 January 2022
- ARD and CAU are the wind farm codes
- Sometimes validity of the prediction may be affected if most of the operating predictors are far away
- `TI` is turbulence intensity and is non-dimensional
- wind speed is measured in meters per second
- power is measured in kilowatts
- calibrated wind direction is relative to true north
- altitude is measured in meters
- since you cannot accurately measure the resource that produces the power (the wind speed), the goal is to predict the power output following a configuration change/upgrade which you would expect to change the performance of the wind turbine
- CAU has no 11, 12, 13 turbines
- we can take the turbine angle to be the same as the wind direction
- look at different wind turbulence intensities

## Upcoming Meeting: Friday 21 January 09:00
### Questions
