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
- [ ] create ui (for the purpose of only having to load the data once)

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
- [ ] load data as a 3D tensor if possible

# Meeting Notes
## 13 January 2022
- **project goal**: predict the power of a target wind turbine based on reference turbines
- wind speed is not always a useful predictor since there is so much variability and uncertainty involved in its measurement
- standard minimum is 4 target wind turbines and 4 reference wind turbines
- data is recorded with 10 minute resolution (the average of each 10 minute interval)

## Upcoming Meeting: Thursday 20 January 09:00
### Questions
- What do the ARD and CAU acronyms stand for?
- Where is the wind farm?
    - Can we visit it? 😇
- Is there a disadvantage to using all non-target turbines as references?
- What is the acronym TI in the data, and what are its units? 
- Is wind speed measured in km/h? mph?
- What are the units for power?
- Is calibrated wind direction represented in degrees relative to the direction the wind turbine is facing?
- Is LocType the relative location of the remaining turbines with respect to turbine 01?
- Is altitude measured in meters above sea level?
