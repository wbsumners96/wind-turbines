# Week 1: January 13 - January 19
## Tasks
- [x] schedule weekly meetings
- [x] set up project management tools (github repo, recurring zoom meeting, etc)
- [x] look through & understand the data
- [x] implement basic model which takes weighted average (according to relative distance) of all other wind turbines at a given timestep
## Notes
- a few questions regarding data notation and units of meaurement

# Week 2: January 20 - January 26
## Tasks
- [x] reorganize the data into a 3D structure (tensor)
- [x] create abstract representation of software design (UML diagram)
- [x] develop a `Predictor` class to provide a common structure for different models
## Notes
- mostly software design

# Week 3: January 27 - February 02
## Tasks
- [x] refactor code to MVC structure
- [x] build UI to easily load the data, trim it to only include portions of interest, and run/compare multiple models at once
- [x] determine and develop an approach to testing method accuracy
## Notes
- implementation of software design
- application of a (distance-dependent) simplistic model onto nicely behaved data

# This week
## Tasks
### Alex
- [ ] develop correlation model for target-reference pairs

### Billy
- [ ] incorporate wind direction

### Mary
- [ ] add data representation (i.e. figure generation) to the UI to better display results
- [ ] refactor code
- [ ] add time select feature to the UI & app
- [ ] add file info to README

### potential future tasks
- [ ] tweak weighting function
- [ ] tweak nonlinearity function $f(power)$
- [ ] incorporate neighboring times
- [ ] incorporate turbulence intensity (TI)
- [ ] incorporate modification dates
- [ ] learn how to determine reference turbines
- [ ] from the measurements given by the turbines at a given time, what's the
  best way to get an "agreed" measurement for e.g. wind speed, wind direction.
  mode/mean/hypermean/stock prices?
  
## Questions
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
