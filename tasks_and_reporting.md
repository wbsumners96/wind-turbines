# Week 1: January 13 - January 19
## Tasks
- [x] schedule weekly meetings
- [x] set up project management tools (github repo, recurring zoom meeting, etc)
- [x] look through & understand the data
- [x] implement basic model which takes weighted average (according to relative
  distance) of all other wind turbines at a given timestep
## Notes
- a few questions regarding data notation and units of meaurement

# Week 2: January 20 - January 26
## Tasks
- [x] reorganize the data into a 3D structure (tensor)
- [x] create abstract representation of software design (UML diagram)
- [x] develop a `Predictor` class to provide a common structure for different
  models
## Notes
- mostly software design

# Week 3: January 27 - February 02
## Tasks
- [x] refactor code to MVC structure
- [x] build UI to easily load the data, trim it to only include portions of
  interest, and run/compare multiple models at once
- [x] determine and develop an approach to testing method accuracy
## Notes
- implementation of software design
- application of a (distance-dependent) simplistic model onto nicely behaved
  data

# Week 4: February 03 - February 09
## Tasks
- [x] fit some windspeed data to a weibull distribution
- [x] fit some power data to a weibull distribution and a lognormal distribution
- [x] add data representation (i.e. figure generation) to the UI to better
  display results
- [x] add the data path as a command line argument to the UI
## Notes
- fitting windspeed data to weibull distribution works better if the wind
  heading is restricted to a particular range
- power data is better fit to weibull distribution than lognormal, but still not
  a great fit 
- the statistical inference task as it is given is to find the probability
  distribution of p\_target | reference measurements + target measurements not
  including power + target configuration, with the goal to either show/unshow
  that updating the configuration of a turbine "improves" the power output of a
  turbine in some way (e.g. mean goes up)
- not incorporating power of reference turbines, it's been somewhat
  interesting to look at p\_target | ws\_target, h\_target, config\_target,
  where ws = windspeed and h = wind heading. we would expect this to be sort
  of normal, with mean hovering around k\*ws^3 for some constant k
- created a widget in julia to show the ARD turbines power dependence on ws
  and heading. interestingly, the widget shows some weird stuff happening
  with turbines 3 and 10. turbine 3 has a fairly strong dependence on the
  heading (at least at ws = 10 m/s, where i tested it), being minimal around
  240 degrees, and maximal around 60 degrees (a difference of 180 degrees). 
  Looking at google maps around ardrossan, this isn't too surprising: if the
  heading of the wind is 240 (i.e. coming from the NE) then it smashes
  against the side of a mountain. what's strange is that this doesn't seem
  to be the case for any other turbine I'm looking at.
- the other weird thing is that across the turbines, the phase 1
  configuration appears to give a worse or roughly equal power output than
  before
- even in turbines which haven't had a phase 1 upgrade. this is
  particularly prominent in turbine 10. turbine 13 is an exception.
- used SciKitLearn KernelRidge regression and trained RBF kernel on a pair of
  turbines

# This week
## Tasks ### Alex
- [ ] develop correlation model for target-reference pairs

### Billy
- [x] fit some windspeed data to a weibull distribution
	- this works better if the wind heading is restricted to a particular range
- [x] fit some power data to a weibull distribution and a lognormal distribution
	- weibull better but not fantastic, as expected
- [ ] attempt to infer the correct probability distribution for the power output
      given windspeed is weibull
- [ ] incorporate wind direction
- [ ] change the `weighted_average.py` predict method to return a dataframe
- [ ] begin some basic statistical inference
	- the task as it is given is to find the probability distribution of
	  p_target | reference measurements + target measurements not including
	  power + target configuration, with the goal to either show/unshow that
	  updating the configuration of a turbine "improves" the power output of a
	  turbine in some way (e.g. mean goes up)
	- not incorporating power of reference turbines, it's been somewhat
	  interesting to look at p_target | ws_target, h_target, config_target,
	  where ws = windspeed and h = wind heading. we would expect this to be sort
	  of normal, with mean hovering around k*ws^3 for some constant k
	- created a widget in julia to show the ARD turbines power dependence on ws
	  and heading. interestingly, the widget shows some weird stuff happening
	  with turbines 3 and 10. turbine 3 has a fairly strong dependence on the
	  heading (at least at ws = 10 m/s, where i tested it), being minimal around
	  240 degrees, and maximal around 60 degrees (a difference of 180 degrees). 
	  Looking at google maps around ardrossan, this isn't too surprising: if the
	  heading of the wind is 240 (i.e. coming from the NE) then it smashes
	  against the side of a mountain. what's strange is that this doesn't seem
	  to be the case for any other turbine I'm looking at.
	- the other weird thing is that across the turbines, the phase 1
	  configuration appears to give a worse or roughly equal power output than
	  before
	- even in turbines which haven't had a phase 1 upgrade. this is
	  particularly prominent in turbine 10. turbine 13 is an exception.
	- big question: how do i do inference? f(p | s, h, c) is reasonably normal

### Mary
- [ ] update UML diagram
- [ ] refactor code
- [ ] add file info to README
- [ ] add time select feature to the UI & app

### potential future tasks
- [ ] tweak weighting function
- [ ] tweak nonlinearity function $f(power)$
- [ ] incorporate neighboring times
- [ ] incorporate turbulence intensity (TI)
- [ ] incorporate modification dates
- [ ] learn how to determine reference turbines
- [ ] from the measurements given by the turbines at a given time, what's the
      best way to get an "agreed" measurement for e.g. wind speed,
	  wind direction. mode/mean/hypermean/stock prices?
  
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
- how to do inference? f(p | s, h, c) is reasonably normal
