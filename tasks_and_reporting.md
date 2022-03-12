# Week 1: Jan 13 - Jan 19
## Tasks
- [x] schedule weekly meetings
- [x] set up project management tools (github repo, recurring zoom meeting, etc)
- [x] look through & understand the data
- [x] implement basic model which takes weighted average (according to relative
  distance) of all other wind turbines at a given timestep
## Notes
- a few questions regarding data notation and units of meaurement

# Week 2: Jan 20 - Jan 26
## Tasks
- [x] reorganize the data into a 3D structure (tensor)
- [x] create abstract representation of software design (UML diagram)
- [x] develop a `Predictor` class to provide a common structure for different
  models
## Notes
- mostly software design

# Week 3: Jan 27 - Feb 02
## Tasks
- [x] refactor code to MVC structure
- [x] build UI to easily load the data, trim it to only include portions of
  interest, and run/compare multiple models at once
- [x] determine and develop an approach to testing method accuracy
## Notes
- implementation of software design
- application of a (distance-dependent) simplistic model onto nicely behaved
  data

# Week 4: Feb 03 - Feb 09
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

# Week 5: Feb 10 - Feb 16
## Tasks
- [x] write presentation
## Notes
Presentation preparation
- 20 minutes total (we're scheduled to start at 15:15)
- 5-10 minutes of questions following the presentation
- we'll aim for about 5 minutes for each speaker
- outline:
  - Mary -- introduce & describe the problem and the data
  - Alex -- discuss the methods/approaches we have tried and the results we have achieved thus far
  - Billy -- discuss approaches we intend to try in the future & provide concluding remarks

# Week 6: Feb 17 - Feb 23
## Tasks
- [x] develop correlation model for target-reference pairs
## Notes
- worked on incorporating a feature to remove turbines affected by the wake of turbuines operating abnormally

# Week 7: Feb 24 - Mar 02
- [x] write "formalized" report outline

# Week 8: Mar 03 - Mar 09
- [x] re-train the models on a larger set of data
- [x] update UML diagram

# This week
## Tasks
### Alex
- [ ] create power vs power delta gain curve figures
- [ ] do power gain curves but with weighted average of pairwise functions
- [ ] try linear regression of outputs of pairwise functions to more rigourously determine weights
- [ ] learn how kernel ridge regression works (in context of determining which
  hyperparameters we chose

### Billy
- [ ] implement wake effects-based data cleaning
- [ ] ask oli for iec doc or some info on how wake function is derived
- [ ] learn how kernel ridge regression works (in context of determining which
  hyperparameters we chose

### Mary
- [ ] integrate existing code/features with existing code structure
- [ ] refactor code
- [ ] add file info to README
- [ ] finalize UML diagram
- [ ] add UML diagram and github link to appendix

## Notes
### Report Specs
- Each student group will collectively prepare a single written report on their topic, including the following sections:
  - title page
  - abstract
  - author contributions (paragraph-long summary that briefly explains each student’s contribution to the project and report)
  - main content (about 15-20 pages, no more than 40 pages)
  - appendices
  - references
- The document itself should be prepared in LaTeX, using 11pt text and standard spacing
- The written report will count 80% towards your final taster project mark
- Students in each project group will receive the same mark unless the author contributions suggest a serious imbalance of workload
- submit an electronic copy of your report (PDF, <10MB) to Katy Cameron, at [katy.cameron@ed.ac.uk](mailto:katy.cameron@ed.ac.uk)
- fill in and submit the attached author declaration form together with your report
- The reports will be marked by two academic members of the supervisory team, or the sole academic supervisor plus an additional academic member of staff.
### Executive Summary Specs
- Each group should produce a short (< 1 page) executive summary, which highlights the main findings of the project, aimed at a non-technical audience. This can, for example, outline
  - the methods employed
  - any particular advantages/disadvantages you noticed
  - any remaining issues that need further investigation
- This will not count towards the mark of the taster project
- Submit an electronic copy (pdf) to Katy Cameron, at [katy.cameron@ed.ac.uk](mailto:katy.cameron@ed.ac.uk)  
## Questions
