# Performance Validation Using Reference Turbines

## Provided Data
The wind turbine data provided to this project by Ventient Energy is neither publicly accessible nor particularly small (specifically, the provided zip is on the order of 250 MB, which grows to close to 1 GB when unzipped). For that reason, the data is not in this repository. Instead, any script which acts directly on the data should take it as a command line argument.

### Interpreting the data
There are two sets of csv files provided - one prefixed with ARD, and the other prefixed with CAU. It is not yet clear (or I probably missed it) what these prefixes mean. For each set, there is a straight data file containing a table with a row for each data point (primary key I guess timestamp and turbine ID?), and columns including wind speed, power, ambient temperature, and wind direction. There is also an additional file containing a "normal operation flag" for each wind turbine. 

`ARD_Data.csv` and `CAU_Data.csv` 
- `ts` timestamp, formatted `DD-MMM-YYYY hh:mm:ss`
- `instanceID` the ID of the wind turbine, prefixed by `ARD` or `CAU` respectively
- `TI` _I'm actually not sure what this one means_
- `Wind_speed` wind speed measured in km/h _is this the correct unit??_
- `Power` power output of the turbine measured in _i don't know the units on this one either_
- `Ambient_temperature` the ambient air temperature measured in degrees Celsius
- `Wind_direction_calibrated` _I'm not really sure how this is calibrated? I assume this is in degrees relative to the direction the wind turbine is facing but I could be wrong_

`ARD_Flag.csv` and `CAU_Flag.csv`
- `ts` timestamp, formatted `DD-MMM-YYYY hh:mm:ss`
- `instanceID` the ID of the wind turbine, prefixed by `ARD`
- `value` a binary value indicating whether the turbine is flagged with abnormal behaviour

Furthermore, for each prefix, there is a spreadsheet containing turbine positions in meters from turbine 1. Prefix CAU has turbines in neighboring farms which do not belong to Ventient.

`ARD Turbine Positions.xlsx` and `CAU Turbine Positions.xlsx`
- `Obstical` the ID of the wind turbine, prefixed by `ARD` or `CAU` respectively
- `LocType` _i think this is the relative location of the remaining turbines in reference to turbine 01 but im not sure_
- `Easting` the distance East from turbine 01, measured in meters
- `Northing` the distance North from turbine 01, measured in meters
- `HubHeight` the height of the turbine hub, measured in meters
- `Diameter` the diameter of the rotation path of the rotor blades, measured in meters
- `Altitude` the altitude at which the turbines are located, measured in _meters above sea level??_

Finally, there is an additional file containing information on when changes were made to turbines which are expected to change their performance. (the following sentence is taken verbatim from Oliver's email but i dont know how to interpret it - *For CAU there were two phases to made, each one with two configurations (A and B).  Really configuration A1 and B2 are quite similar, so we maybe should just be looking at Turbines 20 and 22 in their phase two configuration state, rather than worrying too much about the short lived config A1 and B1 (we can maybe just exclude all of that data from the analysis).  For ARD the config change is simpler as all specified turbines are altered in the same way.*)

_note: maybe we are supposed to have another config changes file specifying updates made to CAU turbines? since the one we have now only concerns ARD ones? we can ask for clarification on his email at the next meeting_

`Config Changes.xlsx`
- `InstanceID` the ID of the wind turbine, prefixed by `ARD`
- `Baseline Config up to` the date of the baseline configuration, formatted `DD/MM/YYYY`
- `Phase 1 upgrade from` the date of the latest upgrade, if applicable, formatted `DD/MM/YYYY` (if not applicable, marked as `N/A`)

Finally, there is an additional file containing information on when changes were made to turbines which are expected to change their performance. (the following sentence is taken verbatim from oliver's email but i dont know how to interpret it - *For CAU there were two phases to made, each one with two configurations (A and B).  Really configuration A1 and B2 are quite similar, so we maybe should just be looking at Turbines 20 and 22 in their phase two configuration state, rather than worrying too much about the short lived config A1 and B1 (we can maybe just exclude all of that data from the analysis). For ARD the config change is simpler as all specified turbines are altered in the same way.*)

## Goals

## Agreed Conventions
* language: python
* libraries: pandas 
