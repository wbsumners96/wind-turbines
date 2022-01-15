# Performance Validation Using Reference Turbines

## Provided Data
The wind turbine data provided to this project by Ventient Energy is neither publicly accessible nor particularly small (in particular, the provided zip is on the order of 250 MB, which grows to close to 1 GB when unzipped). For that reason, the data is not in this repository. Instead, any script which acts directly on the data should take it as a command line argument.

### Interpreting the data
There are two sets of csv files provided - one prefixed with ARD, and the other prefixed with CAU. It is not yet clear (or I probably missed it) what these prefixes mean. For each set, there is a straight data file containing a table with a row for each data point (primary key I guess timestamp and turbine ID?), and columns including wind speed, power, ambient temperature, and wind direction. There is also an additional file containing a "normal operation flag" for each wind turbine. 

Furthermore, for each prefix, there is a spreadsheet containing turbine positions in meters from turbine 1. Prefix CAU has turbines in neighboring farms which do not belong to Ventient.

Finally, there is an additional file containing information on when changes were made to turbines which are expected to change their performance. (the following sentence is taken verbatim from oliver's email but i dont know how to interpret it - *For CAU there were two phases to made, each one with two configurations (A and B).  Real    ly configuration A1 and B2 are quite similar, so we maybe should just be looking at Turbines 20 and 22 in their phase t    wo configuration state, rather than worrying too much about the short lived config A1 and B1 (we can maybe just exclude     all of that data from the analysis).  For ARD the config change is simpler as all specified turbines are altered in th    e same way.*)

## Goals

## Agreed Conventions
(mary ruok with these?)
* language: python
* libraries: pandas 
