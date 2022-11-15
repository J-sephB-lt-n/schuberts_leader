# schuberts_leader
&lt;&lt;under construction>> A Lightweight Framework for Discovering Leading Indicator Variables

**schuberts_leader** is a package with the following aims:

* Depends on no other packages aside from [numpy](https://github.com/numpy/numpy)

* Robustly performs a single task: detection of individual variables in tabular time-series data which have a leading non-linear correlation with a chosen univariate outcome variable

* Does not try to do too much 

	o Does not have include optional extraneous features surrounding the core functionality
	
	o Is not flexible with regards to user input types (accepts data only as numpy arrays of integers or floats)
	
	o Does not make design decisions for the user (e.g. does not attempt to detect or avoid spurious correlations by removing trend and/or seasonality)

	

