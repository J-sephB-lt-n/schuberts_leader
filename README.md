# schuberts_leader

**A Lightweight Framework for Automatic Discovery of Leading Indicators**

<< THIS PROJECT IS UNDER DEVELOPMENT>>

After cleaning up the code base and creating documentation, I will release it on PyPI

A **leading indicator** is a variable which is predictive of a future outcome. An example is the *number of building permit applications* being a predictor for future demand for housing (i.e. the *number of building permit applications* is a **leading indicator** for *housing demand*). 

[schuberts_leader](https://github.com/J-sephB-lt-n/schuberts_leader) executes repeated random searches through a user-provided list of potential predictive features *X* (at random time lags) in search of features with a leading (possibly non-linear) relationship with a user-provided outcome variable *y*. 

Non-linear relationships are modelled using piecewise-continuous linear [regression splines](https://en.wikipedia.org/wiki/Spline_(mathematics)). 

## Requirements

```
# core requirements:
python:     >=3.7		
numpy:      >=1.2

# additional packages required to run the tutorial examples:
pandas:     >=1.5
matplotlib: >=3.6
```

## Example Usage
TODO
