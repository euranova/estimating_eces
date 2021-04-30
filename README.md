# Estimating Expected Calibration Errors

This repository allows the reproducibility of the results reported in the paper "Estimating Expected Calibration Errors". A complete production-ready python module is currently being developped with a more friendly API, and will be available soon for day-to-day use of the tools introduced within the paper.

# Setup

All this code has been tested with Python version 3.6.9. It should also work on more recent Python3 versions. Checking your version of Python can be achieved on the terminal using:

```
$ python3 -V
Python 3.6.9
```

Start by opening a terminal where you want to clone the repository, and use:

```
$ git clone https://github.com/euranova/estimating_eces.git
```

Now that it is done, let's create, activate and populate a virtual environment containing the relevant dependences:

```
$ cd estimating_eces
$ python3 -m virtualenv venv
$ source venv/bin/activate
$ (venv) pip install -r requirements.txt
```

You are now ready to run the scripts located in this repository !

# Running the code to reproduce the graphs visible in the paper

## Figures

All figures are generated within the notebook **figures.ipynb**. You can open it with jupyter notebook directly from within the virtual environment using:

```
$ (venv) jupyter notebook
```

and selecting it on the newly opened interface.

Please note that the last figure can only appear if the experimental setups have been run, which can be done following the instructions in the next section:

## Running the experimental setup

To get the results necessary to get the last figure of the paper, one needs to run the two scripts:

```
$ (venv) python script_classwise.py
$ (venv) python script_confidence.py
```
