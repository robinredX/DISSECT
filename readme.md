# Instructions

# Step 1. Simulation 
If simulation is not needed, set `simulated` in `configs/main_config.py` to True, and go to step 2.

If simulation is needed, set `simulated` to False and follow the following steps:

Change path at `reference` to h5ad file containing single-cell or purified bulk data. 

Change simulation_params:
    `reference_type` to either bulk or sc as needed. Change other parameters as needed.

Run `python simulate.py`

The simulated data will be stored with `name` in the `experiment_folder`.

# Step 2. Deconvolution

To run the steps below,
In `configs/main_config.py` - 

Change path at `reference` to the simulated h5ad file (output of step 1)

(Note: you can use a shell file `dissect.sh` which combines all following commands.) To use that file you will first need to run `chmod +x dissect.sh`

## 2.1 Prepare data
Run `python prepare_data.py`

The processed data will be stored in a new folder `datasets` in the `experiment_folder`

## 2.2 Run deconvolution
Run `python dissect.py`

The deconvolution result (tab seperated txt files) will be stored at `dissect_fractions_[model_number].txt` in the `experiment_folder`.

The trained model weights will be stored at `model_[model_number]` in the `experiment_folder`

## 2.3 Ensemble
If ensemble predictions are neeeded, 

Run `python ensemble.py`
The averaged deconvolution result (tab seperated txt files) will be stored at `dissect_fractions_ens.txt` in the `experiment_folder`.

## 2.4 Run explainer

Run `python explain.py`

The outputs (Expected values per celltype, Raw shapely values as a numpy pickle file and plots mean shapely values per celltype) will be stored at `shap` in the `experiment_folder`.


