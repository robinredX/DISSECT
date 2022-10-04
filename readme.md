# Instructions

# Step 0. Dependencies

Install packages listed in `requirements.txt`. Ideally create a virtual environment or docker image.

# Step 1. Simulation 
If simulation is not needed, set `simulated` in `configs/main_config.py` to True, and go to step 2.

If simulation is not needed, go to `PropsSimulator` and follow the instructions for simulation in `readme.md`.

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

This runs `GradientExplainer` from python package `shap` (https://shap-lrjball.readthedocs.io/en/latest/generated/shap.GradientExplainer.html). This allows to caculate approximated shapley values to attribute contribution of each gene in computing fractions. It should be noted that these are approximations and may not be incorrect. Nonetheless, top genes per cell type based on shapley values should be enriched for specific cell types.

Run `python explain.py`

The outputs (Expected values per celltype, Raw shapely values as a numpy pickle file and plots mean shapely values per celltype) will be stored at `shap` in the `experiment_folder`.


