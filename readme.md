# Instructions

# Step 0. Dependencies

This repository requires `Python 3.8.x`.

Install packages listed in `requirements.txt` through `pip install -r requirements.txt`. Ideally create a virtual environment or docker image. Read on creating virtual environments here: https://docs.python.org/3.8/library/venv.html

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

# Step 3. Quality control

## Step 3.1 Run explainer

This runs `GradientExplainer` from python package `shap` (https://shap-lrjball.readthedocs.io/en/latest/generated/shap.GradientExplainer.html). This allows to caculate approximated SHAP values to attribute contribution of each gene in computing fractions. It should be noted that these are approximations and may not be correct. Nevertheless, top genes per cell type based on SHAP values should be enriched for specific cell types.

Run `python explain.py`

The outputs (Expected values per celltype, Raw shapely values as a numpy pickle file and plots of shap values per celltype) will be stored at `shap` in the `experiment_folder`.

# Step 4. Cell type specific gene expression per sample

Run `python dissect_expr.py`. It uses same config file as before. Please note that dependending on the number of cell types, this step can take quite some time. Use of GPU is recommended.
