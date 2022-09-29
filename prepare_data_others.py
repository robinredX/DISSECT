import os
import sys
import numpy as np
import pandas as pd

from configs.glob_config import config as glob_config
from utils.utils_fn_others import save_sc, save_test

ref = glob_config["reference"]
is_signature = glob_config["is_signature"]
test_path = glob_config["test_dataset"]
test_format = glob_config["test_format"]
savedir = os.path.join(glob_config["experiment_folder"], "results")

#if is_spatial and test_format=="txt":
#    sys.exit("Only h5ad is supported for spatial test set.")
if not os.path.exists(savedir):
    os.mkdir(savedir)

methods = ["CS_datasets", "datasets"]
for method in methods:
    save_sc(ref, savedir, method=method)
    save_test(test_path, test_format, savedir, method=method)

# Save config as json
with open('configs/glob_config.json', 'w') as fp:
    json.dump(config, fp)