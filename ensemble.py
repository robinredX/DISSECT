import os
import sys
import pandas as pd
from configs.main_config import config

def ensemble(config):
    experiment_path = os.path.join(config["experiment_folder"])
    if not os.path.exists(experiment_path):
        sys.exit("Path {} does not exist. Please run prepare_data.py and dissect.py before.".format(experiment_path))

    i=0
    for i in range(len(config["seeds"])):
        df_curr = pd.read_table(os.path.join(config["experiment_folder"], "dissect_fractions_{}.txt".format(i)), index_col=0)
        if i==0:
            df_ens = df_curr
        else:
            df_ens = df_ens + df_curr
    df_ens = df_ens/len(config["seeds"])
    savepath = os.path.join(config["experiment_folder"], "dissect_fractions_ens.txt")
    print("Ensemble predictions are saved to {}".format(savepath))
    df_ens.to_csv(savepath, sep="\t")

if __name__=="__main__":
    ensemble(config)