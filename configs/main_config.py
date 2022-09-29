config = {
    "experiment_folder": "GSE120502_pbmc", 
    "test_dataset": "GSE120502.txt",
    "reference": "sim_pbmc.h5ad",
    "test_dataset_format": "txt", # Either tab-delimited txt file with genes in rows or h5ad file compatible with Scanpy.
    "test_dataset_type": "bulk", # bulk, microarray or spatial
    "duplicated": "first", # In case, there are duplicated genes in the test_dataset. To use the first occuring gene, write first. To sum the duplicated genes, write sum. To take average, write mean
    "normalize_simulated": "cpm", #"cpm", # Only CPM and None is supported. Write CPM if not already TPM/CPM.
    "normalize_test": "cpm", # Write CPM if not already TPM/CPM
    "var_cutoff": 0.1, # variance cutoff for gene filtering
    "test_in_mix": None, # Number of test samples to use in the generation of online mixtures.
    "simulated": True, # True if dataset is already simulated. False, if it is a single-cell dataset.
    "simulation_params": {"n_samples": 5000, # Number of simulations to do
                         "sparse": 0.5, # Proportion of simulations of sparse samples (purified cell populations)
                         "reference_type": "sc", # sc or bulk
                         "unknown": None,
                         "normalize": "cpm",
                         "name": "liver_tumor", # Simulated data will be saved as pbmc8k.
                         "downsample": None,
                         "cells_range": [8,15], # If variable cells per sample. Used for ST simulation.
                         "celltypes_range": [1,6]
                         },
    "sig_matrix": False,
    "mix": "srm",
    "save_config": True,
    "network_params": {"n_hidden_layers": 4, # Number of hidden layers
                        "hidden_units": [512,256,128,64], # Sizes of the hidden dense layers. The length of this list should be same as n_hidden_layers above.
                        "hidden_activation": "relu6", # Activation of hidden layers. Choose ones supported in keras or relu6.
                        "output_activation": "softmax", # Activation of output layer. 
                        "loss": "kldivergence", # Options - kldivergence, l2, l1. KL divergence will only work properly if output activation is softmax.
                        "n_steps": 6000, # Number of training steps
                        "lr": 1e-5, # Learning rate
                        "batch_size": 64, # best - 64 # batch size
                        "dropout": None # If you would like dropoouts in the model, write a list with same number of elements as n_hidden_layers above corresponding to each dropout layer. 
                                        # An example is [0.2,0.2,0,0.3,0.1,0.2]
    }, # Parameters to use to build network. 
    "alpha_range": [0.1,0.9], # Alpha parameter to create mixtures per batch is uniformly sampled between these two values
    "normalization_per_batch": "log1p-MinMax", # normalization of batches. Only per log1p-MinMax or None are supported
    "seeds": [1,2,3,4,5],
}
