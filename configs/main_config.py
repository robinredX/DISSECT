config = {
    "experiment_folder": "experiment",
    "test_dataset": "",
    "reference": "PropsSimulator/experiment/simulated.h5ad",
    "test_dataset_format": "txt", # Either tab-delimited txt file with genes in rows or h5ad file compatible with Scanpy.
    "test_dataset_type": "bulk", # bulk, microarray or spatial
    "duplicated": "first", # In case, there are duplicated genes in the test_dataset. To use the first occuring gene, write first. To sum the duplicated genes, write sum. To take average, write mean
    "normalize_simulated": None, #"cpm", # Only CPM and None is supported. Write CPM if not already TPM/CPM.
    "normalize_test": None, # Write CPM if not already TPM/CPM
    "var_cutoff": 0, # variance cutoff for gene filtering
    "test_in_mix": 1,
    "simulated": True, # True if dataset is already simulated. False, if it is a single-cell dataset.
    "sig_matrix": False,
    "mix": "srm",
    "save_config": True,
    "network_params": {"n_hidden_layers": 4, # Number of hidden layers
                        "hidden_units": [512,256,128,64], # Sizes of the hidden dense layers. The length of this list should be same as n_hidden_layers above.
                        "hidden_activation": "relu6", # Activation of hidden layers. Choose ones supported in keras or relu6.
                        "output_activation": "softmax", # Activation of output layer. 
                        "loss": "kldivergence", # Options - kldivergence, l2, l1. KL divergence will only work properly if output activation is softmax.
                        "n_steps": 5000, # Number of training steps
                        "lr": 1e-5, # Learning rate
                        "batch_size": 64, # best - 64 # batch size
                        "dropout": None # If you would like dropoouts in the model, write a list with same number of elements as n_hidden_layers above corresponding to each dropout layer. 
                                        # An example is [0.2,0.2,0,0.3,0.1,0.2]
    }, # Parameters to use to build network. 
    "alpha_range": [0.1,0.9], # Alpha parameter to create mixtures per batch is uniformly sampled between these two values
    "normalization_per_batch": "log1p-MinMax", # normalization of batches. Only per log1p-MinMax or None are supported
    "models": [1,2,3,4,5],
}
