config = {"experiment_folder": "experiment", # Path to save outputs. Default: expriment
          "scdata": "sc.h5ad", # Path to sc/snRNA-seq data, should be anndata
          "n_samples": None, # Number of samples to generate. 
                             # Default: 1000 times the number of celltypes,
          "type": "bulk", # Bulk or ST
          "celltype_col": "cell.type", # Name of the column corresponding to cell-type labels in adata.obs 
          "batch_col": "batch", # If more than one batches are present, name of the column corrsponding to batch labels in adata.obs
          "cells_per_sample": None, # Number of cells to sample to generate one sample. 
                                    # Default 500
          "downsample": None, # If simulation_type is ST, a float is used to downsample counts
          "filter": { # Filtering of sc/snRNA-seq before simulating
            "min_genes": 200,
            "min_cells": 3,
            "mt_cutoff": 5,
            "min_expr": 0, # 0.0125 # in log2(1+count)
          },
          "concentration": None, # Concentration parameter for dirichlet distribution
                        # Should be a vector of same length as the number of cell-types with non-zero values
                        # Higher concentrations will be favored. e.g. concentration [0.2,0.2,1] for 3 cell-types will make fractions 
                        # of the third cell-types higher.
                        # Default: Vector of ones.
          "prop_sparse": 0.5, # Proportion of sparse samples to generate. Default: 0.5
                          # Sparse samples are samples in which some cell-types do not exist. 
                          # Probabilities of cell-types to not be present in the generate sample are uniform.
          "generate_component_figures": True, # Computes PCA of celltype signatures per generated sample
}
