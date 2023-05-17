st_data_files_synth = [
    # slideseq
    "spatial/simulations_kidney_slideSeq_v2/UMOD-WT.WT-2a_resolution75.h5ad",
    "spatial/simulations_kidney_slideSeq_v2/UMOD-KI.KI-4b_resolution105.h5ad",
    # seqfish
    "spatial/simulations_heart_seqFISH/embryo1_resolution0.11.h5ad",
    "spatial/simulations_heart_seqFISH/embryo2_resolution0.11.h5ad",
    "spatial/simulations_heart_seqFISH/embryo3_resolution0.11.h5ad",
    # merfish
    "spatial/simulations_hypothalamus_MERFISH/0_resolution0.04.h5ad",
    "spatial/simulations_hypothalamus_MERFISH/1_resolution0.04.h5ad",
]

st_data_files_real = [
    # visium
    "spatial/V1_Mouse_Brain_Sagittal_Anterior.h5ad",
    "spatial/lymph_node/st_lymph.h5ad",
]

st_data_files_switched = [
    # slide-seq
    "spatial/simulations_kidney_slideSeq_v2/UMOD-WT.WT-2a_resolution75.h5ad",
    "spatial/simulations_kidney_slideSeq_v2/UMOD-KI.KI-4b_resolution105.h5ad",
    # seqfish
    "spatial/simulations_heart_seqFISH/embryo1_resolution0.11.h5ad",
    "spatial/simulations_heart_seqFISH/embryo2_resolution0.11.h5ad",
    # merfish
    "spatial/simulations_hypothalamus_MERFISH/0_resolution0.04.h5ad",
    "spatial/simulations_hypothalamus_MERFISH/1_resolution0.04.h5ad",
    # "spatial/simulations_kidney_slideSeq_v2/UMOD-WT.WT-2a_resolution75.h5ad",
    # "spatial/simulations_kidney_slideSeq_v2/UMOD-KI.KI-4b_resolution105.h5ad",
    "spatial/simulations_kidney_slideSeq_v2/UMOD-WT.WT-2a_resolution75.h5ad",
    "spatial/simulations_kidney_slideSeq_v2/UMOD-KI.KI-4b_resolution105.h5ad",
]

st_data_files = st_data_files_synth + st_data_files_real + st_data_files_switched

sc_data_files_synth = [
    "spatial/kidney_slideSeq_v2/UMOD-WT.WT-2a.h5ad",
    "spatial/kidney_slideSeq_v2/UMOD-KI.KI-4b.h5ad",
    "spatial/heart_seqFISH/embryo1.h5ad",
    "spatial/heart_seqFISH/embryo2.h5ad",
    "spatial/heart_seqFISH/embryo3.h5ad",
    "spatial/hypothalamus_MERFISH/0.h5ad",
    "spatial/hypothalamus_MERFISH/1.h5ad",
]

sc_data_files_real = [
    "single-cell/Allenbrain_forSimulation_uniquect.h5ad",
    "single-cell/lymph_node/sc_lymph.h5ad",
]

sc_data_files_switched = [
    "single-cell/kidney/kidney_full_v3_harmonized_celltypes_raw_mouseGenes.h5ad",
    "single-cell/kidney/kidney_full_v3_harmonized_celltypes_raw_mouseGenes.h5ad",
    "spatial/heart_seqFISH/embryo3.h5ad",
    "spatial/heart_seqFISH/embryo3.h5ad",
    "spatial/hypothalamus_MERFISH/2.h5ad",
    "spatial/hypothalamus_MERFISH/2.h5ad",
    # "single-cell/kidney/kidney_full_v3_harmonized_celltypes_raw_mouseGenes.h5ad",
    # "single-cell/kidney/kidney_full_v3_harmonized_celltypes_raw_mouseGenes.h5ad",
    "single-cell/kidney/kidney_full_v3_harmonized_celltypes_raw_mouseGenes.h5ad",
    "single-cell/kidney/kidney_full_v3_harmonized_celltypes_raw_mouseGenes.h5ad",
]

sc_data_files = sc_data_files_synth + sc_data_files_real + sc_data_files_switched

dataset_names_synth = [
    "slideSeqv2-kidney-2a",
    "slideSeqv2-kidney-4b",
    "seqFISH-heart-1",
    "seqFISH-heart-2",
    "seqFISH-heart-3",
    "MERFISH-hypothalamus-0",
    "MERFISH-hypothalamus-1",
]

dataset_names_real = [
    "Visium-mouse-brain",
    "Visium-lymph-node",
]

dataset_names_switched = [
    "slideSeqv2-kidney-2a (sc. ref.)",
    "slideSeqv2-kidney-4b (sc. ref.)",
    "seqFISH-heart-1 (switched)",
    "seqFISH-heart-2 (switched)",
    "MERFISH-hypothalamus-0 (switched)",
    "MERFISH-hypothalamus-1 (switched)",
    # "slideSeqv2-kidney-2a (sc. ref., per batch)",
    # "slideSeqv2-kidney-4b (sc. ref., per batch)",
    "slideSeqv2-kidney-2a (sc. ref., not var. cut.)",
    "slideSeqv2-kidney-4b (sc. ref., not var. cut.)",
]

dataset_names = dataset_names_synth + dataset_names_real + dataset_names_switched


celltype_cols_synth = [
    "cell_type",
    "cell_type",
    "celltype_mapped_refined",
    "celltype_mapped_refined",
    "celltype_mapped_refined",
    "cell_type",
    "cell_type",
]

celltype_cols_real = [
    "Celltype",
    "Subset",
]

celltype_cols_switched = [
    "cell_type",
    "cell_type",
    "celltype_mapped_refined",
    "celltype_mapped_refined",
    "cell_type",
    "cell_type",
    # "cell_type",
    # "cell_type",
    "cell_type",
    "cell_type",
]

extra_settings_sim = (
    [None for i in sc_data_files_synth]
    + [None for i in sc_data_files_real]
    + [
        None,
        None,
        None,
        None,
        None,
        None,
        # "per_batch",
        # "per_batch",
        None,
        None,
    ]
)

extra_settings_prep = (
    [None for i in sc_data_files_synth]
    + [None for i in sc_data_files_real]
    + [
        None,
        None,
        None,
        None,
        None,
        None,
        # None,
        # None,
        "no_var_cut",
        "no_var_cut",
    ]
)

celltype_cols = celltype_cols_synth + celltype_cols_real + celltype_cols_switched

# create dataset_map for naming
# map each dataset and reference to a unique name


# legacy should be computed automatically
experiment_dirs = [
    "experiments/experiment_kidney_slideSeq_v2_UMOD-WT.WT-2a_resolution75",
    "experiments/experiment_kidney_slideSeq_v2_105",
    "experiments/experiment_heart_seqFISH/embryo1_resolution0.11",
    "experiments/experiment_heart_seqFISH/embryo2_resolution0.11",
    "experiments/experiment_heart_seqFISH/embryo3_resolution0.11",
    "experiments/experiment_hypothalamus_MERFISH/0_resolution0.04",
    "experiments/experiment_hypothalamus_MERFISH/1_resolution0.04",
    "experiments/experiment_mouse_st",
    "experiments/experiment_lymph_node",
]
