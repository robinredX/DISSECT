st_data_files = [
    "spatial/anca_visium/test_sample.h5ad",
    "spatial/anca_visium/sample_V1_A.h5ad",
    "spatial/anca_visium/sample_V1_B.h5ad",
    "spatial/anca_visium/sample_V1_C.h5ad",
    "spatial/anca_visium/sample_V1_D.h5ad",
    "spatial/anca_visium/sample_V2_A.h5ad",
    "spatial/anca_visium/sample_V2_B.h5ad",
    "spatial/anca_visium/sample_V2_C.h5ad",
    "spatial/anca_visium/sample_V2_D.h5ad",
    "spatial/anca_visium/sample_V3_A.h5ad",
    "spatial/anca_visium/sample_V3_B.h5ad",
    "spatial/anca_visium/sample_V3_C.h5ad",
    "spatial/anca_visium/sample_V3_D.h5ad",
    "spatial/anca_visium/sample_V4_A.h5ad",
    "spatial/anca_visium/sample_V4_B.h5ad",
    "spatial/anca_visium/sample_V4_C.h5ad",
    "spatial/anca_visium/sample_V4_D.h5ad",
    "spatial/anca_visium/sample_V5_A.h5ad",
    "spatial/anca_visium/sample_V5_B.h5ad",
    "spatial/anca_visium/sample_V5_C.h5ad",
    "spatial/anca_visium/sample_V5_D.h5ad",
    "spatial/anca_visium/sample_V6_A.h5ad",
    "spatial/anca_visium/sample_V6_B.h5ad",
    "spatial/anca_visium/sample_V6_C.h5ad",
    "spatial/anca_visium/sample_V6_D.h5ad",
]

sc_data_files = [
    "single-cell/kidney/single_cell_object_raw.h5ad" for i in st_data_files
]

dataset_names = [f"ANCA-visium-{i}" for i in range(len(sc_data_files))]

extra_settings_sim = [None for i in sc_data_files]

extra_settings_prep = [None for i in sc_data_files]

celltype_cols = ["cell_type" for i in sc_data_files]

experiment_dirs = None