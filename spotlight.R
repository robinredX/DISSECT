library(SPOTlight)
library(Seurat)
library(dplyr)
# devtools::install_github('satijalab/seurat-data')
library(SeuratData)
library(rjson)

config <- fromJSON(file="configs/glob_config.json")
workdir <- paste(config[["experiment_folder"]], "/results/datasets", sep="")
setwd(workdir)
files <- list.files()

subject_path <- FALSE

for (i in 1:length(files)){
    if (grepl("test", files[i], fixed=TRUE)){
            test_path <- files[i]}
    if (grepl("counts", files[i], fixed=TRUE)){
            ref_counts_path <- files[i]
        }
    if (grepl("celltypes", files[i], fixed=TRUE)){
            ref_celltypes_path <- files[i]
        }
    if (grepl("subject", files[i], fixed=TRUE)){
            subject_path <- files[i]
        }
}

# single cell
df <- read.csv(ref_counts_path, row.names=1)
df_celltypes <- read.csv(ref_celltypes_path, row.names=1)

sc_data <- CreateSeuratObject(counts=df)
sc_data@meta.data$Celltype <- df_celltypes[,1]

sc_data <- Seurat::SCTransform(sc_data, verbose = FALSE)

# Preprocess
sc_data <- Seurat::RunPCA(sc_data, verbose = FALSE)
sc_data <- Seurat::RunUMAP(sc_data, dims = 1:30, verbose = FALSE)
sc_data <- Seurat::FindNeighbors(sc_data, dims = 1:30, verbose = FALSE)
sc_data <- Seurat::FindClusters(sc_data, verbose = FALSE)

# Spatial data
df_spatial <- read.csv(test_path, row.names=1)
st = CreateSeuratObject(counts = df_spatial, assay="Spatial")
st <- Seurat::SCTransform(st, assay = "Spatial", verbose = FALSE)
st <- Seurat::RunPCA(st, verbose = FALSE)
st <- Seurat::RunUMAP(st, dims = 1:30, verbose = FALSE)
st <- Seurat::FindNeighbors(st, dims = 1:30, verbose = FALSE)
st <- Seurat::FindClusters(st, verbose = FALSE)

Seurat::Idents(object = sc_data) <- sc_data@meta.data$Celltype
cluster_markers_all <- Seurat::FindAllMarkers(object = sc_data, 
                                              assay = "SCT",
                                              slot = "data",
                                              verbose = TRUE, 
                                              only.pos = TRUE, 
                                              logfc.threshold = 1,
                                              min.pct = 0.9)
set.seed(123)
spotlight_ls <- spotlight_deconvolution(se_sc = sc_data,
                                      counts_spatial = st@assays$Spatial@counts,
                                      clust_vr = "Celltype",
                                      cluster_markers = cluster_markers_all,
                                      cl_n = 100,
                                      hvg = 3000,
                                      ntop = NULL,
                                      transf = "uv",
                                      method = "nsNMF",
                                      min_cont = 0.09)

saveRDS(object = spotlight_ls,
        file = "spotlight_results.RDS")
decon_df <- as.data.frame(spotlight_ls[[2]])
write.csv(decon_df, "spotlight_results.csv")