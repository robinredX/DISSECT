# Prepare data
library(dplyr)
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

# Single cell
df <- read.csv(ref_counts_path, row.names=1) # genes x cells
df_celltypes <- read.csv(ref_celltypes_path, row.names=1)
rownames(df_celltypes) <- colnames(df)
saveRDS(df_celltypes, "dstg_scc.rds")


# Spatial
df_spatial <- read.csv(test_path, row.names=1)
common_genes <- intersect(rownames(df), rownames(df_spatial))
df_spatial <- df_spatial[common_genes,]
df <- df[common_genes,]

rownames(df) <- rownames(df_spatial)
saveRDS(df_spatial, "dstg_st.rds")
saveRDS(df, "dstg_sc.rds")