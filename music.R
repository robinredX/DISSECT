library(Biobase)
library(MuSiC)
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

# Bulk
df_bulk <- read.csv(test_path, row.names=1)
eset_bulk <- ExpressionSet(as.matrix(df_bulk))

# single cell
df <- read.csv(ref_counts_path, row.names=1)
df_celltypes <- read.csv(ref_celltypes_path, row.names=1)
subjects <- rep("donor1", dim(df_celltypes)[1])
df_celltypes$subject <- subjects
if (subject_path){
  df_subjects <- read.csv(subject_path, row.names=1)
  df_celltypes$subject <- df_subjects[,1]
}
colnames(df_celltypes) <- c("cellType", "SubjectName")
df_celltypes$cellTypeID <- df_celltypes$cellType
df_celltypes$subjectID <- df_celltypes$SubjectName
rownames(df_celltypes) <- colnames(df)
pheno_data <- AnnotatedDataFrame(data=df_celltypes)
eset <- ExpressionSet(as.matrix(df), phenoData = pheno_data)

# Run MuSiC
est.prop <- music_prop(bulk.eset = eset_bulk, 
                       sc.eset = eset, 
                       clusters = 'cellType',
                       select.ct = unique(df_celltypes$cellType), 
                       verbose = F)
preds <- est.prop$Est.prop.weighted
write.csv(preds, "MuSiC_fractions.csv")

