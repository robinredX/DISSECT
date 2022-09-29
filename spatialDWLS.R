# install.packages("Rfast")
library(Giotto)
library(rjson)

config <- fromJSON(file="configs/glob_config.json")
workdir <- paste(config[["experiment_folder"]], "/results/datasets", sep="")
setwd(workdir)
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
    #if (grepl("loc", files[i], fixed=TRUE)){
    #        loc_path <- files[i]
    #    }
}

# single cell
df <- read.csv(ref_counts_path, row.names=1)
df_celltypes <- read.csv(ref_celltypes_path, row.names=1)
sc_data <- createGiottoObject(raw_exprs = as.matrix(df))
sc_data <- normalizeGiotto(gobject = sc_data, scalefactor = 6000, verbose = T)
sc_data <- calculateHVG(gobject = sc_data)

gene_metadata = fDataDT(sc_data)
featgenes = gene_metadata[hvg == 'yes']$gene_ID
sc_data <- runPCA(gobject = sc_data, genes_to_use = featgenes, scale_unit = F)
signPCA(sc_data, genes_to_use = featgenes, scale_unit = F)

#sc_data@cell_metadata$leiden_clus <- df_celltypes[,1]
sc_data@cell_metadata$cell$rna$leiden_clus <- df_celltypes[,1]

scran_markers_subclusters = findMarkers_one_vs_all(gobject = sc_data,
                                                   method = 'scran',
                                                   expression_values = 'normalized',
                                                   cluster_column = 'leiden_clus')
Sig_scran <- unique(scran_markers_subclusters$feats[which(scran_markers_subclusters$ranking <= 100)])

norm_exp<-2^(sc_data@norm_expr)-1
id<-heart_sc@cell_metadata$leiden_clus
ExprSubset<-norm_exp[Sig_scran,]
Sig_exp<-NULL
for (i in unique(id)){
  Sig_exp<-cbind(Sig_exp,(apply(ExprSubset,1,function(y) mean(y[which(id==i)]))))
}
colnames(Sig_exp)<-unique(id)

# Spatial
instr <- createGiottoInstructions(python_path = "/opt/homebrew/Caskroom/miniforge/base/envs/tf24_1/bin/python")
df_spatial <- read.csv(test_path, row.names=1)
#spatial_loc <- read.csv(loc_path, row.names=1)

st <- createGiottoObject(raw_exprs = as.matrix(df_spatial), instructions=instr)
st <- filterGiotto(gobject = st, expression_threshold = 1,
                                     gene_det_in_min_cells = 10,
                                     min_det_genes_per_cell = 200,
                                     expression_values = c('raw'),
                                     verbose = T)
st <- normalizeGiotto(gobject = st)
st <- calculateHVG(gobject = st)
gene_metadata = fDataDT(st)
featgenes = gene_metadata[hvg == 'yes']$gene_ID
st <- runPCA(gobject = st, genes_to_use = featgenes, scale_unit = F)
signPCA(st, genes_to_use = featgenes, scale_unit = F)
st <- createNearestNetwork(gobject = st, dimensions_to_use = 1:10, k = 10)
#st <- doLeidenCluster(gobject = st, resolution = 0.4, n_iterations = 1000)
st <- doKmeans(gobject = st, centers = 4, name = 'kmeans_clus')
# Run spatialDWLS
st <- runDWLSDeconv(gobject = st, sign_matrix = Sig_exp, cluster_column="kmeans_clus")
results <- as.data.frame(st@spatial_enrichment$DWLS)[-1]
write.csv(decon_df, "spatialDWLS_results.csv")


##########
# Spatial
instr <- createGiottoInstructions(python_path = "/opt/homebrew/Caskroom/miniforge/base/envs/tf24_1/bin/python")
df_spatial <- read.csv(test_path, row.names=1)
st <- CreateSeuratObject(counts=df_spatial)
st_data <- createGiottoObject(
  raw_exprs = st@assays$RNA@counts,
  instructions = instr
)
st_data <- normalizeGiotto(gobject = st_data)
st_data <- calculateHVG(gobject = st_data)
gene_metadata = fDataDT(st_data)
featgenes = gene_metadata[hvf == 'yes']$gene_ID
st_data <- runPCA(gobject = st_data, genes_to_use = featgenes, scale_unit = F)
st_data <- runUMAP(st_data, dimensions_to_use = 1:10)
st_data <- createNearestNetwork(gobject = st_data, dimensions_to_use = 1:10, k = 15)
st_data <- doKmeans(gobject = st_data, centers=4)

# single cell
df <- read.csv(ref_counts_path, row.names=1)
df_celltypes <- read.csv(ref_celltypes_path, row.names=1)
sc <- CreateSeuratObject(counts=df)
sc@meta.data$celltype_final <- df_celltypes
sc_data <- createGiottoObject(
  raw_exprs = sc@assays$RNA@counts,
  instructions = instr
)
sc_data <- normalizeGiotto(gobject = sc_data)
sc_data <- calculateHVG(gobject = sc_data)
gene_metadata = fDataDT(sc_data)
featgenes = gene_metadata[hvf == 'yes']$gene_ID
sc_data <- runPCA(gobject = sc_data, genes_to_use = featgenes, scale_unit = F)
signPCA(sc_data, genes_to_use = featgenes, scale_unit = F)

sc_data@cell_metadata$cell$rna$leiden_clus <- as.character(sc@meta.data[,"celltype_final"])
scran_markers_subclusters = findMarkers_one_vs_all(gobject = sc_data,
                                                   method = 'scran',
                                                   expression_values = 'normalized',
                                                   cluster_column = 'leiden_clus')
Sig_scran <- unique(scran_markers_subclusters$feats[which(scran_markers_subclusters$ranking <= 100)])
norm_exp<-2^(sc_data@expression$cell$rna$normalized)-1
id<-sc_data@cell_metadata$cell$rna$leiden_clus
ExprSubset<-norm_exp[Sig_scran,]
Sig_exp<-NULL
for (i in unique(id)){
  Sig_exp<-cbind(Sig_exp,(apply(ExprSubset,1,function(y) mean(y[which(id==i)]))))
}
colnames(Sig_exp)<-unique(id)

st_data <- runDWLSDeconv(st_data, sign_matrix = Sig_exp, n_cell = 20, cluster_column = "kmeans")
results <- st_data@spatial_enrichment$cell$DWLS
write.csv(results, "spatialDWLS_results.txt")