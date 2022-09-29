#Rscript prepare_data_DSTG.R
cd DSTG/DSTG
Rscript convert_data.R dstg_sc.rds dstg_st.rds dstg_scc.rds
#Rscript convert_data.R synthetic_data/counts.rds synthetic_data/test.rds synthetic_data/temp.rds

python train.py
cd ../../
