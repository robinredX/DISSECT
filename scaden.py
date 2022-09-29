from scaden.process import processing
from scaden.train import training
from scaden.predict import prediction

from configs.main_config import config

sim_path = config["reference"]
real_path = config["test_dataset"]
real_data_type = config["test_dataset_type"]
if real_data_type=="h5ad":
    real_path = [file for file in "results/CS_datasets/" if "test" in file][0]
savename = "results/scaden_results.txt"
var_cutoff = 0.1
processed_path = "results/scaden_processed.h5ad"
model_dir "results/scaden_model"

processing(data_path=real_path, 
            training_data=sim_path, 
            processed_path=processed_path, 
            var_cutoff=var_cutoff)
training(data_path=processed_path, 
        train_datasets="", 
        model_dir=model_dir, 
        batch_size=128, 
        learning_rate=0.0001, 
        num_steps=5000, 
        seed=0
)
prediction(model_dir=model_dir, 
            data_path=real_path,
            out_name=outname,
            seed=0
)