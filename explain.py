import os
import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
import tensorflow as tf
import shap
from configs.main_config import config
import matplotlib.pyplot as plt
from anndata import AnnData
#from tensorflow.python.ops.numpy_ops import np_config

def log1p(x):
    ones = tf.ones_like(x, name='ones')
    x1 = tf.math.add(x, ones)

    numerator = tf.math.log(x1)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))

    return numerator / denominator
    
def normalize_per_batch(x, n_features, epsilon=1e-8):
    x1 = log1p(x)
    min_, max_ = tf.reduce_min(x1, axis=1), tf.reduce_max(x1, axis=1)
    min1, max1 = tf.tile(tf.expand_dims(min_, axis=1), tf.constant([1,n_features])), tf.tile(tf.expand_dims(max_, axis=1), tf.constant([1,n_features]))
    x_normed = (x1 - min1) / (max1 - min1 + epsilon) # epsilon to avoid dividing by zero

    return x_normed

def scaling(x):
    x = np.log2(x + 1)
    mms = pp.MinMaxScaler(feature_range=(0, 1), copy=True)
    x = mms.fit_transform(x)

    return x

def explain(config):
    model_no = 0

    data_path = os.path.join(config["experiment_folder"], "datasets")

    X_train = np.load(os.path.join(data_path, "X_sim.npy")) 
    y_train = np.load(os.path.join(data_path, "y_sim.npy")) 
    X_test = np.load(os.path.join(data_path, "X_real_test.npy")) 
    sample_names = pd.read_table(os.path.join(data_path, "sample_names.txt"), index_col=0)
    celltypes = pd.read_table(os.path.join(data_path, "celltypes.txt"), index_col=0).index.tolist()
    genes = pd.read_table(os.path.join(data_path, "genes.txt"), index_col=0).index.tolist()
    
    X_train_norm = normalize_per_batch(tf.constant(X_train), X_train.shape[1]).numpy()
    X_test_norm = normalize_per_batch(tf.constant(X_test), X_test.shape[1]).numpy()
    df_train = pd.DataFrame(X_train_norm, columns=genes, index=["sample_{}".format(i) for i in range(X_train.shape[0])])
    df_test = pd.DataFrame(X_test_norm, columns=genes, index=["sample_{}".format(i) for i in range(X_test.shape[0])])
    #df_train[df_train!=-1]=0
    #df_train = pd.DataFrame(scaling(X_train), columns=genes, index=["sample_{}".format(i) for i in range(X_train.shape[0])])
    #df_test = pd.DataFrame(scaling(X_test), columns=genes, index=["sample_{}".format(i) for i in range(X_test.shape[0])])

    adata_train = AnnData(df_train, obs=pd.DataFrame(y_train, index=df_train.index, columns=celltypes))
    #idxs = adata_train[adata_train.obs["CD8+ IL17+"]==0].obs.index.tolist()
    #background = np.array(df_train.sample(n=int(df_train.shape[0]/2)))
    background = np.array(df_train) #
    #background = np.array(df_train.loc[idxs])
    #print(background)
    tf.compat.v1.disable_v2_behavior()
    #np_config.enable_numpy_behavior()

    for model_no in [0]: #range(len(config["seeds"])):
        model_folder = os.path.join(config["experiment_folder"], "model_{}".format(model_no))

        model = tf.keras.models.load_model(model_folder)
        e = shap.GradientExplainer(model, background)
        if model_no==0:
            shap_values = e.shap_values(np.array(df_test))
        else:
            temp = e.shap_values(np.array(df_test))
            shap_values = [shap_values[i] + temp[i] for i in range(len(temp))]
            #shap_values = shap_values + temp #e.shap_values(np.array(df_test))
            break

    #shap_values = [shap_values[i]/len(config["seeds"]) for i in range(len(shap_values))]
    
    #print(e.expected_value)
    #df_exp = pd.DataFrame(e.expected_value, index=celltypes, columns=["Expected value"])
    
    shap_dir = os.path.join(config["experiment_folder"], "shap_grad")
    if not os.path.exists(shap_dir):
        os.mkdir(shap_dir)
    else:
        pass #sys.exit("shap_dir {} already exists. Remove the older directory if no longer needed, or specify a new directory".format(shap_dir))

    #df_exp.to_csv(os.path.join(shap_dir, "expected_values.txt"), sep="\t")
    #print("Expected values of each celltype are saved at {}".format(os.path.join(shap_dir, "expected_values.txt")))
    
    shap_values_dict = {}
    for i in range(len(celltypes)):
        celltype = celltypes[i]
        shap_values_dict[celltype] = shap_values[i]
    savepath = os.path.join(shap_dir, "shap_values_all.npy")
    np.save(savepath, shap_values, allow_pickle=True)
    print("Shapley values of each celltype and each sample are saved at {}".format(os.path.join(shap_dir, "shap_values_all.npy")))
    for i in range(len(celltypes)):
        celltype = celltypes[i]
        if celltype!="Unknown":
            shap.summary_plot(shap_values_dict[celltype], df_test, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, "{}_shap_values.png".format(celltype)))
            plt.close()
    print("Plots for mean shap values per celltype are saved at {}".format(shap_dir))

if __name__=="__main__":
    explain(config)