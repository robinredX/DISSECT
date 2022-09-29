import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from anndata import read_h5ad
from tqdm import tqdm
from configs.main_config import config
from utils.network_fn import network
from utils.utils_fn import normalize_per_batch, reproducibility
from scaden2.evaluation_metrics import evaluate

class DISSECT(object):
    def __init__(self, hidden_units, sess, alpha_range, X_sim_np, X_real_np, y_sim_np, X_real_test, batch_size, learning_rate, n_steps, normalize_test, normalize_simulated, celltypes, n_celltypes, n_features, model_dir, model_name, seed):
        self.hidden_units = hidden_units
        self.sess = sess
        self.alpha_range = alpha_range
        self.x_data = X_sim_np
        self.x_data_real = X_real_np
        self.y_data = y_sim_np
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.normalize_test = normalize_test
        self.normalize_simulated = normalize_simulated
        self.n_steps = n_steps
        self.classes = celltypes
        self.n_classes = n_celltypes
        self.n_features = n_features
        self.model_dir = model_dir
        self.x_data_real_test = X_real_test
        self.model_name = model_name
        self.seed = seed

    def data_generator(self, mode):
        if mode=="train":
            self.x_data_ph = tf.compat.v1.placeholder(self.x_data.dtype, self.x_data.shape, name="x_data_ph")
            self.x_data_real_ph = tf.compat.v1.placeholder(self.x_data_real.dtype, self.x_data_real.shape, name="x_data_real_ph")
            self.y_data_ph = tf.compat.v1.placeholder(self.y_data.dtype, self.y_data.shape, name="y_data_ph")
            self.data = tf.data.Dataset.from_tensor_slices((self.x_data_real_ph, self.x_data_ph, self.y_data_ph))
            tf.random.set_seed(self.seed)
            self.data = self.data.shuffle(1000).repeat().batch(batch_size=self.batch_size)
        if mode=="predict":
            self.y_dummy = np.zeros((self.x_data_real_test.shape[0], self.n_classes), dtype=np.float32)
            self.x_data_ph = tf.compat.v1.placeholder(self.x_data_real_test.dtype, self.x_data_real_test.shape, name="x_data_ph_test")
            self.x_data_real_ph = tf.compat.v1.placeholder(self.x_data_real_test.dtype, self.x_data_real_test.shape, name="x_data_real_ph_test")
            self.y_data_ph = tf.compat.v1.placeholder(self.y_dummy.dtype, self.y_dummy.shape, name="y_data_ph_test")
            self.data = tf.data.Dataset.from_tensor_slices((self.x_data_real_ph, self.x_data_ph, self.y_data_ph))
            self.data = self.data.batch(batch_size=self.x_data_real_test.shape[0])

    def compute_loss(self, logits, alphas, targets, batch_size):
        """
        Compute regularized kldiv loss
        :param logits : List of logits
        :param targets: ground truth cell proportions for simulated sequences
        :return: loss
        """
        real_logits     = logits[0] 
        sim_logits      = logits[1]
        mixture_logits = logits[2]
        mixture_targets = alphas*real_logits + (1-alphas)*sim_logits + 1e-6

        kl              = tf.keras.losses.KLDivergence()

        sim_loss        = kl(targets, sim_logits)
        mixture_loss    = kl(mixture_targets, mixture_logits) 
        
        loss1 = tf.cond(self.global_step > 4000, lambda: sim_loss + 0.15*mixture_loss, lambda: sim_loss + 0.1*mixture_loss)
        loss = tf.cond(self.global_step <= 2000, lambda: sim_loss + 0.*mixture_loss, lambda: loss1)
        
        return [sim_loss, mixture_loss, loss]

    def model_fn(self, mode):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        iter = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(self.data), tf.compat.v1.data.get_output_shapes(self.data))
        next_element = iter.get_next()
        self.data_init_op = iter.make_initializer(self.data)

        self.x_real, self.x_sim, self.y = next_element
        self.x_real = tf.cast(self.x_real, tf.float32)
        self.x_sim  = tf.cast(self.x_sim, tf.float32)

        self.training_mode = tf.compat.v1.placeholder_with_default(True, shape=())
        activation = tf.nn.relu6
        #seed=int(np.random.uniform(1,100))
        self.logits, self.alphas = network(hidden_units=self.hidden_units, X_real=self.x_real, X_sim=self.x_sim, n_classes=self.n_classes, mode=self.mode, activation=activation, alpha_range=self.alpha_range, n_features=self.n_features)

        if mode == "train":
            # Loss
            self.loss = self.compute_loss(logits = self.logits, alphas = self.alphas, targets = self.y, batch_size = self.batch_size)
            # Optimizer
            learning_rate=self.learning_rate
            self.optimizer1 = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss[0])
            self.optimizer2 = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss[1])
            self.optimizer3 = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss[2], global_step=self.global_step)

    def train(self):
        mode="train"
        self.mode = mode
        self.data_generator(mode)
        self.model_fn(mode)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
        model_path = os.path.join(self.model_dir, self.model_name)
        self.writer = tf.compat.v1.summary.FileWriter(model_path, self.sess.graph)
        self.eval_writer = tf.compat.v1.summary.FileWriter(os.path.join(self.model_dir, "eval"), self.sess.graph)
        # Initialize datasets
        self.sess.run(self.data_init_op, feed_dict={self.x_data_real_ph: self.x_data_real, self.x_data_ph: self.x_data, self.y_data_ph: self.y_data})

        # Load pre-trained weights if avaialble
        self.load_weights(self.model_dir)
        
        # Training loop
        pbar = tqdm(range(self.n_steps))
        losses = []
        for _ in pbar:
            _, _, _, loss = self.sess.run([self.optimizer1, self.optimizer2, self.optimizer3, self.loss])
            description = "Step: " + str(tf.compat.v1.train.global_step(self.sess, self.global_step)) + ", Loss: {:4.4f}".format(
                loss[2]) + ", Regression loss: {:4.4f}".format(loss[0]) + ", Consistency loss: {:4.4f}".format(loss[1])
            pbar.set_description(desc=description)
            losses.append(loss[2])
        
        df_losses = pd.DataFrame(losses, index=list(range(self.n_steps)), columns=["loss"])
        df_losses.to_csv(os.path.join(model_path, "losses.txt"), sep="\t")

        self.saver.save(self.sess, model_path, global_step=self.global_step)

    def predict(self, sample_names):
        mode = "predict"
        self.mode = mode
        self.data_generator(mode)
        self.model_fn(mode)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())

        self.saver = tf.compat.v1.train.Saver()
        model_path = os.path.join(self.model_dir, self.model_name)
        self.writer = tf.compat.v1.summary.FileWriter(model_path, self.sess.graph)

        # Initialize datasets
        self.sess.run(self.data_init_op, feed_dict={self.x_data_real_ph: self.x_data_real_test, self.x_data_ph: self.x_data_real_test, self.y_data_ph: self.y_dummy})

        # Load pre-trained weights if avaialble
        self.load_weights(self.model_dir)

        predictions = self.sess.run([self.logits[1]], feed_dict={self.training_mode: False})
        pred_df = pd.DataFrame(predictions[0], columns=self.classes, index=sample_names)
        return pred_df

    def load_weights(self, model_dir):
        """
        Load pre-trained weights if available
        :param model_dir:
        :return:
        """
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model parameters restored successfully")
            
def run_dissect_tf1(config):
    dataset_path = os.path.join(config["experiment_folder"], "datasets")
    batch_size = config["network_params"]["batch_size"]
    if not os.path.exists(dataset_path):
        sys.exit("Path {} does not exist. Please run prepare_data.py before.".format(dataset_path))

    print("Loading prepared datasets...")
    X_real_np = np.load(os.path.join(dataset_path, "X_real_train.npy"))
    X_sim_np = np.load(os.path.join(dataset_path, "X_sim.npy"))
    y_sim_np = np.load(os.path.join(dataset_path, "y_sim.npy"))
    X_real_test = np.load(os.path.join(dataset_path, "X_real_test.npy"))
    sample_names = pd.read_table(os.path.join(dataset_path, "sample_names.txt"), index_col=0).index.tolist()
    celltypes = pd.read_table(os.path.join(dataset_path, "celltypes.txt"), index_col=0).index.tolist()

    X_real_np, X_sim_np, y_sim_np = np.array(X_real_np, dtype=np.float32), np.array(X_sim_np, dtype=np.float32), np.array(y_sim_np, dtype=np.float32)
    X_real_test = np.array(X_real_test, dtype=np.float32)
    n_features = X_sim_np.shape[1]
    n_celltypes = len(celltypes)

    network_params = config["network_params"]
    batch_size, learning_rate, n_steps = network_params["batch_size"], network_params["lr"], network_params["n_steps"]
    alpha_range = config["alpha_range"]
    normalize_test, normalize_simulated = config["normalize_test"], config["normalize_simulated"]
    seeds = config["seeds"]
    hidden_units = config["network_params"]["hidden_units"]
    i=0

    for seed in seeds:
        model_dir = os.path.join(config["experiment_folder"], "model_{}".format(i))
        os.mkdir(model_dir)
#        tf.compat.v1.reset_default_graph()
        model_name = "model_{}".format(i)
        print("training model {}".format(i))
        
        #os.environ['PYTHONHASHSEED'] = '2'
        #os.environ['TF_DETERMINISTIC_OPS'] = '1'
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        #reproducibility(seed)
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            model = DISSECT(hidden_units, sess, alpha_range, X_sim_np, X_real_np, y_sim_np, X_real_test, batch_size, learning_rate,  n_steps, normalize_test, normalize_simulated, celltypes, n_celltypes, n_features, model_dir, model_name, seed)
            model.train()
        del model
        tf.compat.v1.reset_default_graph()
        
        with tf.compat.v1.Session() as sess:
            model = DISSECT(hidden_units, sess, alpha_range, X_sim_np, X_real_np, y_sim_np, X_real_test, batch_size, learning_rate,  n_steps, normalize_test, normalize_simulated, celltypes, n_celltypes, n_features, model_dir, model_name, seed)
            predictions = model.predict(sample_names)
        del model
        savename = os.path.join(config["experiment_folder"], "dissect_fractions_{}.txt".format(i))
        print("Saving prediction {} to {}".format(i, savename))
        predictions.to_csv(savename, sep="\t")
        i+=1

    if len(seeds) > 1:
        i = 0
        for i in range(len(seeds)):
            temp = pd.read_table(os.path.join(config["experiment_folder"], "dissect_fractions_{}.txt".format(i)), index_col=0)
            if i==0:
                ens = temp
            else:
                ens = ens + temp
        ens = ens/len(seeds)
        savename = os.path.join(config["experiment_folder"], "dissect_fractions_{}.txt".format("ens"))
        print("Saving ensemble (average) prediction to {}".format(savename))
        ens.to_csv(savename, sep="\t")

if __name__=="__main__":
    run_dissect_tf1(config)
    
