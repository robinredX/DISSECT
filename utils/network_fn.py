import tensorflow as tf

def network1(config, n_celltypes, n_features, training=False):
    """
    Creates network from config file (config), number of celltypes (n_celltypes)
    """
    model = tf.keras.Sequential()
    for l in range(config["n_hidden_layers"]):
        #model.add(tf.keras.layers.Dense(config["hidden_units"][l], "relu"))
        model.add(tf.keras.layers.Dense(config["hidden_units"][l], activation=config["hidden_activation"],
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)))
        if config["dropout"]:
            model.add(tf.keras.layers.Dropout(config["dropout"][l]))

    model.add(tf.keras.layers.Dense(n_celltypes, activation=config["output_activation"]))

    return model

def network2(config, n_celltypes, n_features, training=False):
    """
    Creates network from config file (config), number of celltypes (n_celltypes)
    """
    #model = tf.keras.Sequential()
    input = tf.keras.layers.Input(shape=(n_features,))
    for l in range(config["n_hidden_layers"]):
        if l==0:
            #x = tf.keras.layers.BatchNormalization()(input)
            x = tf.keras.layers.Dense(config["hidden_units"][l], activation=config["hidden_activation"])(input)
        else:
            #y = tf.keras.layers.BatchNormalization()(x)
            #y = x
            x = tf.keras.layers.Dense(config["hidden_units"][l], activation=None)(x)
            y = tf.keras.layers.Activation(config["hidden_activation"])(x)
            y = tf.keras.layers.Dropout(0.2)(y, training=training)
            y = tf.keras.layers.Dense(config["hidden_units"][l], activation=None)(y)
            y = tf.keras.layers.Dropout(0.2)(y, training=training)
            x = tf.keras.layers.Add()([x, y])

    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(config["hidden_activation"])(x)
    x = tf.keras.layers.Dropout(0.2)(x, training=training)
    x = tf.keras.layers.Dense(n_celltypes, activation=config["output_activation"])(x)

    model = tf.keras.Model(inputs=input, outputs=x)
    return model

def loss(loss_fn, y_hat_sim, y_sim, y_hat_real, y_hat_mix, alpha, y_hat_real_s=None):
    """
    Returns regression and consistency losses
    """
    if y_hat_real_s is not None:
        y_mix = alpha*y_hat_real + (1-alpha)*y_hat_real_s
    else:
        y_mix = alpha*y_hat_real + (1-alpha)*y_hat_sim

    if loss_fn == "kldivergence":
        KL = tf.keras.losses.KLDivergence()
        reg_loss = KL(y_sim, y_hat_sim)
        #cons_loss = KL(y_mix, y_hat_mix)
        cons_loss = tf.reduce_mean(tf.math.square(y_mix-y_hat_mix))
    elif loss_fn == "l2":
        reg_loss = tf.reduce_mean(tf.math.square(y_sim-y_hat_sim))
        cons_loss = tf.reduce_mean(tf.math.square(y_mix-y_hat_mix))
    elif loss_fn == "l1":
        reg_loss = tf.reduce_mean(tf.math.abs(y_sim-y_hat_sim))
        cons_loss = tf.reduce_mean(tf.math.abs(y_mix-y_hat_mix))

    return reg_loss, cons_loss

# tf11

import tensorflow as tf
from utils.utils_fn import normalize_per_batch

def network(hidden_units, X_real, X_sim, n_classes, alpha_range, activation, mode, n_features, reuse=True):
    """
    Model function
    :param X_real    : real bulk sequences
    :param X_sim     : Simulated sequences
    :param n_classes : Number of celltypes
    :param mode      : train or predict
    :return:
        list of logits [logit for real, logit for simulated, logit for mixture] and generated alpha
    """

    activation = activation
    with tf.compat.v1.variable_scope("dissect", reuse=tf.compat.v1.AUTO_REUSE): # layer weights are shared for X_real, X_sim and X_mix
        #tf.random.set_seed(42)
        alpha = tf.random.uniform([1], minval=alpha_range[0], maxval=alpha_range[1], dtype=tf.dtypes.float32, seed=None, name="alpha") # random.uniform(0.2, 0.8) # 0.4 # 
        if mode == 'train':
            X_mix = alpha*X_real + (1-alpha)*X_sim 
            
        if mode == 'predict':
            X_mix = X_real

        X_r = normalize_per_batch(X_real, n_features)
        X_s = normalize_per_batch(X_sim, n_features)
        X_m = normalize_per_batch(X_mix, n_features)
        
        layer10 = tf.compat.v1.layers.dense(X_r, units=hidden_units[0], activation=activation, name="dense1")
        layer11  = tf.compat.v1.layers.dense(X_s, units=hidden_units[0], activation=activation, name="dense1")
        layer12  = tf.compat.v1.layers.dense(X_m, units=hidden_units[0], activation=activation, name="dense1")
        
        layer20 = tf.compat.v1.layers.dense(layer10, units=hidden_units[1], activation=activation, name="dense2")
        layer21 = tf.compat.v1.layers.dense(layer11, units=hidden_units[1], activation=activation, name="dense2")
        layer22 = tf.compat.v1.layers.dense(layer12, units=hidden_units[1], activation=activation, name="dense2")
        
        layer30 = tf.compat.v1.layers.dense(layer20, units=hidden_units[2], activation=activation, name="dense3")
        layer31 = tf.compat.v1.layers.dense(layer21, units=hidden_units[2], activation=activation, name="dense3")
        layer32 = tf.compat.v1.layers.dense(layer22, units=hidden_units[2], activation=activation, name="dense3")
        
        layer40 = tf.compat.v1.layers.dense(layer30, units=hidden_units[3], activation=activation, name="dense4")
        layer41 = tf.compat.v1.layers.dense(layer31, units=hidden_units[3], activation=activation, name="dense4")
        layer42 = tf.compat.v1.layers.dense(layer32, units=hidden_units[3], activation=activation, name="dense4")

        logits0 = tf.compat.v1.layers.dense(layer40, units=n_classes, activation=tf.nn.softmax, name="logits_layer")
        logits1 = tf.compat.v1.layers.dense(layer41, units=n_classes, activation=tf.nn.softmax, name="logits_layer")
        logits2 = tf.compat.v1.layers.dense(layer42, units=n_classes, activation=tf.nn.softmax, name="logits_layer")
        
        return [logits0, logits1, logits2], alpha