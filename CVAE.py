"""
Conditional Varational Autoencoder script.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, Input
from tensorflow.keras.models import Model

###############################################################################

#create a sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim)) 
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#-----------------------------------------------------------------------------#
    
#build the encoder    
def encoder(dimensions):
    inputs = Input(shape=(dimensions + 1,)) 
    x = layers.Dense(512, activation='swish')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='swish')(x)
    z_mean = layers.Dense(128, name="z_mean")(x)
    z_log_var = layers.Dense(128, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return Model(inputs=inputs,outputs=[z_mean, z_log_var, z], name="encoder")

#-----------------------------------------------------------------------------#

#build the decoder
def decoder(dimensions):
    latent_inputs = Input(shape=(129,))
    x = layers.Dense(256, activation='swish')(latent_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    decoder_outputs = layers.Dense(dimensions, activation="sigmoid")(x)
    return Model(inputs=latent_inputs,outputs=[decoder_outputs], name="decoder")

#-----------------------------------------------------------------------------#

#define the CVAE as a Model with a custom training step
class condVAE(Model):
    def __init__(self, dimensions):
        super().__init__()
        self.dimensions = int(dimensions)
        self.encoder = encoder(self.dimensions)
        self.decoder = decoder(self.dimensions)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
    
    def call(self, data):
        x, labels = data
        x_con = tf.concat([x, labels],  -1)
        z_mean, z_log_var, z = self.encoder(x_con)
        encoded = tf.concat([z, labels], -1)
        decoded = self.decoder(encoded)
        return decoded
    
    #custom training step
    def train_step(self, data):
        with tf.GradientTape() as tape:
            x, labels = data
            x_con = tf.concat([x, labels],  -1)
            z_mean, z_log_var, z = self.encoder(x_con)
            encoded = tf.concat([z, labels], -1)
            decoded = self.decoder(encoded)
            reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(x, decoded))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1,keepdims=True))
            total_loss = reconstruction_loss + 0.001*kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }