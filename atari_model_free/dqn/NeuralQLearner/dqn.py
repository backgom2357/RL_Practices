import tensorflow as tf

class QValue:

    def __init__(self, config):
        self.config = config

    def DQN(self, action_space):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(16, (8, 8), strides=4, activation='relu', input_shape=(self.config.frame_height, self.config.frame_width,4)))
        model.add(tf.keras.layers.Conv2D(32, (4, 4), strides=2, activation='relu'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.Dense(action_space))
        model.summary()
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.config.learning_rate), momentum=self.config.gradient_momentum)
        return model