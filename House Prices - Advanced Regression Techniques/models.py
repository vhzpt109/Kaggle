import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(self, num_classes=1):
        super(MLP, self).__init__(name='mlp')
        self.num_classes = num_classes

        self.dense1 = tf.keras.layers.Dense(8, activation='relu')
        self.dense2 = tf.keras.layers.Dense(8, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)

        return self.dense3(x)
