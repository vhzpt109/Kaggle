import tensorflow as tf


class LeNet(tf.keras.Model):
    def __init__(self):
        super(LeNet, self).__init__(name='lenet')

        self.conv1 = tf.keras.layers.Conv2D(filters=20, kernel_size=5, padding="same", input_shape=(28, 28, 1))
        self.activation1 = tf.keras.layers.Activation("relu")
        self.pooling1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2 = tf.keras.layers.Conv2D(filters=50, kernel_size=5, padding="same")
        self.activation2 = tf.keras.layers.Activation("relu")
        self.pooling2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv3 = tf.keras.layers.Conv2D(filters=20, kernel_size=5, padding="same")
        self.activation3 = tf.keras.layers.Activation("relu")
        self.pooling3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(500)
        self.activation4 = tf.keras.layers.Activation("relu")
        self.dense2 = tf.keras.layers.Dense(10)
        self.activation5 = tf.keras.layers.Activation("softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.activation1(x)
        x = self.pooling1(x)

        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pooling2(x)

        x = self.conv3(x)
        x = self.activation3(x)
        x = self.pooling3(x)

        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.activation4(x)
        x = self.dense2(x)
        x = self.activation5(x)

        return x