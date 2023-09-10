import tensorflow as tf

class Stage1(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Stage1, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, padding="same")
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=3)
        self.relu = tf.keras.layers.Activation("relu")
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="same")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batchnorm(x)
        x = self.relu(x)
        output = self.maxpool(x)
        return output

class Identity_Block(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(Identity_Block, self).__init__(**kwargs)
        filters1, filters2, filters3 = filters
        self.conv1 = tf.keras.layers.Conv2D(filters=filters1, kernel_size=(1,1), padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters2, kernel_size=(3,3), padding="same")
        self.conv3 = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1,1), padding="same")
        self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=3)
        self.batchnorm2 = tf.keras.layers.BatchNormalization(axis=3) 
        self.batchnorm3 = tf.keras.layers.BatchNormalization(axis=3)
        self.relu = tf.keras.layers.Activation("relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)

        x = tf.keras.layers.add([x, inputs])
        output = self.relu(x)
        return output

class Conv_Block(tf.keras.layers.Layer):
    def __init__(self, filters, strides=(2, 2), **kwargs):
        super(Conv_Block, self).__init__(**kwargs)
        filters1, filters2, filters3 = filters
        self.conv1 = tf.keras.layers.Conv2D(filters=filters1, kernel_size=(1,1), padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters2, kernel_size=(3,3), padding="same")
        self.conv3 = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1,1), padding="same")
        self.conv4 = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1,1), padding="same")
        self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=3)
        self.batchnorm2 = tf.keras.layers.BatchNormalization(axis=3) 
        self.batchnorm3 = tf.keras.layers.BatchNormalization(axis=3)
        self.batchnorm4 = tf.keras.layers.BatchNormalization(axis=3)
        self.relu = tf.keras.layers.Activation("relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)

        shortcut = self.conv4(inputs)
        shortcut = self.batchnorm4(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        output = tf.keras.layers.Activation("relu")(x)
        return output
    
class ExtractFeatures(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ExtractFeatures, self).__init__(**kwargs)
        initial_weights = tf.keras.initializers.glorot_normal()
        self.stage1 = Stage1()
        self.conv2_1 = Conv_Block(filters=[64, 64, 256], strides=(1, 1))
        self.conv2_2 = Identity_Block(filters=[64, 64, 256])
        self.conv2_3 = Identity_Block(filters=[64, 64, 256])

        self.conv3_1 = Conv_Block(filters=[128, 128, 512])
        self.conv3_2 = Identity_Block(filters=[128, 128, 512])
        self.conv3_3 = Identity_Block(filters=[128, 128, 512])
        self.conv3_4 = Identity_Block(filters=[128, 128, 512])

        self.conv4_1 = Conv_Block(filters=[256, 256, 1024])
        self.conv4_2 = Identity_Block(filters=[256, 256, 1024])
        self.conv4_3 = Identity_Block(filters=[256, 256, 1024])
        self.conv4_4 = Identity_Block(filters=[256, 256, 1024])
        self.conv4_5 = Identity_Block(filters=[256, 256, 1024])
        self.conv4_6 = Identity_Block(filters=[256, 256, 1024])

        self.conv5_1 = Conv_Block(filters=[512, 512, 2048])
        self.conv5_2 = Identity_Block(filters=[512, 512, 2048])
        self.conv5_3 = Identity_Block(filters=[512, 512, 2048])

    def call(self, inputs):
        x = self.stage1(inputs)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        output = self.conv3_4(x)

        # x = self.conv4_1(x)
        # x = self.conv4_2(x)
        # x = self.conv4_3(x)
        # x = self.conv4_4(x)
        # x = self.conv4_5(x)
        # output = self.conv4_6(x)

        # x = self.conv5_1(x)
        # x = self.conv5_2(x)
        # output = self.conv5_3(x)
        return output
