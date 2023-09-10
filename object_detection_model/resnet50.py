# Import TensorFlow library
import tensorflow as tf

# Define a custom layer 'Stage1' as a subclass of tf.keras.layers.Layer
class Stage1(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Stage1, self).__init__(**kwargs)
        # Create layers: Convolutional, BatchNormalization, ReLU, MaxPooling
        self.conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same")
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=3)
        self.relu = tf.keras.layers.Activation("relu")
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")

    def call(self, inputs):
        # Define the forward pass of the layer
        x = self.conv(inputs)
        x = self.batchnorm(x)
        x = self.relu(x)
        output = self.maxpool(x)
        return output

# Define a custom layer 'Identity_Block' as a subclass of tf.keras.layers.Layer
class Identity_Block(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(Identity_Block, self).__init__(**kwargs)
        filters1, filters2, filters3 = filters
        # Create convolutional layers and batch normalization for identity block
        self.conv1 = tf.keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters2, kernel_size=(3, 3), padding="same")
        self.conv3 = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1), padding="same")
        self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=3)
        self.batchnorm2 = tf.keras.layers.BatchNormalization(axis=3) 
        self.batchnorm3 = tf.keras.layers.BatchNormalization(axis=3)
        self.relu = tf.keras.layers.Activation("relu")

    def call(self, inputs):
        # Define the forward pass of the layer
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)

        # Add the input tensor to the output (skip connection) and apply ReLU
        x = tf.keras.layers.add([x, inputs])
        output = self.relu(x)
        return output

# Define a custom layer 'Conv_Block' as a subclass of tf.keras.layers.Layer
class Conv_Block(tf.keras.layers.Layer):
    def __init__(self, filters, strides=(2, 2), **kwargs):
        super(Conv_Block, self).__init__(**kwargs)
        filters1, filters2, filters3 = filters
        # Create convolutional layers and batch normalization for convolutional block
        self.conv1 = tf.keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters2, kernel_size=(3, 3), padding="same")
        self.conv3 = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1), padding="same")
        self.conv4 = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1), padding="same")
        self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=3)
        self.batchnorm2 = tf.keras.layers.BatchNormalization(axis=3) 
        self.batchnorm3 = tf.keras.layers.BatchNormalization(axis=3)
        self.batchnorm4 = tf.keras.layers.BatchNormalization(axis=3)
        self.relu = tf.keras.layers.Activation("relu")

    def call(self, inputs):
        # Define the forward pass of the layer
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)

        # Create a shortcut connection with a convolutional layer and batch normalization
        shortcut = self.conv4(inputs)
        shortcut = self.batchnorm4(shortcut)

        # Add the shortcut tensor to the output and apply ReLU
        x = tf.keras.layers.add([x, shortcut])
        output = tf.keras.layers.Activation("relu")(x)
        return output

# Define a custom model 'ExtractFeatures' as a subclass of tf.keras.Model
class ExtractFeatures(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ExtractFeatures, self).__init__(**kwargs)
        initial_weights = tf.keras.initializers.glorot_normal()
        # Create the layers for feature extraction
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
        # Define the forward pass of the model
        x = self.stage1(inputs)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
       
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        output = self.conv5_3(x)
        return output
