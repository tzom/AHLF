import tensorflow as tf

'''
this code is based on:

https://github.com/locuslab/TCN

@article{BaiTCN2018,
	author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
	title     = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
	journal   = {arXiv:1803.01271},
	year      = {2018},
}

'''


class TemporalBlock(tf.keras.Model):
    def __init__(self,filters,kernel_size,padding,dilation_rate,dropout_rate=0.0):
        super(TemporalBlock,self).__init__()
        init=tf.initializers.he_normal()

        self.conv1 = tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size,padding=padding,dilation_rate=dilation_rate,kernel_initializer=init)
        self.ac1 = tf.keras.layers.Activation('relu') 
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)

        self.conv2 = tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size,padding=padding,dilation_rate=dilation_rate,kernel_initializer=init)
        self.ac2 = tf.keras.layers.Activation('relu') 
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

        self.conv1x1 = tf.keras.layers.Conv1D(filters=filters,kernel_size=1,padding='same',kernel_initializer=init)
        self.ac1x1 = tf.keras.layers.Activation('relu') 

    def call(self,x,training):
        prev_x = x

        x=self.conv1(x)
        x=self.ac1(x)
        x=self.drop1(x)

        x=self.conv2(x)
        x=self.ac2(x)
        x=self.drop2(x)

        prev_x = self.conv1x1(prev_x)
        return self.ac1x1(prev_x + x)

class TCN(tf.keras.Model):
    def __init__(self,num_channels,kernel_size=2,padding='causal',dropout_rate=0.0):
        super(TCN,self).__init__()
        assert isinstance(num_channels, list)

        model = tf.keras.Sequential()

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_rate = 2 ** i
            model.add(TemporalBlock(num_channels[i], kernel_size,
                                    padding=padding, dilation_rate=dilation_rate, dropout_rate=dropout_rate))
        self.network = model

    def call(self, x, training):
        return self.network(x, training=training)