import random 
import tensorflow as tf
import numpy as np

def weights(shape):
    in_dim = shape[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.Variable(tf.random_normal(shape, stddev = xavier_stddev))

def bias(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))

aa_size = 24

class EmbeddingBlock(tf.layers.Layer):
    def __init__(self, seq_len, nchan, kernel_size, strides, p_keep, name=None):
        super(EmbeddingBlock, self).__init__(name=name)        
        self.p_keep = p_keep
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.nchan = nchan

    def build(self, input_shape):
        channel_dim = -1
        self.built = True
    
    def call(self, inputs, training=True):
        embedding_size = 100
        emb = tf.Variable(tf.random_uniform([aa_size, embedding_size], -1, 1), dtype=tf.float32)
        pads = tf.constant([[0,1], [0,0]])
        embeddings = tf.pad(emb, pads)
                
        w0 = weights([1, embedding_size, 1, self.nchan])
        b0 = bias([self.seq_len+1, 1, self.nchan])
        
        padding = 1
        pad = tf.constant([[0, 0], [padding, 0]]) #[batch_size, length]
        x = tf.pad(inputs, pad, constant_values=24) #[batch_size, seq_len+padding]
        x = tf.nn.embedding_lookup(embeddings, x) #[batch_size, seq_len+padding, embedding_size]
        x = tf.reshape(x, [-1, self.seq_len+padding, embedding_size, 1])
        x = tf.nn.conv2d(x, w0, padding='VALID', strides=[1,1,1,1], dilations=[1,1,1,1], data_format='NHWC')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x+b0)
        x = tf.nn.dropout(x, self.p_keep)
        
        return x
    
    
class TCBlock(tf.layers.Layer):
    def __init__(self, seq_len, nchan, kernel_size, strides, p_keep, dilation_rate, name=None):
        super(TCBlock, self).__init__(name=name)        
        self.p_keep = p_keep
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.nchan = nchan
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        channel_dim = -1
        self.built = True
    
    def call(self, inputs, training=True):
        w1 = weights([self.kernel_size, 1, self.nchan, self.nchan]) 
        
        padding = (self.kernel_size - 1) * self.dilation_rate
        pad = tf.constant([[0, 0], [padding, 0], [0,0], [0,0]]) #[batch_size, length, 1, channels]
        x = tf.pad(inputs, pad, constant_values=0)        
        x = tf.nn.conv2d(x, w1, padding='VALID', strides=[1,1,1,1], 
                         dilations=[1, self.dilation_rate, 1, 1], data_format='NHWC')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, self.p_keep)
        
        return x
    

class AttentionBlock(tf.layers.Layer):
    def __init__(self, seq_len, nchan, p_keep, name=None):
        super(AttentionBlock, self).__init__(name=name)
        self.seq_len = seq_len
        self.nchan = nchan
        self.p_keep = p_keep
        
    def build(self, input_shape):
        channel_dim = -1
        self.built = True
    
    def call(self, inputs, out0, training=True):
        wo = weights([2*self.nchan, aa_size])
        bo = bias([self.seq_len, aa_size])
        
        out0 = tf.squeeze(out0, [2])
        out0 = out0[:, :-1, :]
        x = tf.squeeze(inputs, [2])
        x = x[:, :-1, :]
        
        # Train attention weights
        wt = np.ones((self.seq_len, self.seq_len), dtype=np.float32) * (-9999)
        wt = np.triu(wt, 1)
        wa = tf.Variable(wt) #this is being updated
        wa_n = tf.constant(np.array(range(1, self.seq_len+1), dtype=np.float32))        
        
        wa_s = tf.nn.softmax(wa, axis=1) #row direction
        # tf.multiply is element-wise multiplication
        wa_s = tf.transpose(tf.multiply(tf.transpose(wa_s), wa_n))
        con = tf.einsum('jk,ikl->ijl', wa_s, out0) #[batch_size, seq_len, nchan]
        
        out = tf.concat([x, con], 2) #[batch_size, seq_len, 2*nchan]
        out = tf.layers.batch_normalization(out)
        out = tf.nn.dropout(out, self.p_keep)
        out = tf.reshape(out, [-1, 2*self.nchan]) #[batch_size*seq_len, 2*nchan]

        out = tf.matmul(out, wo) #[batch_size*seq_len, aa_size]
        out = tf.reshape(out, [-1, self.seq_len, aa_size]) #[batch_size, seq_len, aa_size]
        out = out + bo
        
        return out

    
class TemporalConvNet(tf.layers.Layer):
    def __init__(self, seq_len, nchan, levels, kernel_size, p_keep, name=None):
        super(TemporalConvNet, self).__init__(name=name)
        self.layers = []
        self.num_levels = levels
        
        self.layers.append(EmbeddingBlock(
                seq_len, nchan, kernel_size, strides=1, 
                p_keep=p_keep, name="tblock_{}".format(0)))

        for i in range(0, self.num_levels):
            dilation_size = 2 ** i
            self.layers.append(TCBlock(
                seq_len, nchan, kernel_size, strides=1, dilation_rate=2 ** i, 
                p_keep=p_keep, name="tblock_{}".format(i)))

        self.layers.append(AttentionBlock(
            seq_len, nchan, 
            p_keep=p_keep, name="tblock_{}".format(self.num_levels)))
        
    
    def call(self, inputs, training=True, **kwargs):
        outputs = inputs
        d={}
        for i in range(0, self.num_levels + 2): 
            if i == 0: #FirstBlock
                outputs = self.layers[i](outputs, training = training)
                d['outputs_%d'%i] = outputs
                
            elif i == self.num_levels + 1: #AttentionBlock
                if i % 2 == 0: 
                    outputs += d['outputs_%d'%(i-3)] #residual connection
                outputs = self.layers[i](outputs, d['outputs_%d'%0], 
                                               training = training)
                d['outputs_%d'%i] = outputs
                      
            else: #TemporalBlock
                if i != 2 and i % 2 == 0:
                    outputs += d['outputs_%d'%(i-3)] #residual connection
                outputs = self.layers[i](outputs, training = training)
                d['outputs_%d'%i] = outputs

        return outputs
                           
                           
