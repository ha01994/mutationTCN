from __future__ import print_function
from model import TemporalConvNet
import numpy as np
import random, sys
from scipy.stats import spearmanr
import tensorflow as tf
import os, csv
from sklearn.model_selection import train_test_split

dataset = '../data/'
data_aa = np.load(os.path.join(dataset, 'data_aa.npy'))
print(np.shape(data_aa)) 
test_data_aa = np.load(os.path.join(dataset, 'test_data_aa.npy'))
print(np.shape(test_data_aa)) 
target_values = np.load(os.path.join(dataset, 'target_values.npy'))
print(np.shape(target_values)) 
with open (os.path.join(dataset, 'weights.npy'),"rb") as to_read:
    new_weights = np.load(to_read)
    print(np.shape(new_weights))

train_data, val_data, train_w, val_w = train_test_split(data_aa, new_weights, test_size=0.1, random_state=21)

nchan = 128
keep_prob = 0.7
batch_size = 128
display_step = 1000
training_steps = 150000
starter_learning_rate = 0.001
seq_len = int(np.shape(data_aa)[1])

with open('k_n.csv','r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        if str(line[0]) == 'GAL4':
            kernel_size = int(line[1])
            levels = int(line[2])
            break
        else:
            continue

for run in range(5):
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(10)
        X = tf.placeholder(tf.int32, [None, seq_len])
        W = tf.placeholder(tf.float32, [None])
        p_keep = tf.placeholder('float')

        out = TemporalConvNet(seq_len, nchan, levels, kernel_size, p_keep)(X)

        loss = tf.reduce_mean(
                tf.multiply(
                        tf.reduce_sum(
                                tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        labels=X, logits=out), axis=1), 
                    (W/tf.reduce_mean(W))))

        test_out = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= X, logits=out), axis=1)

        global_step = tf.Variable(0, trainable=False)

        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   decay_steps=50000, decay_rate=0.7, 
                                                   staircase=True)

        train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

        init = tf.global_variables_initializer()

        print("Trainable parameters:", np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))

    def next_batch(x, w, batch_size):
        index = np.arange(len(x))
        random_index = np.random.permutation(index)[:batch_size]
        return x[random_index], w[random_index] #(batch_size, seq_len), (batch_size,)


    with tf.Session(graph=graph) as sess:
        sess.run(init)
        for step in range(1, training_steps+1):
            X_b, W_b = next_batch(train_data, train_w, batch_size)
            sess.run(train_op, feed_dict={X: X_b, W: W_b, p_keep: keep_prob})

            if step % display_step == 0:
                train_error, lr = sess.run([loss, learning_rate], 
                                           feed_dict={X: X_b, W: W_b, p_keep: 1.0})
                X_b, W_b = next_batch(val_data, val_w, batch_size)
                val_error = sess.run(loss, feed_dict={X: X_b, W: W_b, p_keep: 1.0})

                if len(test_data_aa) <= 3000: 
                    logp = -sess.run(test_out, feed_dict={X:test_data_aa, 
                                                          p_keep: 1.0})
                    y1 = target_values
                    spearman = spearmanr(logp,y1)[0]

                else:
                    logp_ = np.array([])                        
                    for i in range(0, int(len(test_data_aa)/3000) + 1):
                        logp = -sess.run(test_out, feed_dict= 
                                         {X: test_data_aa[i*3000:(i+1)*3000], 
                                          p_keep: 1.0})
                        logp_ = np.concatenate((logp_,logp))
                    y1 = target_values
                    spearman = spearmanr(logp_,y1)[0]

                print(step)
                print("Losses:", train_error, val_error)
                print("Spearman: %.4f"%(spearman))


