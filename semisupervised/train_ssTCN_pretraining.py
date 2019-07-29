from __future__ import print_function
from model_ssTCN import Network
import numpy as np
import tensorflow as tf
import sys, os, itertools, csv
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
from utils import *
from train_ssTCN_training import *

dataset = '../data/'

save_path = os.path.join(dataset, 'model/model.ckpt') #pretrained model path
train_x_u, val_x_u, train_w, val_w, _, _, _, _, _, _, seq_len = fetch_data(dataset)

hidden = 10
nchan = 128
batch_size = 128
keep_prob = 0.7
lr_u = 0.001
u_training = 150000
min_iter = 100000
patience = 10
k, n = return_k_n('GAL4', 'k_n.csv')
print('(k,n): (%d,%d)'%(k, n))

tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(10)
    X = tf.placeholder(tf.int32, [None, seq_len])
    y = tf.placeholder(tf.float32, [None])
    W = tf.placeholder(tf.float32, [None])
    p_keep = tf.placeholder('float')

    out, pred_val = Network(seq_len, nchan, n, k, p_keep)(X)

    global_step = tf.Variable(0, trainable=False)

    pre_train_vars = [v for v in tf.trainable_variables() if "DNN" not in v.name]
    print(len(pre_train_vars))

    loss_u = tf.reduce_mean(tf.multiply(tf.reduce_sum(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=X, logits=out), axis=1),
                                        (W/tf.reduce_mean(W))))

    train_u = tf.train.RMSPropOptimizer(lr_u).minimize(loss_u, global_step=global_step)

    saver = tf.train.Saver(pre_train_vars)

    print("Trainable parameters:", np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))


with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    early_stopper = EarlyStopping(patience=patience)
    print("Start pretraining generative model...")

    for u in range(1, u_training+1):
        tr_x_u, tr_w = next_batch(train_x_u, train_w, batch_size)
        _, tr_u_error = sess.run([train_u, loss_u], feed_dict=
                                 {X: tr_x_u, W: tr_w, p_keep: keep_prob})

        if u % 1000 == 0:
            v_x_u, v_w = next_batch(val_x_u, val_w, batch_size)
            v_u_error = sess.run(loss_u, feed_dict={X: v_x_u, W: v_w, p_keep: 1.0})
            print("%d"%u)
            print("Unlabeled - training loss: %.4f, val loss: %.4f"%
                      (tr_u_error, v_u_error))

            if u >= min_iter:
                early_stop_code = early_stopper.validate(v_u_error)
                if early_stop_code == 0: pass #reset
                if early_stop_code == 1: pass #continue
                if early_stop_code == 2: #termination
                    print("...Terminating training by early stopper...")
                    print("...Saving current state...")
                    save_path = saver.save(sess, save_path)
                    print("Model saved in file: %s" % save_path)

                    break

            if u == u_training:
                print("...Terminating training after max # iterations...")
                print("...Saving current state...")
                save_path = saver.save(sess, save_path)
                print("Model saved in file: %s" % save_path)

train_ssTCN_training(dataset)


