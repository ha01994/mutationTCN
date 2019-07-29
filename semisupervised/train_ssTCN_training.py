from __future__ import print_function
from model_ssTCN import Network
import numpy as np
import tensorflow as tf
import sys, os, itertools
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
from utils import *

def train_ssTCN_training(dataset):
    load_path = os.path.join(dataset, 'model/model.ckpt') #pretrained model path

    nchan = 128
    keep_prob = 0.7
    batch_size = 128
    lr_l = 0.0001
    l_training = 50000
    reg_scale = 0.2
    k, n = return_k_n('GAL4', 'k_n.csv')
    print('(k,n): (%d,%d)'%(k, n))
    

    _, _, _, _, train_x_l, train_y_l, val_x_l, val_y_l, test_x_l, test_y_l, seq_len = fetch_data(dataset)
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(10)
        X = tf.placeholder(tf.int32, [None, seq_len])
        y = tf.placeholder(tf.float32, [None])
        W = tf.placeholder(tf.float32, [None])
        p_keep = tf.placeholder('float')

        out, pred_val = Network(seq_len, nchan, n, k, p_keep)(X)

        pre_train_vars = [v for v in tf.trainable_variables() if "DNN" not in v.name]
        loss = tf.losses.mean_squared_error(y, pred_val)

        var = [v for v in tf.trainable_variables() if "DNN" in v.name]       

        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale, 
                                                          scope=None)
        reg_loss = tf.contrib.layers.apply_regularization(l2_regularizer, var)
        loss = loss + reg_loss

        train_l = tf.train.AdamOptimizer(lr_l).minimize(loss)

        saver = tf.train.Saver(pre_train_vars)

        print("Trainable parameters:", np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        pearsonr_=[]

        saver.restore(sess, load_path) 
        print("Model restored from file.")
        print("Start training prediction model...")

        for l in range(1, l_training+1):
            tr_x_l, tr_y_l = next_batch(train_x_l, train_y_l, batch_size)
            _, tr_l_error = sess.run([train_l, loss], 
                     feed_dict={X: tr_x_l, y: tr_y_l, p_keep: keep_prob})

            if l % 500 == 0: 
                v_l_error_ = 0
                for i in range(0, int(len(val_x_l)/3000) + 1):
                    v_l_e = sess.run(loss, feed_dict=
                                      {X: val_x_l[i*3000:(i+1)*3000], 
                                       y: val_y_l[i*3000:(i+1)*3000], p_keep: 1.0})
                    v_l_error_ += v_l_e
                v_l_error = np.mean(v_l_error_)
                print("%d"%l)
                print("training loss: %.4f, val loss:%.4f"%(tr_l_error, v_l_error))

                pred_l_ = np.array([])
                for i in range(0, int(len(val_x_l)/3000) + 1):
                    pred_l = sess.run(pred_val, feed_dict=
                                      {X: val_x_l[i*3000:(i+1)*3000], p_keep: 1.0})
                    pred_l_ = np.concatenate((pred_l_, pred_l))
                pr = pearsonr(pred_l_, val_y_l)[0]
                print("Val set pearsonr: %.4f"%(pr))

                if l == l_training:
                    print("...Terminating training after max # iterations...")


        curr_pred_ = np.array([])
        for i in range(0, int(len(test_x_l)/3000) + 1):
            curr_pred = sess.run(pred_val, feed_dict=
                                 {X: test_x_l[i*3000:(i+1)*3000], p_keep: 1.0})
            curr_pred_ = np.concatenate((curr_pred_, curr_pred))
        tpr = pearsonr(curr_pred_, test_y_l)[0]
        print('Test set pearsonr: %.4f'%tpr)

            
if __name__ == '__main__':
    dataset = '../data/'
    train_ssTCN_training(dataset)
    
