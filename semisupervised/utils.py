import os, csv
import numpy as np
from sklearn.model_selection import train_test_split

def print_and_log(new_line, log_file=None):
    print(new_line)
    if log_file is not None:
        log_file.write(new_line+'\n')

        
def fetch_data(dataset):
    x_u = np.load(os.path.join(dataset, 'data_aa.npy'))
    print("x_u: %s"%(np.shape(x_u),))
    x_l = np.load(os.path.join(dataset, 'test_data_aa.npy'))
    print("x_l: %s"%(np.shape(x_l),))
    y_l = np.load(os.path.join(dataset, 'target_values.npy'))
    y_l = y_l.astype(np.float)
    print("y_l: %s"%(np.shape(y_l),))
    new_weights = np.load(os.path.join(dataset, 'weights.npy'))
    print("new_weights: %s"%(np.shape(new_weights),))
    
    train_x_u, val_x_u, train_w, val_w = train_test_split(x_u, new_weights, test_size=0.1, random_state=21)
    train_x_l, split_x_l, train_y_l, split_y_l = train_test_split(x_l, y_l, test_size=0.15)
    val_x_l, test_x_l, val_y_l, test_y_l = train_test_split(split_x_l, split_y_l, test_size=0.66667)
    seq_len = int(np.shape(x_l)[1])
    
    return train_x_u, val_x_u, train_w, val_w, train_x_l, train_y_l, val_x_l, val_y_l, test_x_l, test_y_l, seq_len
    
    
def return_k_n(d, file):
    with open(file,'r') as f:
        r = csv.reader(f)
        next(r)
        for line in r:
            if str(line[0]) == d:
                k = int(line[1])
                n = int(line[2])
                break
            else:
                continue
    return k,n


def next_batch(x, y, batch_size):
    index = np.arange(len(x))
    random_index = np.random.permutation(index)[:batch_size]
    return x[random_index], y[random_index] #(batch_size, seq_len)


def save(filename, save_results):
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(save_results)
        

class EarlyStopping():
    def __init__(self, patience=0):
        self.step = 0
        self.loss = float('inf')
        self.patience = patience

    def validate(self, loss):
        if self.loss < loss:
            self.step += 1
            if self.step > self.patience:
                return 2 # termination code
            else:
                return 1 # continue code
        else:
            self.step = 0
            self.loss = loss
            return 0 # reset code