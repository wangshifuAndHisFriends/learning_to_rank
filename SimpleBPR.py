# -*- coding: utf-8 -*-
"""

Simpe BPR tasks

Created on Mon Nov 27 13:31:54 2017

@author: TF.ZHOU
"""
import tensorflow as tf
import numpy as np

#----------dataset generating--------------------------------------
def generate_datasets(num_user, num_item, num_rel, train_ratio=0.7):
    r = 10
    user_vec = np.random.normal(size=[num_user, r])
    item_vec = np.random.normal(size=[num_item, r])
    scores   = np.dot(user_vec, item_vec.T)
    top_items = np.argsort(scores, axis=1)
    top_items = top_items[:, :num_rel]
    # split top_items to train and test
    train_clks = []
    test_clks = []
    for item in top_items:
        coins = np.random.uniform(size=[num_rel])
        coins = coins < train_ratio
        train_items = item[coins]
        test_items  = item[~coins]
        train_clks.append(train_items)
        test_clks.append(test_items)
    return train_clks, test_clks

def padding(list_arr, lens ):
    max_len = np.max(lens)
    list_pad = []
    for arr in list_arr:
        if len(arr) < max_len:
            num_pad = max_len - len(arr)
            arr_pad = np.hstack( (arr, np.zeros(num_pad, dtype=np.int)) )
            list_pad.append(arr_pad)
        else:
            list_pad.append(arr)
    return np.vstack(list_pad)
            

def generate_batch(train_clks, user_size, item_size, num_item):
    '''
    args:
        train_clks: list of numpy array size [user_size]
    return:
        users: numpy vector
        positive_items: matrix of size [user_size, item_size]
        negative_items: matrix of size [user_size, item_size]
        pos_lens: length of each positve records
        neg_lens
    '''
    # get users
    num_user = len(train_clks)
    users = np.random.choice(num_user, user_size, replace=False)
    # generate positive items
    positive_items = []
    negative_items = []
    pos_lens = []
    neg_lens = []
    for user in users:
        items = train_clks[user]
        tot_clks = len(items)
        if tot_clks < item_size:
            positive_items.append(items)
            pos_lens.append(tot_clks)
        else:
            positive_items.append(np.random.choice(items, item_size,replace=False))
            pos_lens.append(item_size)
        # get negative samples of the same size
        num_neg = pos_lens[-1]
        neg_lens.append(num_neg)
        neg_items = np.random.choice(num_item, num_neg, replace=False)
        negative_items.append(neg_items)
    pos_lens = np.array(pos_lens)
    neg_lens = np.array(neg_lens)
    positive_items = padding( positive_items, pos_lens )
    negative_items = padding( negative_items, neg_lens )
    return users, positive_items, negative_items, pos_lens, neg_lens

def compute_auc(score_on_positives, score_on_negtives, pos_lens, neg_lens):
    '''
    args:
        score_on_positive: tensor of size [num_user, max_pos_len]
        score_on_negative: tensor of size [num_user, max_neg_len]
        pos_lens: postive score length
        neg_lens: negative score length
    return:
        AUC where 
        AUC = 1/U \sum_{u} 1/ (|num_positive_u| * |num_negative_u|)
            \sum_{j,k}I{ score_positive[u,j] > score_negative[u, k] }
    '''
    auc = 0.0
    usr = 0
    for score_pos, score_neg in zip(score_on_positives, score_on_negtives):
        # compute indicator such that indicator[j,k] = score_pos[j] < score_pos[k]
        score_pos = score_pos[:pos_lens[usr]]
        score_neg = score_neg[:neg_lens[usr]]
        indicator = np.expand_dims(score_pos, axis=1) > np.expand_dims(score_neg, axis=0)
        auc += np.sum(indicator) / ( len(score_pos) * len(score_neg) )
        usr += 1
    auc = auc / len(score_on_negtives)
    return auc


#----------Building Model--------------------------------------
class simpleBPR(object):
    def __init__(self, num_user, num_item, dim ):
        self.user_mat = tf.Variable(initial_value=tf.random_normal(shape=[num_user, dim]) )
        self.item_mat = tf.Variable(initial_value=tf.random_normal(shape=[num_item, dim]) )
    def get_scores(self, users, items_lst ):
        '''
        args:
            user_ids: 1-D tensor vectors size [buser]
            items_list:tensor of size [buser, bitem]
        return:
            scores of size [buser, bitem]
            scores[u, i] = user_mat[user_ids[u],:] * item_mat[items_list[u,i],:]
        '''
        buser = tf.shape(items_lst)[0]
        bitem = tf.shape(items_lst)[1]
        user_mat = tf.nn.embedding_lookup(self.user_mat, users)     
        item_mat = tf.nn.embedding_lookup(self.item_mat, items_lst) 
        user_mat = tf.expand_dims(user_mat, axis=1) 
        user_mat = tf.expand_dims(user_mat, axis=3) 
        user_mat = tf.tile(user_mat, multiples=[1, bitem, 1, 1]) 
        item_mat = tf.expand_dims(item_mat, axis=2) 
        scores = tf.matmul(item_mat, user_mat)
        scores = tf.reshape(scores, [buser, bitem])
        return scores
    def get_loss(self, score_pos, score_neg, pos_len, neg_len):
        '''
        args:
            score_pos: padded positve items  [buser, max_num_pos]
            score_neg: padded negative items [buser, max_num_neg]
            pos_len: 1-D tensor [buser]
            neg_len: 1-D tensor [buser]
        return:
            pair-wise loss
        '''
        # logits[u, i, j] = score_pos[u,i] - score_neg[u,j]
        buser = tf.shape(score_pos)[0]
        max_num_pos = tf.shape(score_pos)[1]
        max_num_neg = tf.shape(score_neg)[1]
        logits = tf.expand_dims(score_pos, axis=2) - tf.expand_dims(score_neg, axis=1)
        labels = tf.ones(shape=[buser, max_num_pos, max_num_neg])
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels) 
        # mask[u, i, j] = I{i < pos_len(u)} * I{j < neg_len(u)}
        mask_pos = tf.expand_dims(tf.range(max_num_pos), axis=0) \
                    < tf.expand_dims( pos_len, axis=1 ) 
        mask_neg = tf.expand_dims(tf.range(max_num_neg), axis=0) \
                    < tf.expand_dims(neg_len, axis=1) 
        mask_pos = tf.to_float(mask_pos)
        mask_neg = tf.to_float(mask_neg)
        mask = tf.expand_dims(mask_pos, axis=2) * tf.expand_dims(mask_neg, axis=1)
        loss = tf.reduce_sum( mask * losses ) / tf.reduce_sum(mask)
#        loss = tf.reduce_mean(losses)
        return loss

## build Graph
tf.reset_default_graph()
dim = 10
num_user = 1000
num_item = 2000
batch_user = 128
batch_item = 10
lrate = 1e-2
max_iter = 10000
#--------------------------------
bpr = simpleBPR(num_user, num_item, dim)
user_samples   = tf.placeholder(tf.int32, [None])
positive_items  = tf.placeholder(tf.int32, [None, None])
negative_items = tf.placeholder(tf.int32, [None, None])
pos_lens = tf.placeholder(tf.int32,[None])
neg_lens = tf.placeholder(tf.int32,[None])
# get scores
positive_scores = bpr.get_scores(user_samples, positive_items) 
negative_scores = bpr.get_scores(user_samples, negative_items)  
# get loss
loss = bpr.get_loss(positive_scores, negative_scores, pos_lens, neg_lens)
train_op = tf.train.AdamOptimizer(lrate).minimize(loss)
# generate dataset
train_clks, test_clks = generate_datasets(num_user, num_item, num_rel=50, train_ratio=0.7)
# start session
with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    # get current batch of data
    for itr in range( max_iter ):
        users_val, positive_items_val, negative_items_val, pos_lens_val, neg_lens_val = \
            generate_batch(train_clks, batch_user, batch_item, num_item)
        feed_dict = {
                user_samples: users_val,
                positive_items: positive_items_val,
                negative_items: negative_items_val,
                pos_lens: pos_lens_val,
                neg_lens: neg_lens_val
        }
        _, pos, neg, loss_val = sess.run([train_op, positive_scores, negative_scores, loss], feed_dict=feed_dict)
        if itr % 100 == 0:
            # generate testing batch
            users_val, positive_items_val, negative_items_val, pos_lens_val, neg_lens_val = \
                generate_batch(test_clks, num_user, 50, num_item)
            feed_dict = {
                    user_samples: users_val,
                    positive_items: positive_items_val,
                    negative_items: negative_items_val,
                    pos_lens: pos_lens_val,
                    neg_lens: neg_lens_val
            } 
            pos, neg= sess.run([positive_scores, negative_scores], feed_dict=feed_dict)
            auc = compute_auc(pos, neg, pos_lens_val, neg_lens_val)
            print('itr = %d, loss = %f, auc = %f'%(itr, loss_val, auc))
        
