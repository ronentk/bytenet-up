# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
from model import *
import tw_data
import pandas as pd
from tw_data import TwitData
import pickle
import os
import argparse
from os.path import join

RELEASE = True
NUM_PREDS = 3

def str_indices(indices,i2b):
    str_ = ''
    for ch in indices:
        if ch > 1:
            str_ += unichr(i2b[ch])
        elif ch == 1:  # <EOS>
            break
    return str_



"""
x in batch_size x category
"""
def stable_normalize_logspace_vec(x_in):
    c = np.max(x_in,axis=1)
    c_rep = np.tile(c,reps=(x_in.shape[1],1)).transpose()
    x_shifted = x_in - c_rep
    exp_x_shifted = np.exp(x_shifted)
    row_sum = np.sum(exp_x_shifted, axis=1)
    row_sum_norm = np.tile(row_sum,reps=(x_in.shape[1],1)).transpose()
    return np.divide(exp_x_shifted, row_sum_norm)
    


if __name__ == "__main__":
    
    # set log level to debug
    tf.sg_verbosity(10)
    
    #
    # hyper parameters
    #
    
    batch_size = 10
    
    description = ('')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--user_id",
        help="user id",
        type=str
    )
    parser.add_argument(
        "day",
        help="day",
        type=int
    )
    
    args = parser.parse_args()
    user = args.user_id
    day = args.day
#    user = '1njv06'
    
    ckpt_dir = join('asset',user)
    out_dir = 'test_users_sc%d' %(day)
    trg_dir = user
    try:
        os.makedirs(os.path.join(out_dir,trg_dir))
    except:
        pass

    
    user_df = pd.read_csv('../users_%d.csv' % (day))
    user_file = list(user_df[user_df['id'] == user]['test_%d'%(day)])[0]
    if user_file == 'na':
        print 'No test for this user available. Exiting..'
        exit(-1)
    test_file = '../' + user_file

    with open('endec100k.pkl', 'rb') as f_in:
        b2i = pickle.load(f_in)
        i2b = pickle.load(f_in)
        
    max_len = 150
    min_len = 0
    
    test_sources,users,ids = tw_data.load_corpus_dict_test(path=test_file, 
                                                            b2i=b2i,min_len=min_len,max_len=max_len)
                                                            
                                                        
    batch_size = min(batch_size,len(test_sources))
            
    # place holders
    x = tf.placeholder(dtype=tf.sg_intx, shape=(batch_size, max_len))
    y_in = tf.placeholder(dtype=tf.sg_intx, shape=(batch_size, max_len))
    y = tf.placeholder(dtype=tf.sg_intx, shape=(batch_size, max_len))
    # vocabulary size
    voca_size = len(b2i)
    
    # make embedding matrix for source and target
    emb_x = tf.sg_emb(name='emb_x', voca_size=voca_size, dim=latent_dim)
    emb_y = tf.sg_emb(name='emb_y', voca_size=voca_size, dim=latent_dim)
    
    # latent from embed table
    z_x = x.sg_lookup(emb=emb_x)
    z_y = y_in.sg_lookup(emb=emb_y)
    
    
    # encode graph ( atrous convolution )
    enc = encode(z_x)
    
    # concat merge target source
    enc = enc.sg_concat(target=z_y)
    
    # decode graph ( causal convolution )
    dec = decode(enc, voca_size)
    
    # greedy search policy
    label = dec.sg_argmax()
    
    loss = dec.sg_ce(target=y, mask=True)
    

    
    
     # run graph for translating
    with tf.Session() as sess:
        # init session vars
        tf.sg_init(sess)
    
        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        
        _step = sess.run(tf.sg_global_step())
        
        for fname in [test_file]:
            
            

            assert(len(test_sources)==len(users) and len(users)==len(ids))
            test_targets = test_sources
            
            
            
            
            num_its = len(test_sources) // batch_size
           
            for t in range(num_its):
                res_list = []
                lock_name = '%s_%s_%d_%d.lck' % (trg_dir,user,_step,t)
                csv_name = '%s_%s_%d_%d.csv' % (trg_dir,user,_step,t)
                lock_path = os.path.join(out_dir,trg_dir,lock_name)
                csv_path = os.path.join(out_dir,trg_dir,csv_name)
                if not os.path.exists(lock_path): 
                    print "Starting batch %d/%d" % (t,num_its)
                    if RELEASE:
                        with open(lock_path,'wb') as f:
                            f.write('tt')
                    abs_batch = t*batch_size
                    batch = test_sources[abs_batch:abs_batch+batch_size]
                    for k in range(NUM_PREDS):
                        pred_prev = np.zeros((batch_size, max_len)).astype(np.int32)
                        pred = np.zeros((batch_size, max_len)).astype(np.int32)
                        
                        for j in range(max_len):
                            print ("\rIteration %d: Predicting char %d/%d " % (t,j,max_len)),
                            # predict character
                            
                            dec_value, out, loss_val = sess.run([dec,label,loss], {x: batch, y_in: pred_prev , y: batch})
                            probs = stable_normalize_logspace_vec(dec_value[:,j,:])
                            sampled = np.zeros(batch_size)
                            for c in range(batch_size):
                                sampled[c] = np.random.choice(voca_size, 1, p=probs[c,:])[0]
                            # update character sequence
                            if k == 1: # first prediction is the most probable estimate
                                pred_chars = out[:, j]
                                pred_type = 'max'
                            else:
                                pred_chars = sampled
                                pred_type = 'sample'
                                
                            if j < max_len - 1:
                                pred_prev[:, j + 1] = pred_chars
                            pred[:, j] = pred_chars
                        
                        for p in range(pred.shape[0]):
                            abs_id = t*batch_size + p
                            pred_str = str_indices(pred[p,:],i2b)
                            res_list.append({'id': ids[abs_id], 'user': users[abs_id] ,
                            'pred': pred_str, 'pred_type': pred_type})
                    
                    res_df = pd.DataFrame(res_list)
                    if RELEASE:
                        res_df.to_csv(csv_path,sep='{',columns=['id','user','pred','pred_type'])
    
                else:
                    continue
                

