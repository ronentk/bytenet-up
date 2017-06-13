import sugartensor as tf
from model import *
import pickle
from tw_data import TwitData
import time
from tqdm import tqdm
from sugartensor import sg_optim
import numpy as np
import argparse
from os.path import join
import os
import shutil
import pandas as pd


ckpt_files = ['model.ckpt-619599.meta',
              'model.ckpt-619599.index',
              'model.ckpt-619599.data-00000-of-00001',
              'checkpoint',
              'graph.pbtxt'
                

            ]
            
def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)  
        
def copy_checkpoint(ckpt_dir,dst_dir):
    for fname in ckpt_files:
        shutil.copy(join(ckpt_dir,fname),join(dst_dir,fname))
        
# console logging function
def console_log(sess_):
    if epoch >= 0:
        tf.sg_info('\tEpoch[%03d:gs=%d] - loss = %s' %
                   (epoch, sess_.run(tf.sg_global_step()),
                    ('NA' if loss_val is None else '%8.6f' % loss_val)))



DATA_DIR = '../data_general'
CKPT_DIR = 'asset/train'
LOSS_THRESHOLD = 0.03
# hyper parameters
#
MAX_EPOCHS = 30
batch_size = 8    # batch size
lr = 0.0001


if __name__ == "__main__":
    
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

    user_df = pd.read_csv('../users_%d.csv' % (day))
    user_file = list(user_df[user_df['id'] == user]['train'])[0]
    if user_file == 'na':
        print 'No train for this user available. Exiting..'
        exit(-1)
    train_file = '../' + user_file
    
    save_dir = join('asset',user)
    try:
        os.makedirs(save_dir)
    except:
        pass

    print "Copying checkpoint.."
    copy_checkpoint(CKPT_DIR,save_dir)  
    
    with open('endec100k.pkl', 'rb') as f_in:
        b2i = pickle.load(f_in)
        i2b = pickle.load(f_in)


    # set log level to debug
    tf.sg_verbosity(1)
    
    # default training options
    opt = tf.sg_opt(lr=lr,
                     save_dir=save_dir,
                     max_ep=MAX_EPOCHS,
                     save_interval=10, log_interval=60,
                     eval_metric=[],
                     max_keep=1, keep_interval=1,
                     tqdm=True)
    
    # default training options
    opt += tf.sg_opt(optim='MaxProp', 
                     beta1=0.9, 
                     beta2=0.99, category='')

    
    
    #
    # inputs
    #
    
    data = TwitData(batch_size=batch_size,b2i=b2i,
     path=train_file)

    # summary writer
    log_dir = opt.save_dir + '/run-%02d%02d-%02d%02d' % tuple(time.localtime(time.time()))[1:5]
    
    # source, target sentence
    x, y = data.source, data.target
    # shift target for training source
    y_in = tf.concat([tf.zeros((batch_size, 1), tf.sg_intx), y[:, :-1]], axis=1)
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
    
    # cross entropy loss with logit and mask
    loss = dec.sg_ce(target=y, mask=True)
    
    #opt += tf.sg_opt(ep_size=data.num_batch)
    opt += tf.sg_opt(ep_size=100)
    # train
    opt += tf.sg_opt(loss=loss)
    
    # get optimizer
    train_op = sg_optim(opt.loss, optim=opt.optim, lr=lr,
                        beta1=opt.beta1, beta2=opt.beta2, category=opt.category)
    
    # checkpoint saver
    saver = tf.train.Saver(max_to_keep=opt.max_keep,
                           keep_checkpoint_every_n_hours=opt.keep_interval)
    
    
    # create supervisor
    sv = tf.train.Supervisor(logdir=opt.save_dir,
                             saver=saver,
                             save_model_secs=opt.save_interval,
                             summary_writer=None,
                             save_summaries_secs=opt.log_interval,
                             global_step=tf.sg_global_step(),
                             local_init_op=tf.sg_phase().assign(True))
    
    
    
    # training epoch and loss
    epoch, loss_val = -1, None
    
    
    # training epoch and loss
    epoch, loss_val = -1, None
    
    
    # create session
    print "Starting session..."
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # console logging loop
        if not opt.tqdm:
            sv.loop(opt.log_interval, console_log, args=(sess, ))
    
        # get start epoch
        _step = sess.run(tf.sg_global_step())
        ep = 1
    
        # check if already finished
        if ep <= opt.max_ep:
    
            # logging
            tf.sg_info('Training started from epoch[%03d]-step[%d].' % (ep, _step))
    
            # epoch loop
            for ep in range(ep, opt.max_ep + 1):
    
                # update epoch info
                start_step = 1
                epoch = ep
    
                # create progressbar iterator
                if opt.tqdm:
                    iterator = tqdm(range(start_step, opt.ep_size), total=opt.ep_size, initial=start_step,
                                    desc='train', ncols=70, unit='b', leave=False)
                else:
                    iterator = range(start_step, opt.ep_size)
    
                # batch loop
                for _ in iterator:
    
                    # exit loop
                    if sv.should_stop():
                        break
    
                    
                    loss_ = opt.loss
                    # call train function
                    batch_loss = sess.run([loss_, train_op])[0]
    
                    # loss history update
                    if batch_loss is not None and \
                            not np.isnan(batch_loss.all()) and not np.isinf(batch_loss.all()):
                        if loss_val is None:
                            loss_val = np.mean(batch_loss)
                        else:
                            loss_val = loss_val * 0.9 + np.mean(batch_loss) * 0.1
                

                console_log(sess)
                
                if loss_val < LOSS_THRESHOLD:
                    print "Loss below %f, breaking" % (LOSS_THRESHOLD)
                    break
    
            # save last version
            saver.save(sess, opt.save_dir + '/model.ckpt', global_step=sess.run(tf.sg_global_step()))
            
            # logging
            tf.sg_info('Training finished at epoch[%d]-step[%d].' % (ep, sess.run(tf.sg_global_step())))
        else:
            tf.sg_info('Training already finished at epoch[%d]-step[%d].' %
                       (ep - 1, sess.run(tf.sg_global_step())))
