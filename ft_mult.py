#!/usr/bin/env python


import pickle
import os, subprocess, time
import sys
from os.path import join
import json
import numpy as np
import pandas as pd
import argparse


DAY = 7

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TRAIN_SCRIPT = os.path.abspath(os.path.join('fine_tune_user.py'))
TEST_SCRIPT = os.path.abspath(os.path.join('test_predict_dict_user.py'))
USERS_BASE_DIR = 'asset'
USERS_FILE = '../users_%d.csv' % (DAY)
USER_TESTS_BASE_DIR = 'test_users_sc%d'%(DAY)

def finished_run(target_dir,target_steps):
    
    fnames = os.listdir(target_dir)
    for fn in fnames:
        if str(target_steps) in fn:
            return True
    return False
    
def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)   
        
def train(user_dir, user_id):
  
    cmd = [sys.executable, TRAIN_SCRIPT, '%d'%(DAY), '--user_id=%s' % (user_id)]
    cmd = ' '.join(cmd)
    print cmd
    results = subprocess.check_output(cmd, shell=True, cwd=SCRIPT_DIR)
    with open(os.path.join(user_dir,'train.log'), 'wb') as f:
        f.write(results)

def test(user_dir, user_id):
  
    cmd = [sys.executable, TEST_SCRIPT, '%d'%(DAY),'--user_id=%s' % (user_id)]
    cmd = ' '.join(cmd)
    print cmd
    results = subprocess.check_output(cmd, shell=True, cwd=SCRIPT_DIR)
    with open(os.path.join(user_dir,'test.log'), 'wb') as f:
        f.write(results)
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Runs training and testing for configurations specified')
    parser.add_argument('--users_file', type=str, default=USERS_FILE, help='Path to json file containing'
                        ' users to run finetuning on. Default: ' + str(USERS_FILE) + '.')
    
    args = parser.parse_args()
    users_fpath = args.users_file

    users_df = pd.read_csv(users_fpath)
    with open('../ubt/users_ord_%d.pkl' % (DAY) , 'rb') as f:
        users_ord = pickle.load(f)
    user_order = [u for (u,e) in users_ord]
        
    
    try:
        for i,user in enumerate(user_order):
            print "Starting finetuning on user %d/%d : %s" % (i,len(user_order),user)
            user_dir = join(USERS_BASE_DIR,user)
            user_test_dir = join(USER_TESTS_BASE_DIR,user)
            flag_file = join(user_dir,'final.flg')
            test_flag_file = join(user_test_dir,'final.flg')
            if not os.path.exists(user_dir):
                train(user_dir,user)
                touch(flag_file)
                print "Finished finetuning on user ", user
            else:
                print "Skipping..."
            if os.path.exists(flag_file):
                if not os.path.exists(user_test_dir):
                    print "Starting testing finetuned model on user ", user
                    test(user_test_dir,user)
                    touch(test_flag_file)
                else:
                    print "Testing already started, skipping..."
                
     
    except KeyboardInterrupt:
        time.sleep(1)
        print 'Finetuning interrupted!'
