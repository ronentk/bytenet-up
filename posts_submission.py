# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:04:02 2017

@author: ronent
"""

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import os
import json

user = ''
api_tok = ''

NUM_PREDS = 3
DAY = 7

def select_final_submission(all_pred_dict,tw_dict,pred_types=['ft_max','all_max','lrg_max','ft_sample','all_sample']):
    final_sub_dict = {}    
    for i,pred_key in enumerate(all_preds.keys()):
        print "Collecting tweet, %d/%d " % (i,len(all_preds))
        pred_df = pd.DataFrame(all_preds[pred_key])
        final_sub_dict[pred_key] = []
        sel_preds = []
        while len(sel_preds) < NUM_PREDS:
            for ptype in pred_types:
                for p in list(pred_df[pred_df['pred_src'] == ptype]['pred']):
                    if len(sel_preds) < NUM_PREDS:
                        sel_preds.append(p)
                    else:
                        break


        final_sub_dict[pred_key] = [pad_with_default(pred,tw_dict,pred_key) for pred in sel_preds]
        
    return final_sub_dict
    
def get_targ_num(tw_dict, u_id):
    return len([x['value'] for x in tw_dict[u_id]['entitiesShortened'] if x['type'] == 'letter'])
def get_default_guess(tw_dict, u_id):
    return [x['value'] for x in tw_dict[u_id]['entitiesShortened'] if x['type'] == 'letter']

def pad_with_default(pred,tw_dict, u_id):
    def_pred = get_default_guess(tw_dict, u_id)
    if len(def_pred) < len(pred):
        return pred[:len(def_pred)]
    pad_pred = pred + def_pred[len(pred):]
    return pad_pred
    
def pred_to_list(pred):
    if pred == "":
        return 'I'
    pred_seq = [w for w in pred.split(' ') if ((w!='') and (w[0].isalpha()))]
    return pred_seq
        
def gather_user_preds(user_path,pred_src='ft'):
    if not os.path.exists(user_path):
        return {}
    all_batches = [ x for x in os.listdir(user_path) if x.endswith('.csv')]
    pred_dict = {}
    
    
    for i,batch_file in enumerate(all_batches):
        batch_file_path = os.path.join(user_path,batch_file)
        batch = pd.read_csv(batch_file_path,sep='{').dropna()
        
        for i,row in batch.iterrows():
            pred = pred_to_list(row['pred'])
            tw_id = row['id']
            if not pred_dict.has_key(tw_id):
                pred_dict[tw_id] = [{'pred': pred, 'pred_src': pred_src + '_' + row['pred_type']}]
            else:
                pred_dict[tw_id].append({'pred': pred, 'pred_src': pred_src + '_' + row['pred_type']})
        
    return pred_dict

resp_jsons = []
word_count = 0
correct_count = 0
fted = 0
all_users_path = '../scoring_%d/tmlc1-scoring-00%d.data.src.csv' % (DAY,DAY)
tests_base_dir = 'test_users_sc%d' % (DAY)
full_test_id = 'train'
full_test_lrg_id_ = 'ft200'
test_df = pd.read_csv(all_users_path,sep='{')
 
 
 
with open('..//scoring_%d/tmlc1-scoring-00%d.json' % (DAY,DAY),'rb') as f:
    sc1 = json.load(f) 
ids = {x['id']:x for x in sc1}

 
all_preds = gather_user_preds(os.path.join(tests_base_dir,full_test_id),pred_src='all')
all_preds_lrg = gather_user_preds(os.path.join(tests_base_dir,full_test_lrg_id_),pred_src='lrg')
for i,row in test_df.iterrows():

    print "Collecting user %s , %d/%d " % (row['user'],i,test_df.shape[0])
    user_path = os.path.join(tests_base_dir,row['user'])
    
    user_preds = gather_user_preds(user_path,'ft')
    
        
    if not user_preds:
        print "No finetuning avaliable"
        continue
    else:
        fted += 1
        for tw_id in user_preds.keys():
            if len(all_preds[tw_id]) == 3:
                all_preds[tw_id] += user_preds[tw_id]
                all_preds[tw_id] += all_preds_lrg[tw_id]
                
            
            
final_dict = select_final_submission(all_preds,ids)



with open('out_%d.sub','wb') as f_out:
    json.dump(final_dict,f_out)
resp = requests.post("http://challenges.tmlc1.unpossib.ly/api/submissions", auth=HTTPBasicAuth(user, api_tok), json=json.loads(json.dumps(final_dict)))
res_json = json.loads(resp.text)
with  open('out_%d'%(DAY),'wb') as f:
    json.dump(res_json,f)


