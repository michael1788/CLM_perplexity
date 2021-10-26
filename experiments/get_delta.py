import os, sys
import time
import argparse
import configparser
import ast
import numpy as np
import csv

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='Get the delta')
parser.add_argument('-c','--configfile', type=str, help='path to config file', required=True)
parser.add_argument('-f','--name_data', type=str, help='Name of the ft file', required=True)
parser.add_argument('-rs','--range_start', type=int, help='Start of the range', required=True)
parser.add_argument('-st','--step', type=int, help='Step of the range', required=True)
parser.add_argument('-re','--range_end', type=int, help='End of the range', required=True)
parser.add_argument('-r','--repeat', type=int, help='Number of repeats', required=True)


if __name__ == '__main__':
    
    start = time.time()
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
        
    verbose = True
    configfile = args['configfile']
    config = configparser.ConfigParser()
    config.read(configfile)
    name_data = args['name_data']
    range_start = args['range_start']
    step = args['step']
    range_end = args['range_end']
    repeat = args['repeat']
    
    mode = str(config['EXPERIMENTS']['mode'])
    
    if verbose: print('\nSTART DELTA EXTRACTION')
    ####################################
    
    
    
    
    ####################################
    # path to save data and to checkpoints
    dir_exp = str(config['EXPERIMENTS']['dir'])
    exp_name = configfile.split('/')[-1].replace('.ini','')
    if repeat>0:
        save_path = f'{dir_exp}/{mode}/{exp_name}/{name_data}/delta/{repeat}/'
        dir_data = f'{dir_exp}/{mode}/{exp_name}/{name_data}/proba_extraction/{repeat}/'
        dir_data_e0 = f'{dir_exp}/{mode}/{exp_name}/{name_data}/proba_extraction_e0/{repeat}/'
    else:
        save_path = f'{dir_exp}/{mode}/{exp_name}/{name_data}/delta/'
        dir_data = f'{dir_exp}/{mode}/{exp_name}/{name_data}/proba_extraction/'
        dir_data_e0 = f'{dir_exp}/{mode}/{exp_name}/{name_data}/proba_extraction_e0/'
    os.makedirs(save_path, exist_ok=True)
    ####################################
    
    
    
       
    ####################################
    # start computing delta
    temp = float(config['SAMPLING']['temp'])
    top_k = int(config['SAMPLING']['top_k']) 
    top_p = float(config['SAMPLING']['top_p'])
    
    for epoch in range(range_start,range_end+1,step):
        # save the SMILES delta
        d_epoch_delta = {}
        # save the SMILES rank at its
        # sampled epoch
        d_epoch_rank = {}
        
        if epoch<10:
            namefile = f'0{epoch}_{temp}_{top_k}_{top_p}'
        else:
            namefile = f'{epoch}_{temp}_{top_k}_{top_p}'
        
        path_origin = f'{dir_data}{namefile}.pkl'
        path_e0 = f'{dir_data_e0}{namefile}.pkl'
        
        data_origin = hp.load_obj(path_origin)
        dict_pp_origin = hp.get_dict_pp(data_origin, FP.CST_PRIOR)
        idx_sorted_origin, smis_origin, vals_origin = hp.get_idx_sorted(dict_pp_origin)
        
        data_e0 = hp.load_obj(path_e0)
        dict_pp_e0 = hp.get_dict_pp(data_e0, FP.CST_PRIOR)
        idx_sorted_0, smis_0, vals_0 = hp.get_idx_sorted(dict_pp_e0)
        
        # use dict to be faster
        d_smis_0 = dict(enumerate(smis_0))
        inv_d_smis_0 = {v:k for k,v in d_smis_0.items()}
        d_idx_sorted_0 = dict(enumerate(idx_sorted_0))
        inv_d_idx_sorted_0 = {v:k for k,v in d_idx_sorted_0.items()}
        
        for i,idx_ori in enumerate(idx_sorted_origin):
            smi = smis_origin[idx_ori]
            rank_ori = i+1
            idx_0 = inv_d_smis_0[smi]
            rank_0 = inv_d_idx_sorted_0[idx_0]+1
            
            # compute the delta rank
            delta = rank_0-rank_ori
            
            d_epoch_delta[smi] = delta
            d_epoch_rank[smi] = rank_ori
        
        # order SMILES by their rank
        sorted_rank = sorted(d_epoch_rank.items(), key=lambda kv: kv[1])
        
        # save a csv files
        with open(f'{save_path}epoch{epoch}_delta_rank.csv','wt') as f:
            w = csv.writer(f, delimiter=',')
            w.writerow(['SMILES', 'rank', 'delta'])
            for entry in sorted_rank:
                smi = entry[0]
                rank = entry[1]
                smi_delta = d_epoch_delta[smi]
                w.writerow([smi, rank, smi_delta])
            
    end = time.time()
    if verbose: print(f'DELTA DONE in {end-start:.2f} seconds')  
    ####################################
        