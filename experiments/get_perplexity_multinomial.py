import os, sys
import time
import argparse
import configparser
import ast
import csv

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='Get the perplexity on multinomial generated SMILES')
parser.add_argument('-c','--configfile', type=str, help='path to config file', required=True)
parser.add_argument('-f','--name_data', type=str, help='Name of the ft file', required=True)
parser.add_argument('-e','--epoch', type=str, help='Which epoch to sample from', required=True)
parser.add_argument('-r','--repeat', type=int, help='Number of repeats', required=True)
parser.add_argument('--from_0', dest='epoch_0', help='Compute perplexity from pretrained model', action='store_true')
parser.add_argument('--from_epoch', dest='epoch_0', help='Compute perplexity from sampling model', action='store_false')
parser.set_defaults(epoch_0=False)
parser.add_argument('--only_novo', dest='de_novo', help='Consider only de novo molecules', action='store_true')
parser.add_argument('--all', dest='de_novo', help='Consider all generated molecules', action='store_false')
parser.set_defaults(de_novo=True)


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
    epoch = args['epoch']
    if len(epoch)==1:
        epoch = f'0{epoch}'
    repeat = args['repeat']
    epoch_0 = args['epoch_0']
    de_novo = args['de_novo']
    if de_novo:
        print(f'Note: will only consider the sampled de novo molecules')
    
    mode = str(config['EXPERIMENTS']['mode'])
    
    if verbose: print('\nSTART PERPLEXITY')
    ####################################
    
    
    
    
    ####################################
    # paths to save data and to generated smi
    dir_exp = str(config['EXPERIMENTS']['dir'])
    exp_name = configfile.split('/')[-1].replace('.ini','')
    if repeat>0:
        if epoch_0:
            savepath = f'{dir_exp}/{mode}/{exp_name}/{name_data}/perplexity_e0/{repeat}/'
            dir_data = f'{dir_exp}/{mode}/{exp_name}/{name_data}/proba_extraction_e0/{repeat}/'
        else:
            savepath = f'{dir_exp}/{mode}/{exp_name}/{name_data}/perplexity/{repeat}/'
            dir_data = f'{dir_exp}/{mode}/{exp_name}/{name_data}/proba_extraction/{repeat}/'
    else:
        if epoch_0:
            savepath = f'{dir_exp}/{mode}/{exp_name}/{name_data}/perplexity_e0/'
            dir_data = f'{dir_exp}/{mode}/{exp_name}/{name_data}/proba_extraction_e0/'
        else:
            savepath = f'{dir_exp}/{mode}/{exp_name}/{name_data}/perplexity/'
            dir_data = f'{dir_exp}/{mode}/{exp_name}/{name_data}/proba_extraction/'
    os.makedirs(savepath, exist_ok=True)
    ####################################

    
    
    
    ####################################
    # get the perplexity
    if de_novo:
        # get back the pretraining data
        pretraining_dir = str(config['DATA']['pretraining_dir'])
        min_len = int(config['PROCESSING']['min_len'])
        max_len = int(config['PROCESSING']['max_len'])
        aug = int(config['AUGMENTATION']['fold'])
        path_pretraining = f'{pretraining_dir}{min_len}_{max_len}_x{aug}.txt'
        pretraining_data = hp.read_with_pd(path_pretraining)
        
        # get back the fine-tuning data
        ft_dir = str(config['DATA']['dir'])
        min_len = int(config['PROCESSING']['min_len'])
        max_len = int(config['PROCESSING']['max_len'])
        aug = int(config['AUGMENTATION']['fold'])
        path_ft = f'{ft_dir}{name_data}/{min_len}_{max_len}_x{aug}/{min_len}_{max_len}_x{aug}.txt'
        ft_data = hp.read_with_pd(path_ft)
        
        # merge, and put in a dict for faster search
        # note: we just care about the keys
        all_data = list(set(pretraining_data + ft_data))
        d_data = {x:0 for x in all_data}
    
    d_epoch_score = {}
    for fn in os.listdir(dir_data):
        if fn.endswith('.pkl'):
            epoch = int(fn.split('_')[0])
            d_smi_probas = hp.load_obj(f'{dir_data}{fn}.pkl')
            
            current_score = {}
            for smi,all_probas in d_smi_probas.items():
                if de_novo and hp.is_novo(smi, d_data):
                    current_score[smi] = hp.perplexity_w_prior(all_probas[:-1], 
                                                               smi, 
                                                               FP.CST_PRIOR,
                                                               remove_end=False)
                    d_epoch_score[epoch] = current_score
                elif not de_novo:
                    current_score[smi] = hp.perplexity_w_prior(all_probas[:-1], 
                                                               smi, 
                                                               FP.CST_PRIOR,
                                                               remove_end=False)
                    d_epoch_score[epoch] = current_score
    
    hp.save_obj(d_epoch_score, f'{savepath}d_epoch_score.pkl')
    
    # we also save a csv file for convenience
    for epoch,d_score in d_epoch_score.items():
        sorted_score = sorted(d_score.items(), key=lambda kv: kv[1])
        with open(f'{savepath}epoch{epoch}_score.csv','wt') as f:
                w = csv.writer(f, delimiter=',')
                w.writerow(['SMILES', 'perplexity'])
                for entry in sorted_score:
                    smi = entry[0]
                    perplexity = entry[1]
                    w.writerow([smi, perplexity])
                    
    
    end = time.time()
    if verbose: print(f'PERPLEXITY DONE in {end-start:.2f} seconds')  
    ####################################
        