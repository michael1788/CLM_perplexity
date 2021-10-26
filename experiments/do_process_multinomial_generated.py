import os, sys
import time
import configparser
import argparse

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='Process generated data for proba extraction')
parser.add_argument('-c','--configfile', type=str, help='path to config file', required=True)
parser.add_argument('-f','--name_data', type=str, help='Name of the ft file', required=True)
parser.add_argument('-e','--epoch', type=str, help='Which epoch to sample from', required=True)
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
    epoch = args['epoch']
    if len(epoch)==1:
        epoch = f'0{epoch}'
    repeat = args['repeat']
    
    mode = str(config['EXPERIMENTS']['mode'])
    
    if verbose: print('\nSTART PROCESSING')
    ####################################
    
    
    
    
    ####################################
    # paths to save data and to generated smi
    dir_exp = str(config['EXPERIMENTS']['dir'])
    exp_name = configfile.split('/')[-1].replace('.ini','')
    if repeat>0:
        savepath = f'{dir_exp}/{mode}/{exp_name}/{name_data}/generated_data_for_extraction/{repeat}/'
        dir_gen = f'{dir_exp}/{mode}/{exp_name}/{name_data}/generated_data/{repeat}/'
    else:
        savepath = f'{dir_exp}/{mode}/{exp_name}/{name_data}/generated_data_for_extraction/'
        dir_gen = f'{dir_exp}/{mode}/{exp_name}/{name_data}/generated_data/'
    
    os.makedirs(savepath, exist_ok=True)
    ####################################

    
    
    
    
    ####################################
    # start
    min_len = int(config['PROCESSING']['min_len'])
    max_len = int(config['PROCESSING']['max_len'])
    vocab = list(FP.CST_PRIOR.keys())
    temp = float(config['SAMPLING']['temp'])
    top_k = int(config['SAMPLING']['top_k']) 
    top_p = float(config['SAMPLING']['top_p'])
    
    namefile = f'{epoch}_{temp}_{top_k}_{top_p}'
    generated_smi = hp.load_obj(f'{dir_gen}{namefile}.pkl')
    smis_for_extraction = {}
    for i,smi in enumerate(generated_smi):
        smi = smi.replace('G', '')
        smi = smi.replace('E', '')
        
        try:
            tokenized_s, _ = hp.process_smi(smi, min_len, max_len, vocab)
            if tokenized_s:
                smis_for_extraction[i] = tokenized_s
        except:
            pass
                
    hp.save_obj(smis_for_extraction, f'{savepath}{namefile}_for_extraction.pkl')

    end = time.time()
    if verbose: print(f'PROCESSING DONE in {end-start:.2f} seconds')  
    ####################################
        