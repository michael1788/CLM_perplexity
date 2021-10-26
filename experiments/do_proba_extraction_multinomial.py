import os, sys
import time
import argparse
import configparser
import ast

from keras.models import load_model

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='Proba extraction on the multinomial generated SMILES')
parser.add_argument('-c','--configfile', type=str, help='path to config file', required=True)
parser.add_argument('-f','--name_data', type=str, help='Name of the ft file', required=True)
parser.add_argument('-e','--epoch', type=str, help='Which epoch to sample from', required=True)
parser.add_argument('-r','--repeat', type=int, help='Number of repeats', required=True)
parser.add_argument('--from_0', dest='epoch_0', help='Extract proba. from pretrained model', action='store_true')
parser.add_argument('--from_normal', dest='epoch_0', help='Extract proba. from sampling model', action='store_false')
parser.set_defaults(epoch_0=False)

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
    
    mode = str(config['EXPERIMENTS']['mode'])
    
    if verbose: print('\nSTART PROBA EXTRACTION')
    ####################################
    
    
    
    ####################################
    # path to save data and to checkpoints
    dir_exp = str(config['EXPERIMENTS']['dir'])
    exp_name = configfile.split('/')[-1].replace('.ini','')
    if repeat>0:
        if epoch_0:
            # we extract the proba with the pretrained model (epoch 0)
            epoch_ckpts = 0
            save_path = f'{dir_exp}/{mode}/{exp_name}/{name_data}/proba_extraction_e0/{repeat}/'
            path_model = str(config['FINETUNING']['LM'])
        else:
            save_path = f'{dir_exp}/{mode}/{exp_name}/{name_data}/proba_extraction/{repeat}/'
            dir_ckpts = f'{dir_exp}/{mode}/{exp_name}/{name_data}/models/{repeat}/'
            path_model = f'{dir_ckpts}{epoch}.h5'
    else:
        if epoch_0:
            # we extract the proba with the pretrained model (epoch 0)
            save_path = f'{dir_exp}/{mode}/{exp_name}/{name_data}/proba_extraction_e0/'
            epoch_ckpts = 0
            path_model = str(config['FINETUNING']['LM'])
        else:
            save_path = f'{dir_exp}/{mode}/{exp_name}/{name_data}/proba_extraction/'
            dir_ckpts = f'{dir_exp}/{mode}/{exp_name}/{name_data}/models/'
            path_model = f'{dir_ckpts}{epoch}.h5'
    os.makedirs(save_path, exist_ok=True)
    ####################################
    
    
    
    
    ####################################
    # Parameters to sample novo smiles
    temp = float(config['SAMPLING']['temp'])
    top_k = int(config['SAMPLING']['top_k']) 
    top_p = float(config['SAMPLING']['top_p'])
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    indices_token = FP.INDICES_TOKEN
    token_indices = FP.TOKEN_INDICES
    n_chars = len(indices_token)
    ####################################
    
    
    
    ####################################
    # get the data
    if repeat>0:
        dir_data = f'{dir_exp}/{mode}/{exp_name}/{name_data}/generated_data_for_extraction/{repeat}/'
    else:
        dir_data = f'{dir_exp}/{mode}/{exp_name}/{name_data}/generated_data_for_extraction/'
    namefile = f'{epoch}_{temp}_{top_k}_{top_p}'
    
    data_for_extr = hp.load_obj(f'{dir_data}{namefile}_for_extraction.pkl')
    ####################################
    
    
    
    
       
    ####################################
    # start probabilites extraction
    model = load_model(path_model)
    
    # extract the proba
    all_proba = {}
    for idx,tsmi in data_for_extr.items():
        forced_proba = hp.forced_sample(model, 
                                        tsmi,
                                        temp, 
                                        start_char, end_char, 
                                        indices_token, token_indices, n_chars)
        all_proba[''.join(tsmi)] = forced_proba
    
    hp.save_obj(all_proba, f'{save_path}{namefile}')
            
        
    end = time.time()
    if verbose: print(f'EXTRACTION DONE in {end-start:.2f} seconds')  
    ####################################
        