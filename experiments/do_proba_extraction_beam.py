import os, sys
import time
import argparse
import configparser
import ast

from keras.models import load_model

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='Get the probabilities of the beam generated SMILES')
parser.add_argument('-c','--configfile', type=str, help='path to config file', required=True)
parser.add_argument('-f','--name_data', type=str, help='Name of the ft file', required=True)
parser.add_argument('-e','--epoch', type=str, help='Which epoch to sample from', required=True)
parser.add_argument('-r','--repeat', type=int, help='Number of repeats', required=True)
parser.add_argument('-k','--width', type=int, help='Beam width (k)', required=True)


if __name__ == '__main__':
    
    start = time.time()
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
        
    verbose = True
    name_data = args['name_data']
    e_model = int(args['epoch'])
    configfile = args['configfile']
    repeat = args['repeat']
    width = args['width']
    if verbose: print(f'\nBeam width: {width}')
    
    config = configparser.ConfigParser()
    config.read(configfile)
    mode = str(config['EXPERIMENTS']['mode'])
    
    if verbose: print('\nSTART BEAM PROBA EXTRACTION')
    ####################################

    
    
    
    ####################################
    # paths to save data and to checkpoints
    dir_exp = str(config['EXPERIMENTS']['dir'])
    exp_name = configfile.split('/')[-1].replace('.ini','')
    if repeat>0:
        save_path = f'{dir_exp}/{mode}/{exp_name}/{name_data}/proba_extraction_beam/{width}/{repeat}/'
        data_dir = f'{dir_exp}/{mode}/{exp_name}/{name_data}/beam_data_for_extraction/{width}/{repeat}/'
        dir_ckpts = f'{dir_exp}/{mode}/{exp_name}/{name_data}/models/{repeat}/'
    else:
        save_path = f'{dir_exp}/{mode}/{exp_name}/{name_data}/proba_extraction_beam/{width}/'
        data_dir = f'{dir_exp}/{mode}/{exp_name}/{name_data}/beam_data_for_extraction/{width}/'
        dir_ckpts = f'{dir_exp}/{mode}/{exp_name}/{name_data}/models/'
    
    os.makedirs(save_path, exist_ok=True)
    ####################################
    
    
    
    ####################################
    # Parameters
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    indices_token = FP.INDICES_TOKEN
    token_indices = FP.TOKEN_INDICES
    n_chars = len(token_indices)
    temp = float(config['SAMPLING']['temp'])
    ####################################
    
    
    
    ####################################
    # start probabilites extraction
    for filename in os.listdir(dir_ckpts):
        if filename.endswith('.h5'):           
            epoch = filename.split('/')[-1].replace('.h5', '')
            if int(epoch)==int(e_model):
                # get the model
                if e_model<10:
                    path_model = f'{dir_ckpts}0{e_model}.h5'
                    path_data = f'{data_dir}0{e_model}_all_smis_for_extraction.pkl'
                    savename = f'{save_path}0{e_model}_proba_extraction.pkl'
                else:
                    path_model = f'{dir_ckpts}{e_model}.h5'
                    path_data = f'{data_dir}{e_model}_all_smis_for_extraction.pkl'
                    savename = f'{save_path}{e_model}_proba_extraction.pkl'
                    
                model = load_model(path_model)
                data_for_extr = hp.load_obj(path_data)
                
                # extract the proba
                all_proba = {}
                for idx,tsmi in data_for_extr.items():
                    if tsmi:
                        forced_proba = hp.forced_sample(model, 
                                                        tsmi,
                                                        temp, 
                                                        start_char, end_char, 
                                                        indices_token, token_indices,
                                                        n_chars)
                        all_proba[''.join(tsmi)] = forced_proba
                
                hp.save_obj(all_proba, savename)
            
        
    end = time.time()
    if verbose: print(f'BEAM EXTRACTION DONE in {end-start:.2f} seconds')  
    ####################################
        