import os, sys
import time
import configparser
import argparse

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='Process beam generated data for proba extraction')
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
    configfile = args['configfile']
    config = configparser.ConfigParser()
    config.read(configfile)
    name_data = args['name_data']
    epoch = args['epoch']
    if len(epoch)==1:
        epoch = f'0{epoch}'
    repeat = args['repeat']
    width = args['width']
    if verbose: print(f'\nBeam width: {width}')
    
    mode = str(config['EXPERIMENTS']['mode'])
    
    if verbose: print('\nSTART PROCESSING')
    ####################################
    
    
    
    
    ####################################
    # paths to save data and to generated smi
    dir_exp = str(config['EXPERIMENTS']['dir'])
    exp_name = configfile.split('/')[-1].replace('.ini','')
    if repeat>0:
        savepath = f'{dir_exp}/{mode}/{exp_name}/{name_data}/beam_data_for_extraction/{width}/{repeat}/'
        dir_gen = f'{dir_exp}/{mode}/{exp_name}/{name_data}/beam_sampling/{width}/{repeat}/'
    else:
        savepath = f'{dir_exp}/{mode}/{exp_name}/{name_data}/beam_data_for_extraction/{width}/'
        dir_gen = f'{dir_exp}/{mode}/{exp_name}/{name_data}/beam_sampling/{width}/'
    
    os.makedirs(savepath, exist_ok=True)
    ####################################

    
    
    
    ####################################
    # process
    min_len = int(config['PROCESSING']['min_len'])
    max_len = int(config['PROCESSING']['max_len'])
    vocab = list(FP.CST_PRIOR.keys())
    
    namefile = f'{epoch}_all_smis'
    generated_smi = hp.read_with_pd(f'{dir_gen}{namefile}.txt')
    smis_for_extraction = {}
    for i,smi in enumerate(generated_smi):
        tokenized_s, _ = hp.process_smi(smi, min_len, max_len, vocab)
        smis_for_extraction[i] = tokenized_s
                
    hp.save_obj(smis_for_extraction, f'{savepath}{namefile}_for_extraction.pkl')

    end = time.time()
    if verbose: print(f'PROCESSING DONE in {end-start:.2f} seconds')  
    ####################################
        