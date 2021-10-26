import os, sys
import time
import argparse
import configparser

import numpy as np

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

from keras.models import load_model

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP
from python import data_generator as dg

parser = argparse.ArgumentParser(description='Run beam search')
parser.add_argument('-c','--configfile', type=str, help='path to config file', required=True)
parser.add_argument('-f','--name_data', type=str, help='Name of the ft file', required=True)
parser.add_argument('-e','--epoch', type=str, help='Which epoch to sample from', required=True)
parser.add_argument('-r','--repeat', type=int, help='Number of repeats', required=True)
parser.add_argument('-k','--width', type=int, help='Beam width (k)', required=True)


def int_to_smile(array, indices_token, pad_char):
    """ 
    From an array of int, return a list of 
    molecules in string smile format
    Note: remove the padding char
    """
    all_mols = []
    for seq in array:
        new_mol = [indices_token[int(x)] for x in seq]
        all_mols.append(''.join(new_mol).replace(pad_char, ''))
    return all_mols

def one_hot_encode(token_lists, n_chars):
    output = np.zeros((1, len(token_lists), n_chars))
    for i, token in enumerate(token_lists):
        output[0, i, int(token)] = 1
    return output

def save_data(candidates, indices_token, start_char, pad_char, end_char, save_path, name_file):
    """
    Save sampled SMILES
    Canonicalised with rdkit
    """
    
    all_smis = []
    
    for i,x in enumerate(candidates):
        # we have to do one more loop because x is a np array 
        # of dimensions bz, len_smiles, vocab. as the bz is one,
        # which is needed for the keras model, we have to do
        # one more loop to extract the SMILES
        for y in x:
            ints = [indices_token[np.argmax(w)] for w in y]
            smi = ''.join(ints).replace(start_char,'').replace(pad_char,'').replace(end_char,'')
            can = hp.get_canon(smi)
            if can:
                all_smis.append(can)
            
    hp.write_in_file(f'{save_path}{name_file}_all_smis.txt', all_smis)

def beam_search_decoder(k, model, vocab_size, max_len, 
                        indices_token, token_indices, name_file, 
                        start_string, pad_char, end_char, save_path, verbose):
    """
    Run the beam search.
    k: (int) width of the beam
    model: keras deep learning model
    vocab_size: (int) number of SMILES characters
    in the vocabulary
    max_len: (int) maximum SMILES length
    indices_token: (dict) keys=index, values: SMILES characters
    token_indices: (dict) keys=SMILES characters, values: index
    name_file: (str) epoch (note: 0X if X<10, where X is the epoch)
    start_string: (str) start SMILES character
    start_string: (str) pad SMILES character
    start_string: (str) end SMILES character
    """
    seed_token = [token_indices[x] for x in start_string]
    
    # candidates is a matrix of len(seed_token) 
    # one hot encoded with k elements
    X = one_hot_encode(seed_token, vocab_size)
    candidates = [X]
    scores = [1]*k
    
    for j in range(max_len):
        current_candidates = []
        current_scores = []
        
        for i,x in enumerate(candidates):
            preds = model.predict(x, verbose=0)[0]
            preds = preds[-1, :]
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds)
            
            # Argsort in descending order, only for the k top
            idx_preds_sorted = np.argsort(preds)[::-1][:k]
            preds_sorted = preds[idx_preds_sorted]
            
            for idx_pred in idx_preds_sorted:
                vec = one_hot_encode([idx_pred], vocab_size)
                new_seq = np.concatenate((x, vec), axis=1)
                current_candidates.append(new_seq)
            
            current_scores.extend([a+b for a,b in zip(scores,preds_sorted.tolist())])
            
        # Find the k best candidates from the scores
        idx_current_best = np.argsort(current_scores)[::-1][:k]
        candidates = [x for i,x in enumerate(current_candidates) if i in idx_current_best]
        scores = [x for i,x in enumerate(current_scores) if i in idx_current_best]           
                
    # save results
    save_data(candidates, indices_token, start_char, pad_char, end_char, save_path, name_file)
        
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
    min_len = int(config['PROCESSING']['min_len'])
    max_len = int(config['PROCESSING']['max_len'])
    
    if verbose: print('\nSTART BEAM SAMPLING')
    ####################################
    
    
    
    
    ####################################
    # paths to save data and to checkpoints
    dir_exp = str(config['EXPERIMENTS']['dir'])
    exp_name = configfile.split('/')[-1].replace('.ini','')
    if repeat>0:
        save_path = f'{dir_exp}/{mode}/{exp_name}/{name_data}/beam_sampling/{width}/{repeat}/'
        dir_ckpts = f'{dir_exp}/{mode}/{exp_name}/{name_data}/models/{repeat}/'
    else:
        save_path = f'{dir_exp}/{mode}/{exp_name}/{name_data}/beam_sampling/{width}/'
        dir_ckpts = f'{dir_exp}/{mode}/{exp_name}/{name_data}/models/'
    
    os.makedirs(save_path, exist_ok=True)
    ####################################
    

    
    
    ####################################
    # Generator parameters
    max_len = int(config['PROCESSING']['max_len'])
    pad_char = FP.PROCESSING_FIXED['pad_char']
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    indices_token = FP.INDICES_TOKEN
    token_indices = FP.TOKEN_INDICES
    vocab_size = len(indices_token)
    ####################################
    
    
    
    
    ####################################
    # start the sampling of new SMILES           
    if verbose: print(f'Sampling from model saved at epoch {epoch}')
    model_path = f'{dir_ckpts}{epoch}.h5'
    model = load_model(model_path)
    
    beam_search_decoder(width, model, vocab_size, max_len, 
                        indices_token, token_indices, epoch, 
                        start_char, pad_char, end_char, save_path, verbose)
    
    end = time.time()
    if verbose: print(f'BEAM SAMPLING DONE in {end - start:.05} seconds')
    ####################################
        