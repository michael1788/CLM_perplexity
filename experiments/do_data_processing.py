import os, sys
import argparse
import configparser
import time
import numpy as np
import random
from random import shuffle
import collections
from rdkit import Chem

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='Run data processing')
parser.add_argument('-c','--configfile', type=str, help='path to config file', required=True)
parser.add_argument('-f','--name_data', type=str, help='Name of the ft file', required=True)

def load_data(data_path, min_len, max_len, randomized_smi, verbose=False):
    """
    Function to load a .txt file of SMILES, 
    prune SMILES by length and check that they
    are convertible to RDKit mol format.
    
    Parameters:
    - data_path (string): path to the dataset.
    - min_len (int): minimum length of SMILES to be kept in the dataset.
    - max_len (int): maximum length of SMILES to be kept in the dataset.
    - randomized_smi (bool): if True, return a randomized version of each smi,
    vs not the canonicalized version
    
    return: 
    data -> a list with SMILES in string format
    data_rdkit -> a list with molecules in RDKit mol format
    """
    data = []
    data_rdkit = []
    
    with open(data_path) as f:
        for line in f:
            newline = line.rstrip('\r\n')
            tokenized = hp.smi_tokenizer(newline)
            if len(tokenized)<=max_len and len(tokenized)>=min_len and check_token(tokenized):
                # convert to RDKit mol format
                mol = Chem.MolFromSmiles(newline)
                  
                if mol is not None:
                    if randomized_smi:
                        print('\nrandomizing smi ON')
                        newline = Chem.MolToSmiles(mol, doRandom=True)
                        
                        tokenized = hp.smi_tokenizer(newline)
                        if len(tokenized)<=max_len and len(tokenized)>=min_len and check_token(tokenized):
                            data.append(newline)
                            data_rdkit.append(mol)
                    else:
                        data.append(newline)
                        data_rdkit.append(mol)
    
    if verbose: print(f'Size of the dataset after pruning by length and check with RDKit: {len(data)}')
    
    return data, data_rdkit

def check_token(tokenized_smi):
    for token in tokenized_smi:
        if token not in list(FP.INDICES_TOKEN.values()):
            return False
    return True

def randomSmiles(mol):
    mol.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0,mol.GetNumAtoms()))
    random.shuffle(idxs)
    for i,v in enumerate(idxs):
        mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(mol)

def smile_augmentation(smile, augmentation, min_len, max_len):
    mol = Chem.MolFromSmiles(smile)
    s = set()
    for i in range(1000):
        smiles = randomSmiles(mol)
        tokenized = hp.smi_tokenizer(smiles)
        if len(tokenized)<=max_len and check_token(tokenized):
            s.add(smiles)
            if len(s)==augmentation:
                break
    
    return list(s)

def augment_dataset(data_ori, augmentation, min_len, max_len, verbose=False):
    """ 
    Function to augment a dataset. 
    
    Parameters:
    - data_ori (list): list of SMILES string to augment.
    - augmentation (int): number of alternative SMILES to create.
    - min_len (int): minimum length of alternative SMILES.
    - max_len (int): maximum length of alternative SMILES.
    
    return: a list alternative SMILES representations of data_ori
    """
    all_alternative_smi = []    
    for i,x in enumerate(data_ori):
        alternative_smi = smile_augmentation(x, augmentation, min_len, max_len)
        all_alternative_smi.extend(alternative_smi)
        if verbose and i%50000:
            print(f'augmentation is at step {i}')
    if verbose:
        print('data augmentation done; number of new SMILES: {len(n_new)}')
        
    return all_alternative_smi
        
def do_processing(split, data_path, augmentation, min_len, max_len, save_dir, randomized_smi, verbose=True):
    """
    Function to process a dataset.
    
    Parameters:
    - split (float): value used to split the dataset between
    the training set and the validation set. E.g., if split is 0.8,
    80% of the data will go in the training set, and 20% in the 
    validation set.
    - data_path (string): path to the dataset.
    - augmentation (int): value to augment the dataset. E.g., if augmentation
    is 10, the SMILES enumeration will be done to add 10 different 
    SMILES encoding for each SMILES (i.e. resulting in a total of 11 representations)
    for a given SMILES in the dataset.
    - min_len (int): minimum length of SMILES to be kept in the dataset.
    - max_len (int): maximum length of SMILES to be kept in the dataset.
    - save_dir (string): directory to save the processed dataset.
    """
    
    # load the data with right SMILES limits, 
    # both in a list and in rdkit mol format 
    data_ori, data_rdkit = load_data(data_path, min_len, max_len, randomized_smi, verbose=verbose)
    
    if verbose: print('Start data processing')
    # define index for the tr-val split
    # and shuffle them
    all_idx = np.arange(len(data_ori))
    idx_split = int(split*len(all_idx))
    np.random.shuffle(all_idx)
    
    # we need to be careful about the case where
    # idx_split = 0 when there is only one 
    # SMILES in the data, e.g. for fine-tuning
    # Also, in the case we want to FT on all the data,
    # we defined the val set equal to the tr set
    if split==1.0:
        # in this case, tr and val are the same
        idx_tr_canon = all_idx
        idx_val_canon = all_idx
    elif idx_split==0:
        # in this case, we use the unique SMILES both  
        # for the training and validation
        idx_tr_canon = [0]
        idx_val_canon = [0]
    else:
        idx_tr_canon = all_idx[:idx_split]
        idx_val_canon = all_idx[idx_split:]
        
    assert len(idx_tr_canon)!=0
    assert len(idx_val_canon)!=0
    
    if verbose:
        print(f'Size of the training set after split: {len(idx_tr_canon)}')
        print(f'Size of the validation set after split: {len(idx_val_canon)}')
    
    d = dict(enumerate(data_ori))
    data_tr = [d.get(item) for item in idx_tr_canon]
    data_val = [d.get(item) for item in idx_val_canon]
    hp.write_in_file(f'{save_dir}data_tr.txt', data_tr)
    hp.write_in_file(f'{save_dir}data_val.txt', data_val)
    
    if augmentation>0:
        if verbose:
            print(f'Data augmentation {augmentation}-fold start')
        

        # Augment separately the training and validation splits
        # It's important to do those steps separetely in order
        # to avoid to have the same molecule represented in 
        # both splits
        tr_aug = augment_dataset(data_tr, augmentation, min_len, max_len, verbose=False) 
        val_aug = augment_dataset(data_val, augmentation, min_len, max_len, verbose=False) 
        
        # Merge with the original data and shuffle
        full_training_set = list(set(data_tr + tr_aug))
        shuffle(full_training_set)
        full_validation_set = list(set(data_val + val_aug))
        shuffle(full_validation_set)
        full_datalist = full_training_set + full_validation_set
                
        if verbose:
            print(f'Size of the training set after agumentation: {len(full_training_set)}')
            print(f'Size of the validation set after agumentation: {len(full_validation_set)}')
                    
        # Create the partitions for the data generators 
        # with the full augmented dataset
        idx_tr = np.arange(len(full_training_set))
        idx_val = np.arange(len(full_training_set), len(full_training_set) + len(full_validation_set))
    
        # Save
        hp.write_in_file(f'{save_dir}{save_name}.txt', full_datalist)
        hp.save_obj(list(idx_tr), save_dir + 'idx_tr')
        hp.save_obj(list(idx_val), save_dir + 'idx_val')
    else:
        # Save
        hp.write_in_file(f'{save_dir}{save_name}.txt', data_ori)
        hp.save_obj(list(idx_tr_canon), f'{save_dir}idx_tr')
        hp.save_obj(list(idx_val_canon), f'{save_dir}idx_val')

        
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
    
    # get back the experiment parameters
    min_len = int(config['PROCESSING']['min_len'])
    max_len = int(config['PROCESSING']['max_len'])
    split = float(config['PROCESSING']['split'])
    mode = config['EXPERIMENTS']['mode']
    dir_data = str(config['DATA']['dir'])
    
    randomized_smi = config.getboolean('PROCESSING', 'randomized_smi')
    
    # check if experiment mode exists
    if mode not in ['LM', 'FT']:
        raise ValueError('The mode you picked does not exist.')
    if verbose: 
        print('\nSTART PROCESSING')
        print(f'Experiment mode: {mode}')
    ####################################
    
    
   
               
    ####################################
    # define the path to the data file
    # and process    
    print(f'\nData being processed: {name_data}')
    full_data_path = f'{dir_data}{name_data}.txt'
    
    # define saving path
    # experiment parameters depending on the mode
    aug = int(config['AUGMENTATION']['fold'])
    if randomized_smi:
        save_name = f'{min_len}_{max_len}_x{aug}_randomized'
    else:
        save_name = f'{min_len}_{max_len}_x{aug}'
    save_dir = f'{dir_data}{name_data}/{save_name}/'
    os.makedirs(save_dir, exist_ok=True)
    
    # Check first if the training data was already done;
    # if yes, we skip the processing.
    if os.path.isfile(f'{save_dir}{save_name}.txt'):
        print(f'Data {save_name} already exist in dir {save_dir}\n-> skipping processing.')
    else:
        do_processing(split, full_data_path, aug, min_len, max_len, 
                      save_dir, randomized_smi, verbose=verbose)
                
    end = time.time()
    print(f'PROCESSING DONE in {end - start:.04} seconds') 
    ####################################
    
    