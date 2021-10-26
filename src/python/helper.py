import sys, os
import pickle
import pandas as pd
import re
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize 
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')


def save_obj(obj, name):
    """save an object with pickle"""
    name = name.replace('.pkl', '')
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """load a pickle object"""
    name = name.replace('.pkl', '')
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def write_in_file(path_to_file, data):
    with open(path_to_file, 'w+') as f:
        for item in data:
            f.write("%s\n" % item)
            
def smi_tokenizer(smi):
    """
    Tokenize a SMILES
    """
    pattern =  "(\[|\]|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Si|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    return tokens
            
def uncharge(smi):
    uncharger = rdMolStandardize.Uncharger()
    m = Chem.MolFromSmiles(smi)
    nm = Chem.MolFromMolBlock(Chem.MolToMolBlock(m))
    un_smi = Chem.MolToSmiles(uncharger.uncharge(m))
    return un_smi

def get_canon(smi):
    """
    smi: (str) a SMILES string.
    Return: the canonical form
    of the SMILES, or none if not
    possible
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is not None: 
        can = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return can
    else:
        return None

def is_ok(tokenized, vocab, min_len, max_len):
    """
    Check if a given SMILES matches
    a set of constrains.
    Input: 
        tokenized: (list) a SMILES, 
        split by SMILES characters.
        vocab: (dict) SMILES vocabulary.
        max_len: (int) maximum SMILES length
    """
    if len(tokenized)>max_len:
        return False
    elif len(tokenized)<min_len:
        return False
    for x in tokenized:
        if x not in vocab:
            return False
    return True

def perplexity_w_prior(probabilites, smi, d_prior, remove_end=True):
    """
    Compute the perplexity of a SMILES given a list
    of probabilities.
    probabilites: (list) of probabilities
    smi: (str) a SMILES
    d_prior: (dict) with keys: SMILES characters,
    and value the prior probability to sample
    the characters
    """
    tokenized_smi = smi_tokenizer(smi)
    # We don't take into account the end character
    # in this version (set to False if already remove
    # beforehand)
    if remove_end:
        probabilites = probabilites[:-1]
    assert len(tokenized_smi)==len(probabilites)
    
    N = len(probabilites)
    weighted_logs = []
    for p,t_s in zip(probabilites, tokenized_smi):
        l = np.log(p) * d_prior[t_s]
        weighted_logs.append(l)
    sum_logs = np.sum(weighted_logs)
    
    return 2**(-(1/N)*sum_logs)
                            
def process_smi(smi, min_len, max_len, vocab):
    """
    Helper function to process
    a SMILES
    """
    un_s = uncharge(smi)
    c_un_s = get_canon(un_s)
    tokenized_s = smi_tokenizer(c_un_s)
    if is_ok(tokenized_s, vocab, min_len, max_len):
        return tokenized_s, c_un_s
    else:
        return None
                
def forced_sample(model, tsmi, temp, start_char, end_char, indices_token, token_indices, n_chars):
    """
    Compute the probability of each character for an existing SMILES.
    model: CLM deep learning model
    tsmi: (list) a SMILES split by SMILES characters,
    without the start nor the end charater
    temp: (float) temperature paramter
    start_char: (str) start character
    end_char: (str) end character
    indices_token: (dict) keys: index, tokens: SMILES characters
    token_indices: (dict) keys: SMILES characters, tokens: index
    n_chars: (int) size of the SMILES characters vocabulary
    """
    assert isinstance(tsmi, list)
    assert tsmi[-1]!=end_char
    assert tsmi[0]!=start_char
    enclosed_smi = [start_char] + tsmi + [end_char]
        
    def get_onehot(idx, n_chars):
        onehot = np.zeros((n_chars,))
        onehot[idx] = 1
        return onehot
    
    tokenized_smi = [token_indices[s] for s in enclosed_smi]
    onehot_smi = np.array([get_onehot(t, n_chars) for t in tokenized_smi])
    onehot_smi = np.expand_dims(onehot_smi, axis=0)
    
    # get proba
    output = model.predict(onehot_smi, verbose=0)[0]
    # we don't take the prediction of the start char character
    forced_proba = output[np.arange(len(output)-1), tokenized_smi[1:]]
                
    return forced_proba 

def is_novo(smi, d_data):
    """
    smi: (str) SMILES strings
    d_data: (dict) training + fine-tuning data
    used, where keys are the SMILES strings. 
    Note: we use a dict as it's faster to search
    """
    if smi in d_data:
        return False
    else:
        return True

def get_dict_pp(data, d_prior):
    """
    Helper function to get
    the perplexity in a dict
    """
    dict_pp = {}
    for key,item in data.items():
        dict_pp[key] = perplexity_w_prior(item, key, d_prior)
        
    return dict_pp

def get_idx_sorted(dict_pp):
    """
    helper function
    """
    smis = []
    vals = []
    for key,val in dict_pp.items():
        smis.append(key)
        vals.append(val)
    idx_sorted = np.argsort(vals)
    
    return idx_sorted, smis, vals

def read_with_pd(path, delimiter='\t', header=None):
    """helper funcion to read a txt file"""
    data_pd = pd.read_csv(path, delimiter=delimiter, header=header)
    return data_pd[0].tolist() 
