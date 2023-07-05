import k2
import numpy
import torch
from IPython.display import Image, SVG

import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import k2
import torch

import sys
icefall_path = "/export/home/lium/priera/icefall"
sys.path.insert(0,icefall_path)

from icefall.lexicon import UniqLexicon

def two_state_hmm_topo(max_tokens, weight=0):
    """
    Given a max number of tokens/labels generates the FSA that allows all the 
    possible strings from 1 to max tokens, using 2-state HMM with out skiping
    """
    arcs = []
    auxlabels = []
    final = max_tokens*3+1
    s = 1
    for ix in range(1,max_tokens+1):
        arcs.extend([
                     [0, s, 0, weight],
                     [s, s+1, ix, weight],
                     [s+1, s+2, ix, weight],
                     [s+2, s+2, ix, weight],
                     [s+2, 0, ix, weight],
                     [s+2, final, -1, weight]])
        s = s+3
        auxlabels.extend([0,ix,ix,ix,0,-1])
    arcs = torch.tensor(arcs,dtype=torch.int32)
    ix = torch.argsort(arcs[:,0])
    auxlabels = torch.tensor(auxlabels)
    fsa = k2.Fsa(arcs[ix], aux_labels=auxlabels[ix])  
    return fsa

def two_state_linear_fsa(token_list):  
    """
    Given a token list generates the `linear FSA` where 
    each token is forced to be emited twice or more (2-state HMM with out skiping)
    """
    s = 0
    arcs = []
    auxlabels = []
    w = 0
    for ix in token_list:
        arcs.extend([[s, s+1, ix, w],
                     [s+1, s+2, ix, w],
                     [s+2, s+2, ix, w]])
        s = s+2
        auxlabels.extend([ix]*3)
    arcs.append([s,s+1,-1, w])
    auxlabels.append(-1)
    return k2.Fsa(torch.tensor(arcs,dtype=torch.int32), aux_labels= torch.tensor(auxlabels))    