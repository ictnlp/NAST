from collections import defaultdict, Counter
import re
import numpy as np
from copy import copy
import math

NEG_INF = -float("inf")


def logsumexp(*args):
  """
  Stable log sum exp.
  """
  if all(a == NEG_INF for a in args):
      return NEG_INF
  a_max = max(args)
  lsp = math.log(sum(math.exp(a - a_max)
                      for a in args))
  return a_max + lsp

def merge_two_probs(Pb, Pnb):
    A_prev = copy(Pb)
    for k,v in Pnb.items():
        if (k in Pb):
            A_prev[k] = logsumexp(A_prev[k], Pnb[k])
        else:
            A_prev[k] = Pnb[k]
    return A_prev

def default_value():
    return NEG_INF

def nested_defaultdict():
    return defaultdict(default_value)


class streaming_prefix_beam_search:
    def __init__(self, dictionary, blank_idx, lm=None, k=25, alpha=0.30, beta=5, cutoff_top_n=10, erasable=False):
        """
        Args:
            
            lm (func): Language model function. Should take as input a string and output a probability.
            k (int): The beam width. Will keep the 'k' most likely candidates at each timestep.
            alpha (float): The language model weight. Should usually be between 0 and 1.
            beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
            erasable (bool): Keep beam width at the end of search in each chunk.
        """
        self.dictionary = dictionary
        self.blank_idx = blank_idx
        self.lm = lm
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.cutoff_top_n = cutoff_top_n
        self.erasable = erasable
        
        #self.Pb = defaultdict(lambda: defaultdict(lambda: NEG_INF))
        #self.Pnb = defaultdict(lambda: defaultdict(lambda: NEG_INF))
        self.Pb = defaultdict(nested_defaultdict)
        self.Pnb = defaultdict(nested_defaultdict)
        O = ()
        self.Pb[0][O] = 0.0
        self.Pnb[0][O] = NEG_INF
        self.A_prev = [O]
        
        self.T_prev = 0
        

    def streaming_search(self, chunk_logits):
        """
        Performs streaming prefix beam search on the output of a CTC network.
        Args:
            chunk_logits (np.ndarray): The CTC output in a chunk. Should be a 2D array (timesteps x alphabet_size)
        
        Retruns: Nothing. Only modify self.A_prev
            
        """
        
        #F = chunk_logits.shape[1]
        T = chunk_logits.shape[0]

        for t in range(0, T):
            pruned_alphabet = np.argsort(chunk_logits[t])[::-1][:self.cutoff_top_n]
            for l in self.A_prev:
                for c_idx in pruned_alphabet:
                    # Extending with a blank
                    if c_idx == self.blank_idx:
                        self.Pb[self.T_prev + t + 1][l] = logsumexp(self.Pb[self.T_prev + t + 1][l], chunk_logits[t][self.blank_idx] + self.Pb[self.T_prev + t][l], chunk_logits[t][self.blank_idx] + self.Pnb[self.T_prev + t][l])
                    else:
                        
                        l_plus = tuple(list(l) + [c_idx])
                        # Extending with the end character
                        if len(l) > 0 and c_idx == l[-1]:
                            self.Pnb[self.T_prev + t + 1][l_plus] = logsumexp(self.Pnb[self.T_prev + t + 1][l_plus], chunk_logits[t][c_idx] + self.Pb[self.T_prev + t][l])
                            self.Pnb[self.T_prev + t + 1][l] = logsumexp(self.Pnb[self.T_prev + t + 1][l], chunk_logits[t][c_idx] + self.Pnb[self.T_prev + t][l])
                        # Extending with any other non-blank character
                        else:
                            self.Pnb[self.T_prev + t + 1][l_plus] = logsumexp(self.Pnb[self.T_prev + t + 1][l_plus], chunk_logits[t][c_idx] + self.Pb[self.T_prev + t][l], chunk_logits[t][c_idx] + self.Pnb[self.T_prev + t][l])
                            
                        # Make use of discarded prefixes
                        if l_plus not in self.A_prev:
                            self.Pb[self.T_prev + t + 1][l_plus] = logsumexp(self.Pb[self.T_prev + t + 1][l_plus], chunk_logits[t][self.blank_idx] + self.Pb[self.T_prev + t][l_plus], chunk_logits[t][self.blank_idx] + self.Pnb[self.T_prev + t][l_plus])
                            self.Pnb[self.T_prev + t + 1][l_plus] = logsumexp(self.Pnb[self.T_prev + t + 1][l_plus], chunk_logits[t][c_idx] + self.Pnb[self.T_prev + t][l_plus])
                        
            
            # Select most probable prefixes
            A_next = merge_two_probs(self.Pb[self.T_prev + t + 1], self.Pnb[self.T_prev + t + 1])
            
            
            sorter = lambda l: A_next[l]
            self.A_prev = sorted(A_next, key=sorter, reverse=True)[:self.k]
            
        self.T_prev = self.T_prev + T
        
        if not self.erasable:
            self.A_prev = self.A_prev[:1]
        
        return