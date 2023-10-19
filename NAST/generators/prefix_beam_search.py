from collections import defaultdict, Counter
import re
import numpy as np
NEG_INF = -float("inf")

class streaming_prefix_beam_search:
    def __init__(self, dictionary, blank_idx, lm=None, k=25, alpha=0.30, beta=5, erasable=False):
        """
        Args:
            
            lm (func): Language model function. Should take as input a string and output a probability.
            k (int): The beam width. Will keep the 'k' most likely candidates at each timestep.
            alpha (float): The language model weight. Should usually be between 0 and 1.
            beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
            erasable (bool): Keep beam width at the end of search in each chunk.
        """
        
        self.Pb = defaultdict(Counter)
        self.Pnb = defaultdict(Counter)
        O = []
        self.Pb[0][O] = 1
        self.Pnb[0][O] = 0
        self.A_prev = [O]
        self.blank_idx = blank_idx
        self.T_prev = 0
        self.dictionary = dictionary
        self.erasable = erasable
        self.lm = lm
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def prefix_beam_search(chunk_probs):
        """
        Performs streaming prefix beam search on the output of a CTC network.
        Args:
            chunk_probs (np.ndarray): The CTC output in a chunk. Should be a 2D array (timesteps x alphabet_size)
        
        Retruns:
            
        """

        F = chunk_probs.shape[1]
        #ctc = np.vstack((np.zeros(F), ctc)) # just add an imaginative zero'th step (will make indexing more intuitive)
        T = chunk_probs.shape[0]


        for t in range(0, T):
            for l in self.A_prev:
                for c in dictionary.symbols:
                    c_idx = dictionary.index(c)
                    
                    # Extending with a blank
                    if c_idx == self.blank_idx:
                        Pb[self.T_prev + t + 1][l] += chunk_probs[t][self.blank_idx] * (Pb[self.T_prev + t][l] + Pnb[self.T_prev + t][l])
                    
                    else:
                        l_plus = l + [c_idx]
                        # Extending with the end character
                        if len(l) > 0 and c_idx == l[-1]:
                            Pnb[self.T_prev + t + 1][l_plus] += chunk_probs[t][c_idx] * Pb[self.T_prev + t][l]
                            Pnb[self.T_prev + t + 1][l] += chunk_probs[t][c_idx] * Pnb[self.T_prev + t][l]

                        # Extending with any other non-blank character
                        else:
                            Pnb[self.T_prev + t + 1][l_plus] += chunk_probs[t][c_idx] * (Pb[self.T_prev + t][l] + Pnb[self.T_prev + t][l])
                            
                        # Make use of discarded prefixes
                        if l_plus not in self.A_prev:
                            Pb[self.T_prev + t + 1][l_plus] += chunk_probs[t][self.blank_idx] * (Pb[self.T_prev + t][l_plus] + Pnb[self.T_prev + t][l_plus])
                            Pnb[self.T_prev + t + 1][l_plus] += chunk_probs[t][c_idx] * Pnb[self.T_prev + t][l_plus]
                        

            # Select most probable prefixes
            A_next = Pb[t] + Pnb[t]
            #sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
            sorter = lambda l: A_next[l]
            self.A_prev = sorted(A_next, key=sorter, reverse=True)[:self.k]
            
        self.T_prev = self.T_prev + T
        if not self.erasable:
            self.A_prev = self.A_prev[0]
        return