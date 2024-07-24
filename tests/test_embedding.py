import unittest
from unittest.mock import patch

import os
import sys
from pathlib import Path as Path

from src.models.temporal_embedding import compute_time_embedding
import numpy as np

class TestTimeEmbedding(unittest.TestCase):
    
    def test_embedding_create_right_lag_number(self):
        X = np.random.randn(10, 3)
        L = 3
        embed = compute_time_embedding(X, L)
        self.assertTrue(embed.shape[1] == X.shape[1] * (2*L+1))
        
    def test_central_lag_equal_to_X(self):
        d = 4
        X = np.random.randn(10, d)
        L = 7
        embed = compute_time_embedding(X,L)
        self.assertTrue(np.all(X == embed[:,L*d:L*d + d]))
        
    def test_all_lags_correct_without_padding(self):
        d = 4
        N = 10
        X = np.random.randn(N, d)
        L = 7
        embed = compute_time_embedding(X,L)
        n_lags = 2*L + 1
        for l in range(n_lags):
            shift  = l - L
            embed_l = embed[:, l*d:(l+1)*d]
            if shift < 0:
                # Consider mirrored version of X, and keep only last L indices (discard boundary mirror)
                pad_up = X[::-1,:][:-1][shift:]
                padded_X = np.vstack((pad_up, X[:N+shift]))
                self.assertTrue(np.all(padded_X == embed_l))
            else:
                if shift > 0:
                    # Consider mirrored version of X and keep only first L indices (discard boundary mirror)
                    pad_low = X[::-1,:][1:][:shift]
                    padded_X = np.vstack((X[shift:], pad_low))
                    self.assertTrue(np.all(padded_X == embed_l))
                else:
                    self.assertTrue(np.all(embed_l == X))
            
if __name__ == '__main__':
    unittest.main()