import unittest
from unittest.mock import patch

import os
import sys
from pathlib import Path as Path
import numpy as np

from src.models.sign_flip import pcorr_from_precision,split_lagged_global_by_lag, gain_cost, flip_diff, suggest_flips
from src.models.temporal_embedding import compute_time_embedding

class TestPcorr(unittest.TestCase):
    
    def test_partial_corr_func_matches_naive_loop_pcorr(self):
        d = 20
        X = np.random.randn(10000, d)
        X[:,1] += 0.3*X[:,0]
        X[:,4] = X[:,4] + 0.2*X[:,3] - 0.7*X[:,10]
        X[:,18] -= 0.1 * X[:,10]
        X[:,15] += 0.9*X[:,10]
        
        # Compute partial covariance matrix, 20 by 20
        precision_mat = np.linalg.inv(np.cov(X.T))
       
        
        # Compute expected empirical pcorr matrix!
        pcorr_expected = np.zeros((d,d))
        for i in range(d):
            for j in range(d):
                pcorr_expected[i,j] = -precision_mat[i,j]/np.sqrt(precision_mat[i,i]*precision_mat[j,j])
        pcorr_recovered = pcorr_from_precision(precision_mat)
        self.assertTrue(np.all(np.abs(pcorr_recovered - pcorr_expected) < 1e-10))
        
    def test_splitting_covar_by_lags_is_correct_D_continuous(self):
        # Create a covariance matrix where the features are "D" continous (ie, from different lags)
        # To do so, we'll simply create fake data and generate corresponding matrices directly.
        N = 10000
        d = 4
        L = 4
        
        # Create a time series
        X = np.random.randn(N, d)
        
        # Create lagged representation of the time series.
        lagged_X = compute_time_embedding(X, L)
        
        # The embedding function comprises a mirror boundary condition. To keep only valid entries, we'll truncate the lags outside.
        lagged_X = lagged_X[L:-L,:]
        
        # Let's compute the expected covariance matrix.
        n_lags = 2*L+1
        covar_expected = np.zeros((d,d,n_lags))
        x_ref = lagged_X[:,L*d:(L+1)*d]
        for l in range(n_lags):
            # Compute the covariance matrix between the lag l and non shifted data in the embedding
            covar_expected[:,:,l] = np.cov(lagged_X[:,l*d:(l+1)*d].T, x_ref.T)[:d,d:] # numpy cov returns a 2*d by 2*d array, select appropriate entry
        
        # Now construct one big covariance directly from the data to be reordered by our method
        covar_big = np.cov(lagged_X.T)
        # Reorder
        covar_recovered = split_lagged_global_by_lag(covar_big, n_lags, order="D")
        self.assertTrue(np.all(np.abs(covar_recovered - covar_expected)<1e-8))
            
            
class TestFlips(unittest.TestCase):
    def test_flip_cost_is_non_negative(self):
        X = np.random.randn(10, 10, 4, 5)
        flips = np.random.choice([-1,1], (4,10))
        cost_pre_flip = gain_cost(flips, X)
        self.assertTrue(cost_pre_flip >= 0)
        
    def test_flip_cost_diff_equal_to_naive_cost_diff(self):
        X = np.random.randn(10, 10, 4, 5)
        flips = np.random.choice([-1,1], (4,10))
        cost_pre_flip = gain_cost(flips, X)
        
        flips_after = np.copy(flips)
        flips_after[3, 2] *= -1
        cost_post_flip = gain_cost(flips_after, X)
        diff_cost_expected = cost_post_flip - cost_pre_flip
        
        diff_cost_actual = flip_diff(flips, X, 2, 3)
        self.assertTrue(abs(diff_cost_actual - diff_cost_expected) < 10e-7)
        
    def test_flip_cost_differs_if_several_flips_applied(self):
        X = np.random.randn(10, 10, 4, 5)
        flips = np.random.choice([-1,1], (4,10))
        cost_pre_flip = gain_cost(flips, X)
        
        flips_after = np.copy(flips)
        flips_after[3, 2] *= -1
        flips_after[1, 1] *= -1
        cost_post_flip = gain_cost(flips_after, X)
        diff_cost_expected = cost_post_flip - cost_pre_flip
        
        diff_cost_actual = flip_diff(flips, X, 2, 3)
        self.assertFalse(abs(diff_cost_actual - diff_cost_expected) < 10e-7)
        
    def test_flip_cost_non_zero(self):
        X = np.random.randn(10, 10, 4, 5)
        flips = np.random.choice([-1,1], (4,10))
        diff_cost_actual = flip_diff(flips, X, 2, 3)
        self.assertTrue(abs(diff_cost_actual) > 1e-7)
        
    def test_flips_iter_return_bigger_than_zero_gain(self):
        X = np.random.randn(10,10,4,5)
        flips, gain = suggest_flips(10, 100, X)
        self.assertTrue(gain > 0)

if __name__ == '__main__':
    unittest.main()