import unittest
from unittest.mock import patch

import os
import sys
from pathlib import Path as Path

from src.data.preprocess import preprocess_fif

class TestPreprocess(unittest.TestCase):
    
    @patch('src.data.preprocess.filtering_step')
    @patch('src.data.preprocess.ica_step')
    @patch('src.data.preprocess.source_setup_step')
    @patch('src.data.preprocess.bem_mesh_step')
    @patch('src.data.preprocess.manual_coreg_step')
    @patch('src.data.preprocess.fwd_step')
    @patch('src.data.preprocess.perform_source_reconstruction')
    @patch('src.data.preprocess.atlasing_step')
    def test_preprocess_no_error(self, filtering_step_mock,ica_step_mock, source_setup_step_mock, bem_mesh_step_mock, manual_coreg_step_mock, fwd_step_mock, perform_source_reconstruction_mock, atlasing_step_mock):
        try:
            # Mock all inner functions with mockito to first check that everything runs smooth
            preprocess_fif(str("1006"), "some_fif_dir", "some_mri_dir", "some_atlas_t1", "some_atlas", skip_steps=[], overwrite=True)
        except Exception as e:
            self.fail("preprocess_fif raised the following exception: " + e)
            
if __name__ == '__main__':
    unittest.main()