import unittest
import pandas as pd
import numpy as np
from pyCLINE.recovery_methods.data_preparation import prepare_data, shuffle_and_split

class TestDataPreparation(unittest.TestCase):

    def setUp(self):
        # Setup common variables for tests
        self.df = pd.DataFrame({
            'time': np.linspace(0, 10, 100),
            'var1': np.sin(np.linspace(0, 10, 100)),
            'var2': np.cos(np.linspace(0, 10, 100))
        })
        self.vars = ['var1', 'var2']
        self.time = 'time'

    def test_prepare_data(self):
        # Test prepare_data function
        df_prepared, df_coef = prepare_data(self.df, self.vars, self.time)
        self.assertTrue('norm var1' in df_prepared.columns)
        self.assertTrue('norm var2' in df_prepared.columns)
        self.assertTrue('var1' in df_coef.columns)
        self.assertTrue('var2' in df_coef.columns)

    def test_prepare_data_invalid_input(self):
        with self.assertRaises(ValueError):
            prepare_data([], self.vars, self.time)
        with self.assertRaises(ValueError):
            prepare_data(self.df, 'var1', self.time)
        with self.assertRaises(ValueError):
            prepare_data(self.df, self.vars, 123)
        with self.assertRaises(ValueError):
            prepare_data(self.df, self.vars, self.time, tmin='invalid')
        with self.assertRaises(ValueError):
            prepare_data(self.df, self.vars, self.time, tmax='invalid')
        with self.assertRaises(ValueError):
            prepare_data(self.df, self.vars, self.time, scheme=123)
        with self.assertRaises(ValueError):
            prepare_data(self.df, self.vars, self.time, norm_method=123)
        with self.assertRaises(ValueError):
            prepare_data(self.df, self.vars, self.time, value_min='invalid')
        with self.assertRaises(ValueError):
            prepare_data(self.df, self.vars, self.time, value_max='invalid')
        with self.assertRaises(ValueError):
            prepare_data(self.df, self.vars, self.time, normalize='invalid')
        with self.assertRaises(ValueError):
            prepare_data(self.df, ['var1', 'varINVALID'], self.time)
        with self.assertRaises(ValueError):
            prepare_data(self.df, self.vars, 'invalid_time')
        with self.assertRaises(ValueError):
            prepare_data(self.df, self.vars, self.time, scheme='invalid')

    def test_shuffle_and_split(self):
        # Test shuffle_and_split function
        input_train, target_train, input_test, target_test, input_val, target_val = shuffle_and_split(
            self.df, self.vars, ['var1'], train_frac=0.7, test_frac=0.15, optimal_thresholding=False
        )
        self.assertIsInstance(input_train, pd.DataFrame)
        self.assertIsInstance(target_train, pd.DataFrame)
        self.assertIsInstance(input_test, pd.DataFrame)
        self.assertIsInstance(target_test, pd.DataFrame)
        self.assertIsInstance(input_val, pd.DataFrame)
        self.assertIsInstance(target_val, pd.DataFrame)

    def test_shuffle_and_split_invalid_input(self):
        # Test shuffle_and_split function with invalid inputs
        with self.assertRaises(ValueError):
            shuffle_and_split([], self.vars, ['var1'])
        with self.assertRaises(ValueError):
            shuffle_and_split(self.df, 'var1', ['var1'])
        with self.assertRaises(ValueError):
            shuffle_and_split(self.df, self.vars, 'var1')
        with self.assertRaises(ValueError):
            shuffle_and_split(self.df, self.vars, ['var1'], train_frac='invalid')
        with self.assertRaises(ValueError):
            shuffle_and_split(self.df, self.vars, ['var1'], test_frac='invalid')
        with self.assertRaises(ValueError):
            shuffle_and_split(self.df, self.vars, ['var1'], train_frac=0.8, test_frac=0.3)
        with self.assertRaises(ValueError):
            shuffle_and_split(self.df, self.vars, ['var1'], train_frac=0.1, test_frac=0.2)
        with self.assertRaises(ValueError):
            shuffle_and_split(self.df, self.vars, ['var1'], optimal_thresholding='invalid')
        with self.assertRaises(ValueError):
            shuffle_and_split(self.df, self.vars, ['var1'], plot_thresholding='invalid')
        with self.assertRaises(ValueError):
            shuffle_and_split(self.df, self.vars, ['varINVALID'])
        with self.assertRaises(ValueError):
            shuffle_and_split(self.df, ['varINVALID'] , ['var1'])

if __name__ == '__main__':
    unittest.main()