import unittest
import torch
import pandas as pd
from pyCLINE import example

class TestExample(unittest.TestCase):
    def test_example_run(self):
        with self.assertRaises(ValueError):
            example('')
        with self.assertRaises(ValueError):
            example('test')
        with self.assertRaises(ValueError):
            example(None)