import unittest
import numpy as np
from pyCLINE.generate_data import FHN, Bicubic, GeneExpression, DelayOscillator
from pyCLINE.model import FHN as FHNModel, Bicubic as BicubicModel, GeneExpression as GeneExpressionModel, DelayOscillator as DelayOscillatorModel
import pandas as pd

class TestGenerateData(unittest.TestCase):

    def test_FHN(self):
        # Test FHN data generation
        try:
            FHN(dt=0.1, N=1000, epsilons=[0.3], n_intiaL_conditions=1)
        except Exception as e:
            self.fail(f"FHN data generation failed with exception: {e}")
        
        # Check the shape of the generated data
        fhn = FHNModel([1, 1, 0.3, 0.5, 0.0])
        u0, v0 = np.meshgrid(np.linspace(-1.25, 1.75, 1), np.linspace(-0.75, 1.75, 1))
        x0 = np.array([u0, v0])
        fhn.generate_data(x0, 0.01, 100, check_period=False)
        
        path= 'data/synthetic_data/'
        fname='FHN_eps=0.3_a=0.0.csv'
        data = pd.read_csv(path+fname)

        self.assertEqual(data.shape, (100,4))

    def test_Bicubic(self):
        # Test Bicubic data generation
        try:
            Bicubic(dt=0.1, N=1000, n_intiaL_conditions=1)
        except Exception as e:
            self.fail(f"Bicubic data generation failed with exception: {e}")
        
        # Check the shape of the generated data
        bicubic = BicubicModel([-0.5, 0.5, -1/3])
        u0, v0 = np.meshgrid(np.linspace(-1.25, 1.75, 1), np.linspace(-0.75, 1.75, 1))
        x0 = np.array([u0, v0])
        bicubic.generate_data(x0, 0.01, 100, check_period=False)

        path= 'data/synthetic_data/'
        fname='Bicubic.csv'
        data = pd.read_csv(path+fname)

        self.assertEqual(data.shape, (100,4))

    def test_GeneExpression(self):
        # Test GeneExpression data generation
        try:
            GeneExpression(dt=0.1, N=1000, n_intiaL_conditions=1)
        except Exception as e:
            self.fail(f"GeneExpression data generation failed with exception: {e}")
        
        # Check the shape of the generated data
        gene_expression = GeneExpressionModel([1, 0.05, 1, 0.05, 1, 0.05, 1, 1, 0.1, 2])
        u0, v0 = np.meshgrid(np.linspace(0, 1.75, 1), np.linspace(0, 1.75, 1))
        x0 = np.array([u0, v0])
        gene_expression.generate_data(x0, 0.01, 100, check_period=False)

        path= 'data/synthetic_data/'
        fname='GeneExpression.csv'
        data = pd.read_csv(path+fname)

        self.assertEqual(data.shape, (100,4))

    def test_DelayOscillator(self):
        # Test DelayOscillator data generation
        try:
            DelayOscillator(N=1000)
        except Exception as e:
            self.fail(f"DelayOscillator data generation failed with exception: {e}")
        
        # Check the shape of the generated data
        delay_osci = DelayOscillatorModel([4, 10, 2])
        delay_osci.generate_data(y_0=0, dt=0.4, t_max=400, check_period=False)
        path= 'data/synthetic_data/'
        fname='DelayOscillator.csv'
        data = pd.read_csv(path+fname) 
        self.assertEqual(data.shape, (int(400/0.4),2))

    # def test_invalid_parameters(self):
    #     # Test invalid parameters for FHN
    #     with self.assertRaises(ValueError):
    #         FHN(dt=-0.1, N=1000, epsilons=[0.3], n_intiaL_conditions=1)
    #     with self.assertRaises(ValueError):
    #         FHN(dt=0.1, N=-1000, epsilons=[0.3], n_intiaL_conditions=1)
    #     with self.assertRaises(ValueError):
    #         FHN(dt=0.1, N=1000, epsilons=[0.3], n_intiaL_conditions=0)

    #     # Test invalid parameters for Bicubic
    #     with self.assertRaises(ValueError):
    #         Bicubic(dt=-0.1, N=1000, n_intiaL_conditions=1)
    #     with self.assertRaises(ValueError):
    #         Bicubic(dt=0.1, N=-1000, n_intiaL_conditions=1)
    #     with self.assertRaises(ValueError):
    #         Bicubic(dt=0.1, N=1000, n_intiaL_conditions=0)

    #     # Test invalid parameters for GeneExpression
    #     with self.assertRaises(ValueError):
    #         GeneExpression(dt=-0.1, N=1000, n_intiaL_conditions=1)
    #     with self.assertRaises(ValueError):
    #         GeneExpression(dt=0.1, N=-1000, n_intiaL_conditions=1)
    #     with self.assertRaises(ValueError):
    #         GeneExpression(dt=0.1, N=1000, n_intiaL_conditions=0)

        # # Test invalid parameters for DelayOscillator
        # with self.assertRaises(ValueError):
        #     DelayOscillator(N=-1000)

if __name__ == '__main__':
    unittest.main()