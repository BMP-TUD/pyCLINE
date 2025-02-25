import unittest
import torch
import pandas as pd
from pyCLINE.recovery_methods.nn_training import (
    NeuralNetworkSetupError,
    NeuralNetworkSetupTrainingError,
    NeuralNetworkDataError,
    configure_FFNN_model,
    train_FFNN_model,
    FFNN
)

class TestNNTraining(unittest.TestCase):

    def setUp(self):
        # Setup common variables for tests
        self.Nin = 2
        self.Nout = 1
        self.Nlayers = 2
        self.Nnodes = 8
        self.lr = 1e-4
        self.optimizer_name = 'Adam'
        self.loss_fn = torch.nn.MSELoss
        self.input_train = pd.DataFrame(torch.randn(100, self.Nin).numpy())
        self.target_train = pd.DataFrame(torch.randn(100, self.Nout).numpy())
        self.input_test = pd.DataFrame(torch.randn(20, self.Nin).numpy())
        self.target_test = pd.DataFrame(torch.randn(20, self.Nout).numpy())
        self.validation_data = (pd.DataFrame(torch.randn(20, self.Nin).numpy()), pd.DataFrame(torch.randn(20, self.Nout).numpy()))

    def test_configure_FFNN_model_errors(self):
        with self.assertRaises(NeuralNetworkSetupError):
            configure_FFNN_model(0, self.Nout, self.Nlayers, self.Nnodes)
        with self.assertRaises(NeuralNetworkSetupError):
            configure_FFNN_model(self.Nin, 0, self.Nlayers, self.Nnodes)
        with self.assertRaises(NeuralNetworkSetupError):
            configure_FFNN_model(self.Nin, self.Nout, 0, self.Nnodes)
        with self.assertRaises(NeuralNetworkSetupError):
            configure_FFNN_model(self.Nin, self.Nout, self.Nlayers, 0)
        with self.assertRaises(NeuralNetworkSetupError):
            configure_FFNN_model(self.Nin, self.Nout, self.Nlayers, self.Nnodes, lr=0)
        with self.assertRaises(NeuralNetworkSetupError):
            configure_FFNN_model(self.Nin, self.Nout, self.Nlayers, self.Nnodes, optimizer_name='')
        with self.assertRaises(NeuralNetworkSetupError):
            configure_FFNN_model(self.Nin, self.Nout, self.Nlayers, self.Nnodes, loss_fn=None)

    def test_train_FFNN_model_data_errors(self):
        model, optimizer, loss_fn = configure_FFNN_model(self.Nin, self.Nout, self.Nlayers, self.Nnodes)

        with self.assertRaises(NeuralNetworkDataError):
            train_FFNN_model(model, optimizer, loss_fn, pd.DataFrame(), self.target_train, self.input_test, self.target_test, self.validation_data)
        with self.assertRaises(NeuralNetworkDataError):
            train_FFNN_model(model, optimizer, loss_fn, self.input_train, pd.DataFrame(), self.input_test, self.target_test, self.validation_data)
        with self.assertRaises(NeuralNetworkDataError):
            train_FFNN_model(model, optimizer, loss_fn, self.input_train, self.target_train, pd.DataFrame(), self.target_test, self.validation_data)
        with self.assertRaises(NeuralNetworkDataError):
            train_FFNN_model(model, optimizer, loss_fn, self.input_train, self.target_train, self.input_test, pd.DataFrame(), self.validation_data)
        with self.assertRaises(NeuralNetworkDataError):
            train_FFNN_model(model, optimizer, loss_fn, self.input_train.iloc[:50], self.target_train, self.input_test, self.target_test, self.validation_data)
        with self.assertRaises(NeuralNetworkDataError):
            train_FFNN_model(model, optimizer, loss_fn, self.input_train, self.target_train, self.input_test.iloc[:10], self.target_test, self.validation_data)
        with self.assertRaises(NeuralNetworkDataError):
            train_FFNN_model(model, optimizer, loss_fn, self.input_train, self.target_train, self.input_test, self.target_test, (pd.DataFrame(), self.validation_data[1]))
        with self.assertRaises(NeuralNetworkDataError):
            train_FFNN_model(model, optimizer, loss_fn, self.input_train, self.target_train, self.input_test, self.target_test, (self.validation_data[0], pd.DataFrame()))
        with self.assertRaises(NeuralNetworkDataError):
            train_FFNN_model(model, optimizer, loss_fn, self.input_train, self.target_train, self.input_test, self.target_test, (self.validation_data[0].iloc[:10], self.validation_data[1]))

    def test_train_FFNN_model_setup_errors(self):
        model, optimizer, loss_fn = configure_FFNN_model(self.Nin, self.Nout, self.Nlayers, self.Nnodes)

        with self.assertRaises(NeuralNetworkSetupTrainingError):
            train_FFNN_model(model, optimizer, loss_fn, self.input_train, self.target_train, self.input_test, self.target_test, self.validation_data, epochs=0)
        with self.assertRaises(NeuralNetworkSetupTrainingError):
            train_FFNN_model(model, optimizer, loss_fn, self.input_train, self.target_train, self.input_test, self.target_test, self.validation_data, batch_size=0)

if __name__ == '__main__':
    unittest.main()