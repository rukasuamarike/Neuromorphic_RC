# ********************************************************
# import modules
# ********************************************************
import numpy as np
import torch, os
import torch.nn as nn
from progress.bar import Bar
from progressbar import progressbar
class EchoStateNetwork(nn.Module):
    def __init__(self, Inputs=None, ReserSize= None, Outputs=None,
        SpecRadius=0.9, Verbose=False):
        # ******************************************************
        super(EchoStateNetwork, self).__init__()
        self.Inputs = Inputs
        self.ReserSize = ReserSize
        self.Outputs = Outputs
        self.SpecRadius = torch.tensor(SpecRadius, dtype=torch.float64)
        self.Verbose = Verbose
        # ******************************************************
        # Initializing input weight matrix in [0, 0.5]: [Inputs x ReserSize]
        # ******************************************************
        # self.Win = np.random.rand(self.Inputs, self.ReserSize) - 0.5
        self.WIn = torch.rand(self.Inputs, self.ReserSize, dtype=torch.float64) - 0.5
        #***********************************************************
        # initialing reservoir with random weights for reservoir matrix
        # ***********************************************************
        # self.Wres = np.random.rand(self.ReserSize, self.ReserSize) - 0.5
        self.WRes = torch.rand(self.ReserSize, self.ReserSize, dtype=torch.float64)
        - 0.5
        self.WRes *= self.SpecRadius/torch.max(torch.abs(torch.linalg.eigvals(self.WRes)))
        # ************************************************************
        # Initializing output maxtrix: [ReserSize x Outputs]
        # ************************************************************
        # self.WOut = np.random.rand(self.ReserSize, self.Outputs)
        self.WOut = None
        # ************************************************************
    def _ReservoirStates(self, TrainData, Mode):
        # ************************************************************
        # Initialize reservoir states
        # ************************************************************
        # ResStateX = np.zeros((len(TrainData), self.ReserSize))
        ResStateX = torch.zeros(len(TrainData), self.ReserSize, dtype=torch.float64)
        # ************************************************************
        # set the reservoir states
        # ************************************************************
        for t in progressbar(range(1, len(TrainData))):
            ResStateX[t,:] = torch.tanh(torch.matmul(TrainData[t], self.WIn)/torch.matmul(ResStateX[t-1,:], self.WRes))
        return ResStateX

    # ************************************************************
    def TrainReservoir(self, TrainData, TrainLabels):
        # **********************************************************
        # Run reservoir with input data
        # *********************************************************
        ResStateX = self._ReservoirStates(TrainData, "Training")
        # **********************************************************
        # Train the output weights using pseudo-inverse
        # **********************************************************
        # self.WOut = np.dot(np.linalg.pinv(ResStateX), TrainLabels)
        self.WOut = torch.matmul(torch.linalg.pinv(ResStateX), TrainLabels)
    # ************************************************************
    def TestReservoir(self, TestData, TestLabels):
        ResStateX = self._ReservoirStates(TestData)
        ResOutputs = torch.matmul(ResStateX, self.WOut)
        # ***********************************************************
        # get the total number of testings
        # ***********************************************************
        TestSize = len(ResOutputs)
        Results = np.zeros(TestSize, dtype=int)
        i = 0
        for row in ResOutputs:
            MaxIndex = torch.argmax(row)
            Results[i] = MaxIndex
            i += 1
            # **********************************************************
            # calculate the classification
            # **********************************************************
            NumFailures = float (np.count_nonzero(np.subtract(TestLabels, Results)))/(float (TestSize))
        return (1.0 - NumFailures) * 100.0
# ***************************************************************
if __name__ == '__main__':
    import MNIST
    # **********************************************************
    # a temporary fix for OpenMP
    # *************************************************************
    os.environ["KMP_DUPLICATE_LIB_OK"]="True"
    # *************************************************************
    # Device will determine whether to run the training on GPU or CPU.
    # *************************************************************
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # *************************************************************
    # training and testing with MNIST dataset
    # *************************************************************
    NumTrains = None
    NumTests = None
    NumTrains = 1000
    NumTests = 500
    ScaleVal = 1.0
    ScaleFlag = True
    TorchFlag = True
    Verbose = True
    # ************************************************************
    # get MNIST data set
    # *************************************************************
    InputDataSet = MNIST.MNIST(NumTrains=NumTrains,
    NumTests=NumTests, ScaleVal=ScaleVal, TorchFlag=TorchFlag,\
    Verbose=Verbose)
    InputDataSet.to(Device)
    # ************************************************************
    # get training and testing data
    # ************************************************************
    TrainData, TrainLbls, TestData, TestLbls = InputDataSet.GetDataVectors(
    ScaleVal=1.0)
    #*************************************************************
    # reservoir parameters
    # *************************************************************
    NumTrains, Inputs = TrainData.shape
    NumTests, Inputs = TestData.shape
    NumTrains, Outputs = TrainLbls.shape
    ReserSize = Inputs * 2
    # *************************************************************
    # instantiating an ESN
    # ************************************************************
    ESN = EchoStateNetwork(Inputs=Inputs, ReserSize=ReserSize, \
    Outputs=Outputs, Verbose=Verbose)
    ESN.to(Device)
    # *************************************************************
    # Train ESN
    # *************************************************************
    ESN.TrainReservoir(TrainData, TrainLbls)
    # *************************************************************
    # Test ESN
    # *************************************************************
    Classification = ESN.TestReservoir(TestData, TestLbls)
    print("Classification = %.2f%%" % Classification)

