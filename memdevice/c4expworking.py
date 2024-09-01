import numpy as np
import torch, os
import torch.nn as nn

"""
Class for the memcapacitive model Biolek C4 - bipolar with threshold model
    D. Biolek, M. Di Ventra, and Y. V. Pershin, "Reliable SPICE simulations of
    memristors, memcapacitors and meminductors," Radioengineering, vol. 22,
    no. 4, pp. 945-968, 2013.
"""


def to_torch_column_stack(list_of_nums):

    if not list_of_nums:
        raise ValueError("Input list cannot be empty.")

    # Check if all elements have the same length
    if any(len(sublist) != len(list_of_nums[0]) for sublist in list_of_nums[1:]):
        raise ValueError(
            "All sub-lists within the input list must have the same length."
        )

    # Convert sub-lists (if any) to tensors and stack them
    tensors = [torch.tensor(sublist) for sublist in list_of_nums]
    return torch.stack(tensors, dim=1)


def expand_constant_stack(constant_stack, input_size):

    # Get the number of constants and batch size
    num_constants = constant_stack.shape[1]
    batch_size = constant_stack.shape[0]

    # Create a tensor with ones for broadcasting
    ones_tensor = torch.ones((batch_size, input_size))

    # Expand each constant value across the desired width
    expanded_constants = constant_stack.unsqueeze(1).repeat(1, input_size, 1)

    # Multiply with ones for element-wise replication
    return expanded_constants * ones_tensor


class ConstPreset:
    def __init__(
        self,
        size=1,
        a=2.5e3,
        b=1.0,
        c=1.0,
        Beta=70e-6,
        b1=10e-3,
        b2=10e-6,
        logConst=0.0,
    ):
        # constants for decay effect
        # self.a = 5e2
        # self.b = 1.0
        # self.c = 0.2
        self.size = size
        self.a = torch.tensor(a, dtype=torch.float64)
        self.b = torch.tensor(b, dtype=torch.float64)
        self.c = torch.tensor(c, dtype=torch.float64)

        # specific constants for the model
        self.Beta = torch.tensor(Beta, dtype=torch.float64)
        self.b1 = torch.tensor(b1, dtype=torch.float64)
        self.b2 = torch.tensor(b2, dtype=torch.float64)
        self.LogConst = torch.tensor(logConst, dtype=torch.float64)
        self.vector = torch.stack(
            [self.a, self.b, self.c, self.Beta, self.b1, self.b2, self.LogConst]
        )


class FactorConstPreset:  # vals for resetting
    def __init__(self, size=1, Cmin=1e-12, Cmax=100e-12, InitVals=0.0, Vth=0.0):
        # constants for decay effect
        self.size = (size,)
        self.Cmin = torch.tensor(Cmin, dtype=torch.float64)
        self.Cmax = torch.tensor(Cmax, dtype=torch.float64)
        self.Vth = torch.tensor(Vth, dtype=torch.float64)
        self.InitVals = torch.tensor(InitVals, dtype=torch.float64)

        # these values are specific to the device
        self.vector = torch.stack([self.Cmin, self.Cmax, self.Vth, self.InitVals])


class BiolekC4Memcapacitor(nn.Module):
    def __init__(
        self,
        size=1,
        Size=None,
        InitVals=0.0,
        Percent=0.0,
        DecayEffect=False,
        Theta=1.0,
        Vth=0.0,
        Verbose=False,
    ):
        FunctionName = "Biolek::__init__()"

        self.ModelName = "BiolekC4"

        # perecent
        self.Percent = Percent
        self.size = size
        self.constDecay = ConstPreset(size)
        self.factorySetting = FactorConstPreset(size=size, InitVals=InitVals, Vth=Vth)
        # save parameters
        self.DecayEffect = DecayEffect
        self.Verbose = Verbose
        self.InitVals = self.factorySetting.InitVals
        # constants for the Biolek Bipolar Threshold model - C4
        self.Cmin = self.factorySetting.Cmin
        self.Cmax = self.factorySetting.Cmax
        # the threshold voltage for crossbar classifier; it is not needed for
        # reservoir
        self.Vth = self.factorySetting.Vth
        # charge
        self.Q = torch.tensor(0.0, dtype=torch.float64)
        # log constant
        self.LogConst = self.constDecay.LogConst
        # state variable
        self.Rho = self._CalRho(self.InitVals)
        self.dRho = torch.tensor(0.0, dtype=torch.float64)

        # these values are specific to the device
        self.Cmin = self.Cmin
        self.Cmax = self.Cmax
        tempvector = torch.stack([self.Rho, self.dRho, self.Q])
        self.vector = torch.cat(
            [tempvector.view(-1), self.factorySetting.vector.view(-1)], dim=0
        )

        self.vector = self.vector.reshape(-1, 1)  # Reshape to column
        print(self.vector.shape)  # torch.Size([6, 1])
        print(self.vector)

        # set the inputs and outputs
        if Size is not None:
            (self.Inputs, self.Outputs) = Size

        # display the message
        if self.Verbose:
            Msg = "\n==> BiolekC4 Memcapacitor Model ..."
            print(Msg)

            Msg = "...%-25s: Cmin = %.8g, Cmax = %.8g, Size = %s" % (
                FunctionName,
                self.Cmin,
                self.Cmax,
                str(Size),
            )
            print(Msg)

            Msg = "...%-25s: Init = %.2g, Rho = %s" % (
                FunctionName,
                self.InitVals,
                self.Rho,
            )
            print(Msg)

            Msg = "...%-25s: Vth = %.g, Verbose = %s, Decay Effect = %s" % (
                FunctionName,
                self.Vth,
                str(self.Verbose),
                str(self.DecayEffect),
            )
            print(Msg)

    # private functions for the class
    def _CalRho(self, InitVals):
        # calculate delta C
        DeltaC = self.Cmax - self.Cmin
        return torch.add(self.Cmin, torch.mul(InitVals, DeltaC))

    def _Torch_logaddexp(self, x1, x2):
        diff = torch.min(x2 - x1, x1 - x2)
        return torch.max(x1, x2) + torch.log1p(torch.exp(diff))

    # private methods step function
    def _stps(self, x, b):
        Val = torch.exp(-self._Torch_logaddexp(self.LogConst, torch.div(-x, b)))
        return Val

    # private method absolute function
    def _abss(self, x, b):
        Const1 = self._stps(x, b) - self._stps(-x, b)
        Val = torch.mul(x, Const1)
        return Val

    # private method function f(vc)
    def _fs(self, v, b):

        Const1 = self._abss(v + self.Vth, b) - self._abss(v - self.Vth, b)
        Const2 = torch.mul(0.5, Const1)
        Const3 = v - Const2
        Val = torch.mul(self.constDecay.Beta, Const3)
        return Val

    # private method function ws
    def _ws(self, x, v):

        Const1 = self._stps(1.0 - torch.div(x, self.Cmax), self.constDecay.b2)
        Const2 = torch.mul(self._stps(v, self.constDecay.b1), Const1)
        Const3 = self._stps(torch.div(x, self.Cmin) - 1, self.constDecay.b2)
        Const4 = torch.mul(self._stps(-v, self.constDecay.b1), Const3)
        Val = torch.add(Const3, Const4)
        return Val

    def _CalI(self, PrevQ, dt=0.0):
        FunctionName = "_CalI()"

        # check the delta t
        if dt == 0.0:
            # set the error message
            ErrMsg = "%s: invalid dt = <%.8g>" % (FunctionName, dt)
            raise ValueError(ErrMsg)

        Val = torch.div(self.Q - PrevQ, dt)
        return Val

    # get methods for the class
    def _CheckRho(self, RhoVals):
        # check the condition to get mask tensor
        MaskIndices = RhoVals.le(self.Cmin)
        Const1 = torch.mul(RhoVals - self.Cmin, self.constDecay.a)
        Const2 = torch.add(torch.mul(RhoVals, self.constDecay.c), self.constDecay.b)
        DecayFactors = torch.div(Const1, Const2)
        DecayFactors[MaskIndices] = 0.0
        return DecayFactors

    def _EvaldRho(self, Vin, Rho):
        # check the flag
        if self.DecayEffect:
            DecayFactors = self._CheckRho(Rho)
            Const1 = torch.mul(self._fs(Vin, self.constDecay.b1), self._ws(Rho, Vin))
            return Const1 - DecayFactors
        else:
            return torch.mul(self._fs(Vin, self.constDecay.b1), self._ws(Rho, Vin))

    def GetdMemCap(self, Vin):
        return self._EvaldRho(Vin, self.Rho)

    def GetMemCap(self):
        return self.Rho

    def GetCminCmax(self):
        return self.Cmin, self.Cmax

    def GetChargeQ(self):
        return self.Q

    def GetVth(self):
        return self.Vth.data

    def SetVth(self, Vth=0.0):
        self.Vth = self.Vth * 0.0 + Vth

    def GetModelName(self):
        return self.ModelName

    def UpdateRho(self, V, dt):
        if self.size == 1:
            self.UpdateSingleRho(V, dt)
        else:
            self.UpdateMultiRho(V, dt)

    def UpdateSingleRho(self, V, dt):
        # updating Rho
        # keep and old copy of Rho
        OldRho = self.Rho
        # get delta Rho
        dRho = self.GetdMemCap(V)
        self.Rho = torch.add(OldRho, dRho * dt)

        # check for boundaries
        print("CHECK", self.Rho.shape, self.Cmin.shape)
        # turns into torch.lt
        if self.Rho < self.Cmin:
            self.Rho = self.Cmin
        elif self.Rho > self.Cmax:
            self.Rho = self.Cmax
        # calculating the charge
        self.Q = torch.mul(self.Rho, V)

    def UpdateMultiRho(self, V, dt):
        # updating Rho
        OldRho = self.Rho
        # get delta Rho
        dRho = self.GetdMemCap(V)
        self.Rho = torch.add(OldRho, dRho * dt)

        # check for boundaries
        if self.Rho < self.Cmin:
            self.Rho = self.Cmin
        elif self.Rho > self.Cmax:
            self.Rho = self.Cmax
        # calculating the charge
        self.Q = torch.mul(self.Rho, V)

    # TODO: reset/ init function
    def reset_vals(self, InitVals):
        self.GetInitRho(InitVals)

    def GetInitRho(self, InitVals):
        FunctionName = "Biolek::GetInitRho()"

        # display the message
        if self.Verbose:
            Msg = "...%-25s: getting initial values of Rho..." % (FunctionName)
            print(Msg)

        return self._CalRho(self.factorySetting.InitVals)

    def RandomInitCVals(self, InitVals):
        return self.GetInitRho(InitVals)

    # calculate dC = dRho * dt
    def GetdC(self, Vin, Rho, dt):
        dRho = self._EvaldRho(Vin, Rho)
        Val = Torch.mult(dRho, dt)
        return Val

    def GetdW(self, Vin, Rho):
        return self._EvaldRho(Vin, Rho)

    # TODO: turn into getvals with VinVals, dt
    # Vins are from prev MNA
    # TODO: add the update vals after I get the MNA vals
    def ComputeCap(self, Vs, dt):
        # signal sources
        print("\n SIM", Vs)
        size = Vs.shape[0]
        # convert signal source values to tensors
        # Vs = torch.from_numpy(Vs)

        # initializing charge Q and current
        i = torch.from_numpy(np.zeros(size))
        QCal = torch.from_numpy(np.zeros(size))
        MemCap = torch.from_numpy(np.zeros(size))

        # calculate the charge QCal and current i
        MemCap = self.GetMemCap()
        self.UpdateRho(Vs, dt)

        MemCap = self.GetMemCap()

        QCal = self.GetChargeQ()

        i = self._CalI(QCal, dt)

        # return MemCap,
        return (QCal, MemCap, i)


class DoubleMemcapacitor(nn.Module):
    def __init__(self, InitVals1=0.0, InitVals2=0.0, **kwargs):
        super(DoubleMemcapacitor, self).__init__()

        # Create two instances of BiolekC4Memcapacitor
        self.memcapacitor1 = BiolekC4Memcapacitor(InitVals=InitVals1, **kwargs)
        self.memcapacitor2 = BiolekC4Memcapacitor(InitVals=InitVals2, **kwargs)

    def GetMemCap(self):
        # Return the two memcapacitor values as a 2-column matrix
        return torch.stack(
            (self.memcapacitor1.GetMemCap(), self.memcapacitor2.GetMemCap()), dim=1
        )

    def GetdMemCap(self, Vin):
        # Return the two memcapacitor dRho values as a 2-column matrix
        return torch.stack(
            (self.memcapacitor1.GetdMemCap(Vin), self.memcapacitor2.GetdMemCap(Vin)),
            dim=1,
        )

    # You can add more methods as needed for your specific application


# Example usage:
if __name__ == "__main__":
    # Initialize DoubleMemcapacitor
    double_memcapacitor = DoubleMemcapacitor(
        InitVals1=0.0, InitVals2=0.0, DecayEffect=True, Theta=0.5, Vth=0.1, Verbose=True
    )

    # Accessing memcapacitor values
    print("Memcapacitor 1:", double_memcapacitor.memcapacitor1.GetMemCap())
    print("Memcapacitor 2:", double_memcapacitor.memcapacitor2.GetMemCap())

if __name__ == "__main__":
    from os.path import join
    import sys
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({"figure.autolayout": True})
    from scipy.interpolate import interp1d

    # check the platform to import mem-models

    # set the function name
    FunctionName = "Biolek::main()"

    # set the model name
    ModelName = "BiolekC4"

    SignalType = "Sine"
    Ampl = 0.55
    Freq = 50e3
    Offset = 0
    N = 3
    NumStep = 1e3
    dt = 2.0010e-08
    Verbose = True
    ts = np.linspace(0, 2 / Freq, 2000)

    # set the input signals
    # Input = (SignalType=SignalType, Ampl=Ampl,Offset=Offset, Freq=Freq, Sample=NumStep, NumCycles=N, Verbose=Verbose)

    # create the sine wave
    Vs = Ampl * np.sin(np.pi * 2 * Freq * ts)

    # parameters the memcapacitor model
    InitVals = 0.0
    DecayEffect = True
    Theta = 0.5
    Vth = 0.1
    MemcapDevice = BiolekC4Memcapacitor(
        size=1,
        InitVals=InitVals,
        DecayEffect=DecayEffect,
        Theta=Theta,
        Vth=Vth,
        Verbose=True,
    )

    # get the min and max values
    Cmin, Cmax = MemcapDevice.GetCminCmax()
    print("Cmin = ", Cmin)
    print("Cmax = ", Cmax)
    Tau = 0.05 * Cmin
    print("Tau  = ", Tau)

    # compute values
    DictVals = {
        "QCal": [],
        "MemC": [],
        "i": [],
    }

    for v in Vs:
        vv = torch.tensor([v], dtype=torch.float64)
        out = MemcapDevice.ComputeCap(vv, dt)
        DictVals["QCal"].append(out[0].item())
        DictVals["MemC"].append(out[1].item())
        DictVals["i"].append(out[2].item())

    # extract information
    QCal = DictVals["QCal"]
    MemCap = DictVals["MemC"]
    i = DictVals["i"]

    # plot results
    # set the time scale
    TimeScale = 1e3
    ts *= TimeScale
    TimeLabel = "Time (ms)"
    LineWidth = 1.5
    FontSize = 14
    font = {"family": "Times New Roman", "size": FontSize}
    plt.rc("font", **font)  # pass in the font dict as kwargs
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"

    Fig = plt.figure("Q-V Response")
    plt.title("Biolek Q-V Plot")
    plt.grid(linestyle="dotted")
    # plt.plot(VSpice, QSpice, "-", label="Q-V Spice", linewidth=LineWidth)
    plt.plot(Vs, QCal, "-", color="b", label="Q-V Cal", linewidth=LineWidth)
    plt.xlabel("Vin (V)")
    plt.ylabel("Charge (q)")
    plt.axis("tight")
    # save the figure
    plt.savefig("memc4")
    # get the figure handle
    print(MemCap)
    Fig = plt.figure("C-V")
    plt.title("Biolek C-V Plot")
    plt.grid(linestyle="dotted")
    plt.plot(
        Vs, np.asarray(MemCap) * 1e12, "-", color="b", label="C-V", linewidth=LineWidth
    )
    plt.xlabel("Vin (V)")
    plt.ylabel("Capacitance (pF)")
    plt.axis("tight")

    plt.show()
