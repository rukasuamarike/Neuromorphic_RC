# import modules
from scipy import signal
from os.path import join
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# Functions in this module
def _ProcessArguments():
    # process the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-f", "--Netlist", type=str, required=True, help="Netlist file for circuit."
    )

    # get the arguments
    args = ap.parse_args()
    return args


def ReadCompInfo(Netlist):
    ColLbs = ["DEV", "N1", "N2", "VAL"]
    CompInfo = pd.read_csv(
        Netlist,
        delimiter=" ",
        header=None,
        index_col=False,
        comment="*",
        skipinitialspace=True,
    )
    CompInfo.columns = ColLbs
    CompInfo["DEV"] = CompInfo["DEV"].str.upper()
    return CompInfo


def ExtractNetlistSym(DfCompInfo):
    # get the heading names
    Header = list(DfCompInfo.columns)

    # extract voltage source list
    VolInd = np.asarray([ADev[0] == "V" in ADev for ADev in DfCompInfo["DEV"].values])
    if VolInd.any():
        VList = DfCompInfo.loc[VolInd].to_numpy()
    else:
        VList = None

    # extract current source list
    CurInd = np.asarray([ADev[0] == "I" in ADev for ADev in DfCompInfo["DEV"].values])
    if CurInd.any():
        IList = DfCompInfo.loc[CurInd].to_numpy()
    else:
        IList = None

    # extract resistor list
    ResInd = np.asarray([ADev[0] == "R" for ADev in DfCompInfo["DEV"].values])
    if ResInd.any():
        RList = DfCompInfo.loc[ResInd].to_numpy()
    else:
        RList = None

    # extract capacitor list
    CapInd = np.asarray([ADev[0] == "C" for ADev in DfCompInfo["DEV"].values])
    if CapInd.any():
        CList = DfCompInfo.loc[CapInd].to_numpy()
    else:
        CList = None

    # extract memristor list
    MResInd = np.asarray([ADev[:2] == "MR" for ADev in DfCompInfo["DEV"].values])
    if MResInd.any():
        MRList = DfCompInfo.loc[MResInd].to_numpy()
    else:
        MRList = None

    # extract memcapacitor list
    MCapInd = np.asarray([ADev[:2] == "MC" for ADev in DfCompInfo["DEV"].values])
    if MCapInd.any():
        MCList = DfCompInfo.loc[MCapInd].to_numpy()
    else:
        MCList = None

    # extract nodes
    N1 = DfCompInfo["N1"].values
    N2 = DfCompInfo["N2"].values
    NodeList = np.unique([N1, N2]).astype(int)
    NumNodes = len(NodeList) - 1

    # set the return dictionary
    return {
        "Header": Header,
        "VList": VList,
        "IList": IList,
        "RList": RList,
        "CList": CList,
        "MRList": MRList,
        "MCList": MCList,
        "NodeList": NodeList,
        "NumNodes": NumNodes,
        "dt": None,
    }


def ExtractNetlists(NetParms):
    # check the list
    if NetParms["VList"] is not None:
        VList = torch.from_numpy(NetParms["VList"][:, 1:].astype(np.double))
    else:
        VList = None

    if NetParms["IList"] is not None:
        IList = torch.from_numpy(NetParms["IList"][:, 1:].astype(np.double))
    else:
        IList = None

    if NetParms["RList"] is not None:
        RList = torch.from_numpy(NetParms["RList"][:, 1:].astype(np.double))
    else:
        RList = None

    if NetParms["CList"] is not None:
        CList = torch.from_numpy(NetParms["CList"][:, 1:].astype(np.double))
    else:
        CList = None

    if NetParms["MRList"] is not None:
        MRList = torch.from_numpy(NetParms["MRList"][:, 1:].astype(np.double))
    else:
        MRList = None

    if NetParms["MCList"] is not None:
        MCList = torch.from_numpy(NetParms["MCList"][:, 1:].astype(np.double))
    else:
        MCList = None

    if NetParms["MCList"] is not None:
        NodeList = NetParms["NodeList"]
    else:
        NodeList = None

    if NetParms["NumNodes"] is not None:
        NumNodes = NetParms["NumNodes"]
    else:
        NumNodes = None

    dt = NetParms["dt"]

    AllLists = {
        "VList": VList,
        "IList": IList,
        "RList": RList,
        "CList": CList,
        "MRList": MRList,
        "MCList": MCList,
        "NodeList": NodeList,
        "NumNodes": NumNodes,
        "dt": dt,
        "MemResObj": NetParms["MemResObj"],
        "MemCapObj": NetParms["MemCapObj"],
    }

    return AllLists


# MNA class to build Matrix A, vect X, and vector Z
class MNA(nn.Module):
    def __init__(self, NetParms=None, Verbose=False):
        # **********************************************************************
        super(MNA, self).__init__()

        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "MNA::__init__()"

        # **********************************************************************
        # Save the network parameters
        # **********************************************************************
        self.NetParams = NetParms
        self.VList = NetParms["VList"]
        self.IList = NetParms["IList"]
        self.RList = NetParms["RList"]
        self.CList = NetParms["CList"]
        self.MRList = NetParms["MRList"]
        self.MCList = NetParms["MCList"]
        self.NodeList = NetParms["NodeList"]
        self.NumNodes = NetParms["NumNodes"]
        self.NumVs = len(self.VList) if self.VList is not None else 0
        self.NumIs = len(self.IList) if self.IList is not None else 0
        self.NumR = len(self.RList) if self.RList is not None else 0
        self.NumMR = len(self.MRList) if self.MRList is not None else 0
        self.NumC = len(self.CList) if self.CList is not None else 0
        self.NumMC = len(self.MCList) if self.MCList is not None else 0
        self.dt = NetParms["dt"]
        self.Verbose = Verbose
        # init params
        self.MatrixSize = self.NumNodes + self.NumVs
        self.MatrixA = torch.zeros(
            (self.MatrixSize, self.MatrixSize), dtype=torch.float64
        )
        self.VectorX = torch.zeros(self.MatrixSize, dtype=torch.float64)
        self.VectorZ = torch.zeros(self.MatrixSize, dtype=torch.float64)
        # self._ConvertSymAllLists()
        self._BuildMatrixA_VectorZ()
        self._Build_VectorZ()
        # self._BuildVectorZ_VList(self.NumNodes)
        # self._BuildVectX()
        if self.Verbose:
            print(
                f"\n==> MNA Class ...\n...Vs = {self.NumVs}, Is = {self.NumIs}, R = {self.NumR}, MR = {self.NumMR}, C = {self.NumC}, MC = {self.NumMC}\n...matrix size = {self.MatrixSize} x {self.MatrixSize}\n...Matrix A = {self.MatrixA.shape}, Vector X = {self.VectorX.shape}, Vector Z = {self.VectorZ.shape}"
            )

    def tupleList(self, listid):
        print(self.NetParams[listid][0].tolist())
        merged_tuples = [
            (i[0].int(), i[1].int(), i[2].item()) for i in self.NetParams[listid]
        ]
        print(merged_tuples)
        return merged_tuples

    def _BuildMatrixA_VectorZ(self):
        dt = 1e-3
        mergedlist = self.tupleList("RList")  # pattern: 1/R
        for i in range(len(mergedlist) - 1):
            self.MatrixA[i][i] += sum(
                [1 / dev[2] for dev in mergedlist if dev[0] == i + 1 or dev[1] == i + 1]
            )
        for i in mergedlist:
            offset = 1
            if i[0] > 0 and i[1] > 0:
                if not i[0] == i[1]:
                    print(i[0], i[1])
                    self.MatrixA[i[0] - offset][i[1] - offset] += -(1 / i[2])
                    self.MatrixA[i[1] - offset][i[0] - offset] += -(1 / i[2])

        mergedvlist = self.tupleList("VList")
        offset = self.MatrixSize - self.NumVs
        for idx, i in enumerate(mergedvlist):
            if i[0] == 0 or i[1] == 0:
                vval = max(i[0], i[1])
                self.MatrixA[vval - 1][offset + idx] += 1 if i[1] == 0 else -1
                self.MatrixA[offset + idx][vval - 1] += 1 if i[1] == 0 else -1
            else:
                self.MatrixA[i[0] - 1][offset + idx] += 1
                self.MatrixA[offset + idx][i[0] - 1] += 1
                self.MatrixA[i[1] - 1][offset + idx] += -1
                self.MatrixA[offset + idx][i[1] - 1] += -1

        # mergedmclist = self.tupleList("MCList") # pattern: 1/R
        # for i in range(len(mergedmclist) - 1):

        #     self.MatrixA[i][i] += sum(
        #         [dev[2]/dt for dev in mergedmclist if dev[0] == i + 1 or dev[1] == i + 1]
        #     )
        # for i in mergedmclist:
        #     offset = 1
        #     if i[0] > 0 and i[1] > 0:
        #         if not i[0] == i[1]:
        #             print(i[0], i[1])
        #             self.MatrixA[i[0] - offset][i[1] - offset] += -(1 / i[2])
        #             self.MatrixA[i[1] - offset][i[0] - offset] += -(1 / i[2])

    def _Build_VectorZ(self):
        mergedvlist = self.tupleList("VList")
        test = sorted([(max(i[0], i[1]), i[2]) for i in mergedvlist])
        offset = self.MatrixSize - self.NumVs
        for idx, i in enumerate(test):
            self.VectorZ[offset + idx] += i[1]
        print("Voltages", self.VectorZ)
        print(self.MatrixA)
        mergedilist = self.tupleList("IList")
        for i in mergedilist:
            if i[0] == 0 or i[1] == 0:
                ival = max(i[0], i[1])
                self.VectorZ[ival - 1] += i[2] if i[0] == 0 else -i[2]
            else:
                self.VectorZ[i[0] - 1] += -i[2]
                self.VectorZ[i[1] - 1] += i[2]

    def _Get_VectorX(self):
        return torch.linalg.solve(self.MatrixA, self.VectorZ)

    def _Update_VectorZ(self):
        self.VectorZ = self._Get_VectorX()


def Signal(Type, Amp, Freq, Offset, NoCycles=1, NpCycle=1000, Verbose=None):
    # set the function name
    FunctionName = "Signal()"

    # display the message
    if Verbose is not None:
        Msg = "...%-25s: input sine signal ..." % (FunctionName)
        print(Msg)

    # set the information of the signal
    Period = 1 / Freq
    T = NoCycles * Period
    Num = NoCycles * NpCycle

    # check the signal
    if Type == "Square":
        t = np.linspace(0, T, num=Num)
        v = torch.from_numpy(Amp * signal.square(2 * np.pi * Freq * t) + Offset)
        t = torch.from_numpy(t)
    else:
        t = torch.linspace(0, T, steps=Num)
        v = Amp * torch.sin(2 * torch.pi * Freq * t) + Offset
    return v, t


if __name__ == "__main__":
    # process the arguments
    args = _ProcessArguments()

    # read components from the netlist
    DfCompInfo = ReadCompInfo(args.Netlist)
    print(" ")
    print(args.Netlist)
    print(DfCompInfo)

    # create input signal
    Type = "Square"
    # Type    = "Sine"
    Amp = 2.0
    Freq = 1
    Offset = 0
    Cycles = 2
    v, t = Signal(Type, Amp, Freq, Offset, NoCycles=Cycles, Verbose=True)

    # extract the component list
    DefaultDt = t[1] - t[0]
    SymFlag = True
    NetParms = ExtractNetlistSym(DfCompInfo)
    NetParms["dt"] = DefaultDt
    NetParms["MemResObj"] = None
    NetParms["MemCapObj"] = None
    NetParms = ExtractNetlists(NetParms)
    # print(NetParms)

    # Create MNA object and build Matrix A, Vector X, and Vector Z
    print(" ")
    Verbose = True
    MNAObj = MNA(NetParms=NetParms, Verbose=Verbose)
    MNAObj
