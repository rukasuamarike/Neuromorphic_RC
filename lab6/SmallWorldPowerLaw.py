# ******************************************************************************
# import modules
# ******************************************************************************
import networkx as nx
import numpy as np
import time
import pandas as pd

# ******************************************************************************
# import pytorch package
# ******************************************************************************
import torch
import torch.nn as nn

# ******************************************************************************
class SmallWorldPowerLaw(nn.Module):
    """
    Creating a small-world graph: starting with a 2D grid mesh and replacing each
    local link with a global link using a power-law decay distribution
                P = l^-alpha
        * L             : size of the network (MxN) for a grid graph
        * D             : is the dimension of a grid graph
        * Alpha         : the decay exponent: Alpha < D + 1
        * Beta          : is the percentage of replacing local links
                                Beta = 0        -> regular network
                                Beta = 100.0    -> completely random network
        * Delta         : number of extra links to be added
        * Gamma         : percent of removed links
        * BoundaryFlag  : for boundary condition of a grid graph
        * NumNodes      : number of nodes for a regular random graph
        * Degree        : node degree for a regular random graph
        * MaxTries      : number of tries if the network is disconnected
    """
    def __init__(self, InitGraph="Grid_Graph", InitGraphObject=None, L=None, \
            Alpha=0.0, Beta=0.0, Delta=0.0, Gamma=0.0, BoundaryFlag=False, \
            NumNodes=0, Degree=0, GndNodeFlag=False, Verbose=False, MaxTries=3):
        # **********************************************************************
        # set the dimension of the super class
        # **********************************************************************
        super(SmallWorldPowerLaw, self).__init__()

        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "SWPL::__init__()"

        # **********************************************************************
        # reset the graph name
        # **********************************************************************
        self.GraphName = "SmallWorldPowerLaw"

        # **********************************************************************
        # save the verbose flag
        # **********************************************************************
        self.Verbose = Verbose

        # **********************************************************************
        # display the message
        if self.Verbose:
            # **********************************************************************
            # display the information
            Msg = "\n==> Instantiating <%s>..." % (self.GraphName)
            print(Msg)

        # **********************************************************************
        # save the parameters
        # **********************************************************************
        self.InitGraphName = InitGraph
        self.InitGraphObject = InitGraphObject
        self.L            = L
        self.D            = 0
        self.Alpha        = Alpha
        self.Beta         = Beta
        self.Delta        = Delta
        self.Gamma        = Gamma
        self.BoundaryFlag = BoundaryFlag
        self.GndNodeFlag  = GndNodeFlag
        self.InOutDegFlag = False
        self.NumNodes     = NumNodes
        self.Degree       = Degree
        self.MaxTries     = MaxTries

        # **********************************************************************
        # get the rows and colums
        # **********************************************************************
        self.Rows, self.Cols = self.L

        # **********************************************************************
        # check the parameters
        # **********************************************************************
        self._CheckParameters()

        # **********************************************************************
        # reset variables
        # **********************************************************************
        self.InDegree       = 0
        self.OutDegree      = 0
        self.InitLinks      = 0
        self.NumRemove      = 0
        self.NumReplace     = 0

        # **********************************************************************
        # setting wire length and wire resistance
        # **********************************************************************
        self.Rnw_Max         = 200.0
        self.TotalWireLen    = 0.0
        self.TotalRes        = 0.0

        # **********************************************************************
        # variables for wiring costs
        # **********************************************************************
        self.NodeMatrix     = None
        self.NodeCoordinates = None

        # **********************************************************************
        # reset variables
        # **********************************************************************
        self.IndexMatrix    = None

        # **********************************************************************
        # the memristor size
        # **********************************************************************
        """
            Dongale, T. D., et al. "Investigation of process parameter variation
            in the memristor based resistive random access memory (RRAM): Effect
            of device size variations." Materials Science in Semiconductor
            Processing 35 (2015): 174-180.
            between 5nm - 50nm
        """
        # **********************************************************************
        self.DeviceLength   = 10e-9
        self.WireScale      = 1.0
        self.NanoWireScale  = self.DeviceLength * self.WireScale

        # **********************************************************************
        # the nanowire resistance
        # **********************************************************************
        """
            Selzer, Franz, et al. "Electrical limit of silver nanowire electrodes:
            Direct measurement of the nanowire junction resistance." Applied Physics
            Letters 108.16 (2016): 163302.
            Rnw = (4.96 ± 0.18) ohms/micro.
        """
        # **********************************************************************
        # the minimum wire resistance is 0.25 * 2.0 ohms/ micro
        # **********************************************************************
        self.NanoWireResistance = 5.0
        self.OneMicroMeter      = 1e-6
        self.RWmin              = self.NanoWireScale * self.NanoWireResistance / \
                self.OneMicroMeter

        # **********************************************************************
        # create a connected graph
        # **********************************************************************
        self._CreateSWGraph()

        # **********************************************************************
        # calculating network wire length
        # **********************************************************************
        self._CalculateNetworkWireLength()

        # **********************************************************************
        # get the node degree
        # **********************************************************************
        self.NodeDegree = self._NodeDegree()

        # **********************************************************************
        # get the number of edges
        # **********************************************************************
        self.NumEdges = self.SWGraphObject.number_of_edges()

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # display the information
            Msg = "...%-25s: Start Graph = <%s>, Result Graph = <%s>" % (FunctionName, \
                    self.InitGraphName, self.GraphName)
            print(Msg)

            # display the information
            Msg = "...%-25s: L = %s, D = %d, Alpha = %.8g, Beta = %.8g, Delta = %d, Gamma = %.8g" % \
                    (FunctionName, str(self.L), self.D, self.Alpha, self.Beta, \
                    self.Delta, self.Gamma)
            print(Msg)

            # display the information
            Msg = "...%-25s: Nodes = %d, Edges = %d, Node degree = %d, Tries = %d" % \
                    (FunctionName, self.Nodes, self.NumEdges, self.NodeDegree, self.MaxTries)
            print(Msg)

            # display the information
            Msg = "...%-25s: Init Links = %d, Removed Links = %d, Replaced Links = %d" % \
                    (FunctionName, self.InitLinks, self.NumRemove, self.NumReplace)
            print(Msg)

            # display the information
            Msg = "...%-25s: (Delta) Added Links = %s" % (FunctionName, str(self.Delta))
            print(Msg)

            Msg = "...%-25s: Total wire length = %.5g" % (FunctionName, self.TotalWireLen)
            print(Msg)

    # **************************************************************************
    def _ResetWireInfo(self):
        self.TotalWireLen = 0.0
        self.TotalRes = 0.0

    # **************************************************************************
    def _CheckParameters(self):
        # set the function name
        FunctionName = "SWPL::_CheckParameters()"

        # check Beta
        if self.Beta < 0:
            # format error message
            ErrMsg = "%s: <Beta = %.5g> => invalid" % (FunctionName, self.Beta)
            raise ValueError(ErrMsg)

        # check Alpha
        if self.Alpha < 0.0:
            # format error message
            ErrMsg = "%s: <Alpha = %.5g> => invalid" % (FunctionName, self.Alpha)
            raise ValueError(ErrMsg)

        # check the starting graph
        if self.InitGraphName == "Grid_Graph":
            # display the message
            if self.Verbose:
                # display the information
                Msg = "...%-25s: checking the parameters..." % (FunctionName)
                print(Msg)

            # check the dimension
            if self.L is None:
                # format error message
                ErrMsg = "%s: <L = %s> => invalid" % (FunctionName, str(self.L))
                raise ValueError(ErrMsg)

            # check the dimension
            self.D = len(self.L)

            # # check the condtion for alpha
            # if self.Alpha > (self.D + 3):
            #     # format error message
            #     ErrMsg = "%s: Alpha = %.5g >  D + 1 = %.5g> => invalid" % (FunctionName, \
            #             self.Alpha, self.D + 1)
            #     raise ValueError(ErrMsg)

        # check the starting graph
        elif self.InitGraphName == "Random_Regular":
            # check the number of nodes
            if self.NumNodes == 0:
                # format error message
                ErrMsg = "%s: Nodes = <%d> => invalid" % (FunctionName, self.NumNodes)
                raise ValueError(ErrMsg)

            # check the number of degree
            if self.Degree == 0:
                # format error message
                ErrMsg = "%s: Nodes = <%d> => invalid" % (FunctionName, self.Degree)
                raise ValueError(ErrMsg)
        else:
            # format error message
            ErrMsg = "%s: <%s> => Unknow graph type" % (FunctionName, self.InitGraphName)
            raise ValueError(ErrMsg)

    # **************************************************************************
    def _CreateEdgeList(self):
        # set the function name
        FunctionName = "SWPL::_CreateEdgeList()"

        # display the message
        if self.Verbose:
            # display the information
            Msg = "...%-25s: creating a list of edges..." % (FunctionName)
            print(Msg)

        # reset the list of edges
        EdgeList = []

        # set the connection matrix
        NumNodes    = self.Rows * self.Cols
        self.ConnectionMatrix   = np.zeros((NumNodes, NumNodes), dtype=bool)

        # generate edge list
        for i in range(self.Rows):
            for j in range(self.Cols):
                # chech the row number
                if i + 1 < self.Rows:
                    # set the nodes a and b
                    a   = self.NodeMatrix[i,j]
                    b   = self.NodeMatrix[i+1,j]

                    # save the edge
                    EdgeList.append([a, b])

                    # save the connection matrix
                    self.ConnectionMatrix[a,b]  = True
                    self.ConnectionMatrix[b,a]  = True

                    # EdgeList.append([self.NodeMatrix[i,j], self.NodeMatrix[i+1,j]])

                # check the column number
                if j + 1 < self.Cols:
                    # set the nodes a and b
                    a   = self.NodeMatrix[i,j]
                    b   = self.NodeMatrix[i,j+1]

                    # save the edge
                    EdgeList.append([a, b])

                    # save the connection matrix
                    self.ConnectionMatrix[a,b]  = True
                    self.ConnectionMatrix[b,a]  = True

                    # EdgeList.append([self.NodeMatrix[i,j], self.NodeMatrix[i,j+1]])
        return EdgeList

    # **************************************************************************
    def _BuildNodeCoordinates(self):
        # set the function name
        FunctionName = "SWPL::_BuildNodeCoordinates()"

        # display the message
        if self.Verbose:
            # display the information
            Msg = "...%-25s: building node coordinates..." % (FunctionName)
            print(Msg)

        # reset the variable
        NodeCors    = np.zeros((self.Rows * self.Cols, 2), dtype=int)

        # go through the grid and build the coordinates
        for i in range(self.Rows):
            for j in range(self.Cols):
                NodeCors[self.NodeMatrix[i,j]]  = i,j

        return NodeCors

    # **************************************************************************
    # Calculate the Euclidean distance of two points: (x1, y1) and (x2, y2)
    #       d = sqrt[(x1 - x2)^2 + (y1 - y2)^2]
    # **************************************************************************
    def _EuclideanDist(self, NumA, NumB):
        # Distance    = self.NanoWireScale * np.sqrt(np.sum(np.power(np.subtract(self.NodeCoordinates[NumA], \
        #        self.NodeCoordinates[NumB]), 2)))
        Distance    = np.sqrt(np.sum(np.power(np.subtract(self.NodeCoordinates[NumA], \
               self.NodeCoordinates[NumB]), 2)))
        return Distance

    # **************************************************************************
    def _SetNodeIndices(self, IndexMatrix, i, Len, Array):
        # set the indices
        Start   = i * Len
        End     = Start + Len

        # set the index
        IndexMatrix[Start:End,0] = i
        IndexMatrix[Start:End,1] = Array

    # **************************************************************************
    def _BuildNodeIndices(self):
        # set the function name
        FunctionName = "SWPL::_BuildNodeIndices()"

        # display the message
        if self.Verbose:
            # display the information
            Msg = "...%-25s: building node indices..." % (FunctionName)
            print(Msg)

        # calcualte the total node
        TotalNodes  = self.Rows * self.Cols

        # reset the varables
        IndexArray  = np.arange(TotalNodes, dtype=int)
        IndexMatrix = np.zeros((TotalNodes * TotalNodes, 2), dtype=int)

        # set the indices
        Temp    = [self._SetNodeIndices(IndexMatrix, i, TotalNodes, IndexArray) for i in range(TotalNodes)]

        # print("IndexMatrix")
        # print(pd.DataFrame(IndexMatrix))
        # exit()
        return IndexMatrix

    # **************************************************************************
    def _CalEuclideanNodeIndices(self, EuclideanMatrix, i, j):
        EuclideanDist           = self._EuclideanDist(i, j)
        EuclideanMatrix[i,j]    = EuclideanDist
        EuclideanMatrix[j,i]    = EuclideanDist

    # **************************************************************************
    def _BuildEuclideanMatrix(self):
        # set the function name
        FunctionName = "SWPL::_BuildEuclideanMatrix()"

        # display the message
        if self.Verbose:
            # display the information
            Msg = "...%-25s: building Euclidean distance matrix..." % (FunctionName)
            print(Msg)

        # reset the variable
        TotalNodes      = self.Rows * self.Cols
        EuclideanMatrix = np.zeros((TotalNodes, TotalNodes))

        # get the index matrix
        if self.IndexMatrix is None:
            self.IndexMatrix = self._BuildNodeIndices()

        # print("self.IndexMatrix")
        # print(pd.DataFrame(self.IndexMatrix))
        # exit()
        # # go through the grid and build the coordinates
        # for i in range(TotalNodes):
        #     for j in range(TotalNodes):
        #         # EuclideanMatrix[i,j] = np.sqrt(np.sum(np.power(np.subtract(NodeCors[i], NodeCors[j]), 2)))
        #         EuclideanDist           = self._EuclideanDist(i, j)
        #         EuclideanMatrix[i,j]    = EuclideanDist
        #         EuclideanMatrix[j,i]    = EuclideanDist

        Temp    = [self._CalEuclideanNodeIndices(EuclideanMatrix, i, j) for i, j in \
                self.IndexMatrix]

        # return EuclideanMatrix * self.DeviceLength * self.WireScale

        # print("EuclideanMatrix")
        # print(pd.DataFrame(EuclideanMatrix))
        # exit()

        return EuclideanMatrix

    # **************************************************************************
    def _GenerateGraphFromEdges(self, EdgeList):
        # **********************************************************************
        # create an ordered graph from the edges
        # **********************************************************************
        AGraph      = nx.Graph()

        # **********************************************************************
        # get the tuple list of edges
        # **********************************************************************
        ListOfEdges = [tuple(Row) for Row in EdgeList]

        # **********************************************************************
        # adding edges to the graph
        # **********************************************************************
        AGraph.add_edges_from(ListOfEdges)

        # **********************************************************************
        # remove all self-loop edges
        # **********************************************************************
        AGraph.remove_edges_from(nx.selfloop_edges(AGraph))

        return AGraph

    # **************************************************************************
    def _CreateGridGraph(self):
        # set the function name
        FunctionName = "SWPL::_CreateGridGraph()"

        # display the message
        if self.Verbose:
            # display the information
            Msg = "...%-25s: creating grid graph..." % (FunctionName)
            print(Msg)

        # **********************************************************************
        # calculate the number of nodes
        # **********************************************************************
        self.Nodes = self.NumNodes   = self.Rows * self.Cols

        # **********************************************************************
        # set the range for creating nodes
        # **********************************************************************
        if self.GndNodeFlag:
            self.NodeArray   = np.arange(self.NumNodes)
        else:
            self.NodeArray   = np.arange(1, self.NumNodes + 1)

        # **********************************************************************
        # save the node list
        # **********************************************************************
        self.NodeList   = np.copy(self.NodeArray)
        # print("NodeList = ", self.NodeList)

        # **********************************************************************
        # shuffle the nodes
        # **********************************************************************
        # np.random.shuffle(NodeArray)

        # **********************************************************************
        # create a matrix for node number
        # **********************************************************************
        self.NodeMatrix = self.NodeArray.reshape(self.Rows, self.Cols)

        # **********************************************************************
        # create the edge list
        # **********************************************************************
        self.EdgeList   = self._CreateEdgeList()
        self.NumLinks   = len(self.EdgeList)

        # **********************************************************************
        # building node coordinates for wiring cost
        # **********************************************************************
        self.NodeCoordinates    = self._BuildNodeCoordinates()

        # building the Euclidean distant matrix
        self.EuclideanDistMatrix = self._BuildEuclideanMatrix()

        # print("EuclideanDistMatrix[a,b] = ", self.EuclideanDistMatrix[NodeOne, NodeTwo])

        # **********************************************************************
        # display the information
        # **********************************************************************
        # print("Node Matrix")
        # print(pd.DataFrame(self.NodeMatrix))
        # print("Node coordinates")
        # print(pd.DataFrame(self.NodeCoordinates))
        # print("EuclideanDistMatrix")
        # print(pd.DataFrame(self.EuclideanDistMatrix))
        # exit()

        # **********************************************************************
        # create an ordered graph from the edges
        # **********************************************************************
        GridGraph   = self._GenerateGraphFromEdges(self.EdgeList)
        return GridGraph

    # # **************************************************************************
    # def _LabelWithGndNode(self, Graph):
    #     # set the function name
    #     FunctionName = "SWPL::_LabelWithGndNode()"
    #
    #     # display the message
    #     if self.Verbose:
    #         # display the information
    #         Msg = "...%-25s: making labels to nodes..." % (FunctionName)
    #         print(Msg)
    #
    #     # calculating the number of nodes and set the labels
    #     self.Nodes = Graph.number_of_nodes()
    #
    #     # check the ground node flag
    #     if self.GndNodeFlag:
    #         Labels = np.arange(0, self.Nodes)
    #     else:
    #         Labels = np.arange(1, self.Nodes + 1)
    #
    #     # shuffle the labels
    #     np.random.shuffle(Labels)
    #
    #     # label the nodes
    #     Mapping = dict(zip(list(Graph.nodes()), Labels))
    #     NewGraph = nx.relabel_nodes(Graph, Mapping)
    #     return NewGraph
    #
    # # **************************************************************************
    # def _LabelNodes(self, Graph):
    #     # set the function name
    #     FunctionName = "SWPL::_LabelNodes()"
    #
    #     # # display the message
    #     # if self.Verbose:
    #     #     # display the information
    #     #     Msg = "...%-25s: relabeling nodes..." % (FunctionName)
    #     #     print(Msg)
    #
    #     # calculating the number of nodes and set the labels
    #     self.Nodes = Graph.number_of_nodes()
    #
    #     # # check the ground node flag
    #     # if self.GndNodeFlag:
    #     #     self.NodeList = np.arange(0, self.Nodes)
    #     # else:
    #     #     self.NodeList = np.arange(1, self.Nodes + 1)
    #
    #     self.NodeList = np.arange(1, self.Nodes + 1)
    #     Labels = np.copy(self.NodeList)
    #     # print("self.NodeList = ", self.NodeList)
    #
    #     # shuffle the labels
    #     np.random.shuffle(Labels)
    #
    #     # label the nodes
    #     Mapping = dict(zip(list(Graph.nodes()), Labels))
    #     NewGraph = nx.relabel_nodes(Graph, Mapping)
    #     return NewGraph

    # **************************************************************************
    def _RemoveLinks(self, Gamma):
        # # set the function name
        # FunctionName = "SWPL::_RemoveLinks()"
        #
        # # display the message
        # if self.Verbose:
        #     # display the information
        #     Msg = "...%-25s: remove links = <%.8g>" % (FunctionName, self.Gamma)
        #     print(Msg)

        # shuffle the list of edges
        np.random.shuffle(self.EdgeList)

        # set the number of remove links
        self.NumRemove = np.around(Gamma * len(self.EdgeList) / 100.0).astype(int)

        # remove links with a global link
        for n in range(self.NumRemove):
            # get the edge
            a, b = self.EdgeList.pop(0)

            # remove the edge
            self.ConnectionMatrix[a,b]      = False
            self.ConnectionMatrix[b,a]      = False

            # adjust the Euclidean distant matrix
            self.EuclideanDistMatrix[a,b]   = 0.0
            self.EuclideanDistMatrix[b,a]   = 0.0

            # remove an edge from the graph
            # self.InitGraphObject.remove_edge(a, b)

        # adjust the number of links
        self.NumLinks = len(self.EdgeList)
        return self.NumLinks

    # **************************************************************************
    def _NodeLengthMatrix(self):
        # # reset the length matrix
        # LengthMatrix = np.zeros((self.Nodes, self.Nodes))
        #
        # # get all the edges
        # AllEdges = Graph.edges()
        #
        # # build the Euclidean length matrix
        # for i, j in AllEdges:
        #     LengthMatrix[i,j]   = self.EuclideanDistMatrix[i,j]

        return np.copy(self.EuclideanDistMatrix)

    # **************************************************************************
    # def _NodeLengthMatrix(self, NodeLengths):
    #     # reset the length matrix
    #     LengthMatrix = np.zeros((self.Nodes, self.Nodes))
    #
    #     # get the keys from the dictionary
    #     keylist = NodeLengths.keys()
    #
    #     # build the lenght matrix
    #     for i in keylist:
    #         for j in NodeLengths[i]:
    #             LengthMatrix[i-1,j-1] = NodeLengths[i][j]
    #
    #     return LengthMatrix

    # **************************************************************************
    def _MatrixDecayPowerLaw(self, LengthMatrix):
        # set the function name
        FunctionName = "SWPL::_MatrixDecayPowerLaw()"

        # # indices of nonzero elements
        # mask = LengthMatrix != 0

        # reset the result matrix
        # TempMatrix = np.zeros(LengthMatrix.shape)

        # calculate the decay power law distribution
        TempMatrix  = np.power(LengthMatrix, -self.Alpha)

        # reset the values of node itself
        m, n    = TempMatrix.shape
        TempMatrix[np.arange(m), np.arange(m)]  = 0.0
        # print("Length matrix")
        # print(pd.DataFrame(LengthMatrix))
        # # print("TempMatrix = ", TempMatrix)

        # print("TempMatrix   = ", TempMatrix.shape)
        # print("TempMatrix")
        # print(pd.DataFrame(TempMatrix))
        # exit()

        # calculate the sum matrix
        SumMatrix = np.sum(TempMatrix, axis=1).reshape(self.Nodes, 1)
        # SumMatrix = np.sum(TempMatrix, axis=1)

        # print("SumMatrix")
        # print(pd.DataFrame(np.sum(TempMatrix, axis=1)))
        # exit()

        # check for zero elements
        if np.count_nonzero(SumMatrix) < self.Nodes:
            # check the verbal flag
            if self.Verbose:
                # format error message
                ErrMsg = "...%-25s: init graph error => unconnected elements..." % (FunctionName)
                print(ErrMsg)
            return None, False

        Results = np.divide(TempMatrix, SumMatrix)

        # print("Results  = ", Results.shape)
        # print(pd.DataFrame(Results))
        #
        # CheckSum    = np.sum(Results, axis=1).reshape(self.Nodes, 1)
        # print("CheckSum")
        # print(pd.DataFrame(CheckSum))
        # exit()

        return Results, True

    # **************************************************************************
    def _ConvertEdgeViewToList(self, Data):
        return [list(edge) for edge in Data]

    # **************************************************************************
    def _GenerateSWGraph(self):
        # reset the edge list
        EdgeList    = []
        NumNodes    = self.Rows * self.Cols

        # make a copy of connection matrix
        ConnectionMatrix    = np.copy(self.ConnectionMatrix)

        # build the edge list
        for i in range(NumNodes):
            for j in range(NumNodes):
                if ConnectionMatrix[i, j]:
                    # save the edge list
                    EdgeList.append([i,j])

                    # reset the connection matrix since the edge is already saved
                    ConnectionMatrix[i,j] = False
                    ConnectionMatrix[j,i] = False

        # save the edge list of small-world power-law graph
        self.EdgeList   = EdgeList
        return self._GenerateGraphFromEdges(EdgeList)

    # **************************************************************************
    def _CreateSWPowerLawGraph(self):
        # set the function name
        FunctionName = "SWPL::_CreateSWPowerLawGraph()"

        # display the message
        if self.Verbose:
            # display the information
            Msg = "...%-25s: creating small-world power-law graph..." % (FunctionName)
            print(Msg)

        # set the number of tries for creating the graph
        NumTries    = 5
        DoneFlag    = False

        # try to create a grid graph
        for n in range(NumTries):
            # creat a regular graph
            self.InitGraphObject = self._CreateGridGraph()

            # get the number of edges
            self.InitLinks  = len(self.EdgeList)
            NumEdges        = self.NumLinks

            # calculate the number of edge to be removed
            if self.Gamma > 0.0:
                NumEdges = self._RemoveLinks(self.Gamma)

            # check the Beta. If Beta is equal 0, the graph is regular graph.
            if self.Beta == 0.0:
                # return regular graph
                return self.InitGraphObject

            # calculate the number of edge to be replace
            self.NumReplace = np.around(self.Beta * NumEdges / 100.0).astype(int)

            # ******************************************************************
            # setting scaling to avoid l = 1 or l = 0, 1^x = 1 and 0^x = 1 where
            # l is the Euclidean distance between two nodes.
            # ******************************************************************
            Offset          = 1.5
            LengthMatrix    = np.copy(self.EuclideanDistMatrix) + Offset

            # print(pd.DataFrame(LengthMatrix))
            # exit()

            # calculate the distribution matrix
            DistributionMatrix, DoneFlag = self._MatrixDecayPowerLaw(LengthMatrix)

            # print("DistributionMatrix")
            # print(pd.DataFrame(DistributionMatrix))
            #
            # print(pd.DataFrame(np.sum(DistributionMatrix, axis=1)))
            # exit()

            # check the status flag
            if DoneFlag:
                break
            else:
                # reset the InitGraphObject and try again
                self.InitGraphObject = None

                # wait a bit and try again
                time.sleep(np.random.random() * 0.5)

        # check for zero elements
        if not DoneFlag:
            # format error message
            ErrMsg = "%s: fail to create regular graph" % (FunctionName)
            raise ValueError(ErrMsg)

        # shuffle the list of edges
        np.random.shuffle(self.EdgeList)

        # set the maximal number of tries
        MaxTries = 10e3

        # replace a local link with a global connection
        for n in range(self.NumReplace):
            a, b = self.EdgeList[n]

            # remove the local edge
            self.ConnectionMatrix[a,b]  = False
            self.ConnectionMatrix[b,a]  = False

            # the probability list of node a
            ProbabilityList = DistributionMatrix[a,:]

            # loop until a valid node is obtained
            Counter = 0
            while True:
                # pick a random node according to the power distribution
                c = np.random.choice(self.NodeList, size=1, p=ProbabilityList)[0]

                # check the node
                if c != a:
                    # check for the existent edge
                    HasEdgeFlag = self.ConnectionMatrix[a,c] or self.ConnectionMatrix[c,a]

                    # check the flag
                    if not HasEdgeFlag:
                        # the edge is good
                        break

                # adjust the counter
                Counter += 1

                # check the Counter
                if Counter > MaxTries:
                    # format error message
                    ErrMsg = "%s: cannot find a connection node <%d tries>" % \
                            (FunctionName, Counter)
                    raise ValueError(ErrMsg)

            # add a global edge
            self.ConnectionMatrix[a,c]  = True
            self.ConnectionMatrix[c,a]  = True

        return self._GenerateSWGraph()

    # **************************************************************************
    def _CheckNewGraph(self, AGraph, RemovedEdges):
        # set the function name
        FunctionName = "SWPL::_CheckNewGraph()"

        # check the number of new edges
        if RemovedEdges <= 0:
            return AGraph

        # display the message
        if self.Verbose:
            # display the information
            Msg = "...%-25s: checking small-world power-law network..." % (FunctionName)
            print(Msg)

        # get the edge list
        EdgeList = self._ConvertEdgeViewToList(AGraph.edges())

        # shufle the list
        np.random.shuffle(EdgeList)

        # get the number of new edges
        NumEdges = len(EdgeList)

        # remove extra edges
        Index = 0
        for i in range(RemovedEdges):
            while Index < NumEdges:
                # get an edge
                u, v = EdgeList[Index]

                # remove an edge
                AGraph.remove_edge(u,v)

                # try next element in the list
                Index += 1

                # check the condition of the graph
                if nx.is_connected(AGraph):
                    # the graph is OK, go the next edge
                    break

                else:
                    # reconnect the graph, and try the next edge
                    AGraph.add_edge(u, v)

        # check the new edge list
        if not nx.is_connected(AGraph):
            # format error message
            ErrMsg = "%s: Error in creating connected SW graph" % (FunctionName)
            raise ValueError(ErrMsg)

        return AGraph

    # **************************************************************************
    def _ConnectSubGraphs(self, AGraph):
        # set the function name
        FunctionName = "SWPL::_ConnectSubGraphs()"

        # display the message
        if self.Verbose:
            # display the information
            Msg = "...%-25s: connecting subgraphs of SW networks..." % (FunctionName)
            print(Msg)

        # get the subgraphs
        # SubGraphs = list(nx.connected_component_subgraphs(AGraph))
        SubGraphs = [AGraph.subgraph(c).copy() for c in nx.connected_components(AGraph)]

        # set the subgraph information
        SubGraphInfor = []

        # search for the largest connected graph and build subgraph information
        IndexMainGraph = i = 0
        MainGraph      = None
        CurrentNodes   = 0
        for sg in SubGraphs:
            # check the subgraph length
            NodeList = list(sg.nodes())
            # print("NodeList = ", NodeList)

            # set the number of node
            Nodes = len(NodeList)

            # check the number of node
            if Nodes > 1:
                # set the subgraph information entry
                Entry = [sg, NodeList, self._ConvertEdgeViewToList(sg.edges())]
            else:
                # subgraph only has one node
                Entry = [sg, NodeList, None]

            # save the entry
            SubGraphInfor.append(Entry)

            # check the number of nodes
            if Nodes > CurrentNodes:
                # adjust the index and current nodes
                IndexMainGraph = i
                CurrentNodes   = Nodes

            # adjust the index
            i += 1

        # set the new main graph
        NewMainGraph = SubGraphInfor[IndexMainGraph][0]

        # main graph node list
        MainNodes = list(NewMainGraph.nodes())

        # print("subgraphs           = ", len(SubGraphInfor))
        # print("nodes in main graph = ", len(MainNodes))

        # check the length of subgraphs
        if len(MainNodes) > len(SubGraphInfor):
            # set the connection subnodes of the main graph
            SubNodes = np.random.choice(MainNodes, len(SubGraphInfor) - 1, replace=False)
        else:
            # set the connection subnodes of the main graph
            SubNodes = np.random.choice(MainNodes, len(SubGraphInfor) - 1)

        # print("before main nodes = ", len(NewMainGraph.nodes()))
        # print("SubNodes = ", SubNodes)
        # exit()

        # connect all subgraphs
        NumAddEdges = NumSubGraphs = len(SubGraphInfor)
        j = 0
        for i in range(NumSubGraphs):
            # check for the main graph
            if i == IndexMainGraph:
                # skip the rest
                continue

            # get the node a
            a  = SubNodes[j]
            j += 1

            # get the node list and edges from the subgraph
            NodeList = SubGraphInfor[i][1]
            Edges    = SubGraphInfor[i][2]

            # check the node list
            if len(NodeList) > 1:
                b = np.random.choice(NodeList, 1)[0]
            else:
                b = NodeList[0]

            # adding an edge
            NewMainGraph.add_edge(a, b)

            # check the edge list
            if Edges is not None:
                NewMainGraph.add_edges_from(Edges)

        # remove the added edges to maintain the total edges
        return self._CheckNewGraph(NewMainGraph, NumAddEdges)

    # **************************************************************************
    def _AddLinks(self):
        # set the function name
        FunctionName = "SWPL::_AddLinks()"

        # display the message
        if self.Verbose:
            # display the information
            Msg = "...%-25s: adding links to SW network..." % (FunctionName)
            print(Msg)

        # get the node list
        NodeList = list(self.SWGraphObject.nodes())

        # number of nodes for each links
        NumNodes = 2

        # set the maximal number of tries
        MaxTries = 10e3

        # add links to the graph
        for i in range(self.Delta):
            # reset the counter
            Counter = 0

            # loop until the condition is OK
            while True:
                # select two random nodes from the node list
                a, b = np.random.choice(self.NodeList, NumNodes, replace=False)

                # set the flag
                Flag = self.SWGraphObject.has_edge(a,b) or self.SWGraphObject.has_edge(b,a)

                # add an edge if the condition is OK
                if not Flag:
                    # the edge is good, break the loop
                    break

                # increment the counter
                Counter     += 1

                # check the Counter
                if Counter > MaxTries:
                    # format error message
                    ErrMsg = "%s: cannot find a connection node <%d tries>" % \
                            (FunctionName, Counter)
                    raise ValueError(ErrMsg)

            # add a global edge
            self.SWGraphObject.add_edge(a, b)

            # adjust the Euclidean distant matrix
            Distance    = self._EuclideanDist(a, b)
            self.EuclideanDistMatrix[a,b]   = Distance
            self.EuclideanDistMatrix[b,a]   = Distance

    # **************************************************************************
    def _EdgeListWithWireResistance(self, EdgeList):
        # reset the result array
        ResultArray = []
        """
            * ******************************************************************
            * 0, 1, 2 , 3
            ********************************************************************
            * a, b, Rw, Distance
        """
        # file the result array
        for a, b in EdgeList:
            # print("a, b = ", a, b)
            # check for one node edge
            if a == b:
                # skip the rest
                continue

            # check for a valid distance
            if self.EuclideanDistMatrix[a,b] == 0.0:
                Distance    = self._EuclideanDist(a, b)

                # update the distant matrix
                self.EuclideanDistMatrix[a,b]   = Distance
                self.EuclideanDistMatrix[b,a]   = Distance
            else:
                Distance    = self.EuclideanDistMatrix[a,b]

            # save the connection nodes and the wire resistance
            Distance *=  self.NanoWireScale
            Rw  = Distance * self.NanoWireResistance / self.OneMicroMeter

            ResultArray.append([a, b, Rw, Distance])

        return np.asarray(ResultArray)

    # **************************************************************************
    def _FinalCheck(self):
        # **********************************************************************
        # make copye of node array
        # **********************************************************************
        NodeArray   = np.copy(self.NodeArray)

        # **********************************************************************
        # check for unconnected nodes
        # **********************************************************************
        UnconnectedNodes = np.delete(NodeArray, self.NodeList)

        # **********************************************************************
        # adding unconnected nodes to the device list
        # **********************************************************************
        if UnconnectedNodes.size > 0:
            # ******************************************************************
            # reset the unconnected node matrix
            # ******************************************************************
            UnNodeMatrix    = np.zeros((UnconnectedNodes.size, len(self.EdgeList[0])))

            # ******************************************************************
            # fill the unconected matrix
            # ******************************************************************
            UnNodeMatrix[:,0]  = UnconnectedNodes
            UnNodeMatrix[:,1]  = UnconnectedNodes

            # ******************************************************************
            # adding this to the device list
            # ******************************************************************
            self.EdgeList   = np.vstack((self.EdgeList, UnNodeMatrix))

    # **************************************************************************
    def _CheckValidEdgeList(self):
        # set the function name
        FunctionName = "SWPL::_CheckValidEdgeList()"

        # display the message
        if self.Verbose:
            # display the information
            Msg = "...%-25s: check for a valid edge list..." % (FunctionName)
            print(Msg)

        # get the indices of zero resistance entries
        Indices = np.argwhere(self.EdgeList[:,2] == 0.0)

        # check the indices
        if Indices.size > 0:
            # format error message
            ErrMsg = "%s: invalid edge list" % (FunctionName)
            raise ValueError(ErrMsg)

    # **************************************************************************
    def _CreateSWGraph(self):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "SWPL::_CreateSWGraph()"

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # display the information
            Msg = "...%-25s: creating small-world power-law network..." % (FunctionName)
            print(Msg)

        # # reset the Flag
        # DoneFlag = False

        # **********************************************************************
        # create a graph
        # **********************************************************************
        self.SWGraphObject = self._CreateSWPowerLawGraph()

        # **********************************************************************
        # set the number of tries for creating the graph
        # **********************************************************************
        NumTries    = 3
        DoneFlag    = False

        # **********************************************************************
        # try to create a connected graph
        # **********************************************************************
        for n in range(self.MaxTries):
            # reset the flag
            TrySuccess = False

            # ******************************************************************
            # trying to create graph
            # ******************************************************************
            for m in (range(NumTries)):
                try:
                    # create a graph
                    self.SWGraphObject = self._CreateSWPowerLawGraph()

                    # it is ok
                    TrySuccess = True
                    break

                except Exception as Error:
                    # ErrMsg = "<%3d> => %s" % (n, Error)
                    # print(ErrMsg)
                    # reset the flag
                    # TrySuccess = False

                    # wait a bit and try again
                    time.sleep(np.random.random() * 0.5)

            # ******************************************************************
            # check the flag
            # ******************************************************************
            if TrySuccess:
                # check for connected graph
                DoneFlag = nx.is_connected(self.SWGraphObject)

            # ******************************************************************
            # display the message
            # ******************************************************************
            if self.Verbose:
                print("Tries = ", n, ":", m, ", DoneFlag = ", str(DoneFlag))

            # ******************************************************************
            # check the flag
            # ******************************************************************
            if DoneFlag:
                # graph is OK, break the loop
                break

        # **********************************************************************
        # check the flag
        # **********************************************************************
        if not TrySuccess:
            # format error message
            ErrMsg = "%s: failing to create a connected graph" % (FunctionName)
            raise ValueError(ErrMsg)

        # **********************************************************************
        # check the flag
        # **********************************************************************
        if not DoneFlag:
            # ******************************************************************
            # check the verbose flag
            # ******************************************************************
            if self.Verbose:
                # display the information
                Msg = "...%-25s: network is disconected, try connect subgraph..." % (FunctionName)
                print(Msg)

            # ******************************************************************
            # connecting all subgraphs
            # ******************************************************************
            self.SWGraphObject = self._ConnectSubGraphs(self.SWGraphObject)

        # **********************************************************************
        # check the add link variables
        # **********************************************************************
        if self.Delta > 0:
            self._AddLinks()

        # **********************************************************************
        # save the node list and edge list
        # **********************************************************************
        self.NodeList = list(self.SWGraphObject.nodes())
        self.EdgeList = self._EdgeListWithWireResistance(self.SWGraphObject.edges())

        # **********************************************************************
        # check for error in wire resistance
        # **********************************************************************
        Rnw = np.amax(self.EdgeList[:,2])
        if Rnw >= self.Rnw_Max:
            # format error message
            ErrMsg = "%s: Rnw is large <%.4g> " % (FunctionName, Rnw)
            raise ValueError(ErrMsg)

        # **********************************************************************
        # check for valid edge list
        # **********************************************************************
        self._CheckValidEdgeList()

        # **********************************************************************
        # perform the final check
        # **********************************************************************
        self._FinalCheck()

    # # **************************************************************************
    # def _SWNetworkGraph(self):
    #     # set the function name
    #     FunctionName = "SWPL::_SWNetworkGraph()"
    #
    #     # set the initial values
    #     NumTries    = 3
    #     DoneFlag    = False
    #
    #     # trying to create graph
    #     for m in (range(NumTries)):
    #         try:
    #             # create a graph
    #             self._CreateSWGraph()
    #
    #             # it is ok
    #             DoneFlag = True
    #             break
    #
    #         except Exception as Error:
    #             # ErrMsg = "<%3d> => %s" % (n, Error)
    #             # print(ErrMsg)
    #             # reset the flag
    #             # TrySuccess = False
    #
    #             # wait a bit and try again
    #             time.sleep(np.random.random() * 0.5)
    #
    #     # check the flag
    #     if not DoneFlag:
    #         # format error message
    #         ErrMsg = "%s: error in creating a connected graph" % (FunctionName)
    #         raise ValueError(ErrMsg)

    # **************************************************************************
    def _CalculateNetworkWireLength(self):
        # set the function name
        FunctionName = "SWPL::_CalculateNetworkWireLength()"

        # display the message
        if self.Verbose:
            # display the information
            Msg = "...%-25s: calculating total network wire length..." % (FunctionName)
            print(Msg)

        # wire length
        WireLength  = np.asarray([[self.EuclideanDistMatrix[int(a), int(b)], Rw] for a, b, Rw, d in self.EdgeList])

        # get the total wire length and wire resistance
        self.TotalWireLen, self.TotalRes = np.sum(WireLength, axis=0)

        # scale the length to nano scale
        self.TotalWireLen    *= self.NanoWireScale

    # **************************************************************************
    def _NodeDegree(self):
        # set the function name
        FunctionName = "SWPL::_NodeDegree()"

        # display the message
        if self.Verbose:
            # display the information
            Msg = "...%-25s: computing node degree." % (FunctionName)
            print(Msg)

        # get the node degree dictionary
        NodeDegreeDict = self.SWGraphObject.degree()

        # calculate the average node degree
        NodeDegree = 0.0
        for Node in self.NodeList:
            # calculating the node dgree
            NodeDegree += float(NodeDegreeDict[Node])

        # get the number of nodes
        NumNodes = float(len(self.NodeList))

        # calculating the average degree
        NodeDegree /= NumNodes

        return NodeDegree

    # **************************************************************************
    def GetGraphName(self):
        return self.GraphName

    # **************************************************************************
    def GetInitGraph(self):
        return "Init" + self.InitGraphName, self.InitGraphObject

    # **************************************************************************
    def GetSWGraph(self):
        return self.SWGraphObject

    # **************************************************************************
    def GetNodes(self):
        return self.NodeList

    # **************************************************************************
    def GetGndNodeFlag(self):
        return self.GndNodeFlag

    # **************************************************************************
    def GetNodesAndEdges(self):
        return self.NodeList, np.asarray(self.EdgeList)

    # **************************************************************************
    def GetNumEdges(self):
        # return len(self.EdgeList)
        return self.SWGraphObject.number_of_edges()

    # **************************************************************************
    def GetNodeDegree(self):
        return self.NodeDegree, self.InDegree, self.OutDegree

    # **************************************************************************
    def GetNumNodesAndEdgesSWGraph(self):
        return len(self.NodeList), len(self.EdgeList)

    # **************************************************************************
    def GetNumNodesAndEdgesInitGraph(self):
        return self.NumNodes, self.InitLinks

    # **************************************************************************
    def GetEdgeList(self):
        return np.asarray(self.EdgeList)

    # **************************************************************************
    def GetTotalWireInfo(self):
        return self.TotalWireLen, self.TotalRes

    # **************************************************************************
    def GetNodeCoordinates(self):
        return self.NodeCoordinates

    # **************************************************************************
    def GetWireMinMaxRes(self):
        WireMin = np.amin(self.EdgeList[:,2])
        WireMax = np.amax(self.EdgeList[:,2])
        return WireMin, WireMax

    # **************************************************************************
    def GetRWmin(self):
        return self.RWmin

    # **************************************************************************
    def GetMaxNodeNumber(self):
        return np.amax(self.NodeList)

    # # **************************************************************************
    # def DisconnectedSWGraph(self):
    #     # set the function name
    #     FunctionName = "SWPL::DisconnectedSWGraph()"
    #
    #     # display the message
    #     if self.Verbose:
    #         # display the information
    #         Msg = "...%-25s: creating small-world power-law network..." % (FunctionName)
    #         print(Msg)
    #
    #     # reset the Flag
    #     DoneFlag = False
    #
    #     # try to create a connected graph
    #     for n in range(self.MaxTries):
    #         # create a graph
    #         Graph = self._CreateSWPowerLawGraph()
    #
    #         # check for connected graph
    #         if not nx.is_connected(Graph):
    #             break;
    #
    #     # get the graph
    #     SubGraphs = nx.connected_component_subgraphs(Graph)
    #     i = 0
    #     for sg in SubGraphs:
    #         i += 1
    #         print(i, "nodes = ", len(sg.nodes()))

# *****************************************************************************
def PlotGraph(GraphObject, GraphName, Nodes, Edges, Label=True):
    # set the figure title
    GraphTitle = "%s Graph (Nodes = %d, Edges = %d)" % (GraphName, Nodes, Edges)
    plt.figure(GraphTitle)

    # display the graph
    # nx.draw_networkx(SWGraph, alpha=1.0, node_color="b", font_color="w",\
    #         font_family="Times New Roman")
    # nx.draw_networkx(GraphObject, alpha=1.0, node_color="b", font_color="w")
    # nx.draw_networkx(GraphObject, with_labels=True, node_size=500, node_color="b")
    # nx.draw_networkx(GraphObject, with_labels=Label, alpha=1.0, node_color="b", font_color="w")
    # nx.draw_networkx(GraphObject, with_labels=False, alpha=1.0, node_color="b", font_color="w")
    # nx.draw_networkx(GraphObject, with_labels=True, alpha=1.0, node_color="b", font_color="w")

    nx.draw_networkx(GraphObject,
        with_labels=False,
        node_size=25,
        # node_color=[0.5,0.5,0.5],
        # node_color="#87CEEB",
        node_color="b",
        # alpha=0.5,
        font_color="w")

    # plt.tick_params(
    #     axis="both",       # changes apply to the x_axis
    #     which="both",      # both major and minor ticks are affected
    #     bottom="off",      # ticks along the bottom edge are off
    #     left="off",        # ticks along the bottom edge are off
    #     top="off",         # ticks along the top edge are off
    #     labelbottom="off",
    #     labelleft="off") # labels along the bottom edge are off
    plt.axis('off')

    plt.tick_params(
        axis="both",       # changes apply to the x_axis
        which="both",      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left=False,        # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labelleft=False) # labels along the bottom edge are off
    # get nodes and edges
    # Nodes, Edges = GraphClass.GetNodesAndEdges()

    # set the file name
    FileNameEps = join("Figures", GraphName + ".eps")
    FileNameSvg = join("Figures", GraphName + ".svg")
    # FileNamePng = join("Figures", GraphName + ".png")

    # save the figure
    # plt.savefig(FileNameEps, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(FileNameSvg, bbox_inches="tight", pad_inches=0.2)
    # plt.savefig(FileNamePng)

# *****************************************************************************
if __name__ == "__main__":
    # **************************************************************************
    # import modules
    # **************************************************************************
    import os, sys
    from os.path import join, isdir, isfile

    # # **************************************************************************
    # # check the platform to import mem-models
    # # **************************************************************************
    # Platform = sys.platform
    # if (Platform == "linux2") or (Platform == "linux"):
    #     # **********************************************************************
    #     # get the home path
    #     # **********************************************************************
    #     HomePath    = os.getenv("HOME")
    #     if "WAVE" in HomePath:
    #         MemModelPath = join(HomePath, "PythonMemModels")
    #         MemComon     = join(HomePath, "MemCommonFuncs")
    #     elif "home" in HomePath:
    #         MemModelPath = join(HomePath, "PythonMemModels")
    #         MemComon     = join(HomePath, "MemCommonFuncs")
    #     else:
    #         MemModelPath = join(HomePath, "workspace", "PythonMemModels")
    #         MemComon     = join(HomePath, "workspace", "MemCommonFuncs")
    #
    # elif Platform == "darwin":
    #     MemModelPath = join(HomePath, "PythonMemModels")
    #     MemComon     = join(os.getenv("HOME"), "MemCommonFuncs")
    #
    # elif Platform == "win32":
    #     HomePath     = os.getenv("USERPROFILE")
    #     MemModelPath = join(HomePath, "Documents", "PythonMemModels")
    #     MemComon     = join(HomePath, "Documents", "MemCommonFuncs")
    # else:
    #     # format error message
    #     Msg = "unknown platform => <%s>" % (Platform)
    #     raise ValueError(Msg)
    #
    # # **************************************************************************
    # # append to the system path
    # # **************************************************************************
    # # sys.path.append(MemModelPath)
    # sys.path.append(MemComon)

    # **************************************************************************
    # import module
    # **************************************************************************
    from MathUtils import FindFactors

    # **************************************************************************
    # set for autolayout matplotlib
    # **************************************************************************
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"figure.autolayout": True})

    # **************************************************************************
    # set the font family
    # **************************************************************************
    # FontSize = 12
    # FontSize = 14
    FontSize = 16
    # FontSize = 18
    # FontSize = 20
    # FontSize = 22
    # FontSize = 24
    # FontSize = 28
    font = {"family": "Times New Roman", "size": FontSize}
    plt.rc("font", **font)  # pass in the font dict as kwargs
    plt.rcParams["mathtext.fontset"] = "cm"

    # **************************************************************************
    import cProfile

    # **************************************************************************
    """ A temporary fix for the error:
        Initializing libiomp5md.dll, but found libiomp5md.dll
        already initialized """
    # **************************************************************************
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    # **************************************************************************
    # set the function name
    # **************************************************************************
    FunctionName = "SWPL::main()"

    # **************************************************************************
    # initial starting graph
    # **************************************************************************
    InitGraph = "Grid_Graph"
    # InitGraph = "Random_Regular"

    # paramters
    # Nodes   = 800
    # M     = 20
    # N     = 20

    # *************************************************************************
    # nodes = 97, Alpha = 2.7228368, Beta = 87.760039, Gamma = 3.8981189
    # *************************************************************************
    """
            * Alpha         : the decay exponent: Alpha < D + 1
            * Beta          : is the percentage of replacing local links
                                    Beta = 0        -> regular network
                                    Beta = 100.0    -> completely random network
            * Delta         : number of extra links to be added
            * Gamma         : percent of removed links
    in = 18, out = 7, nodes = 193, A = 0.39105211, B = 54.031976, G = 3.9793934, D = 9
    """
    # **************************************************************************
    # parameters for small-world power-law graph
    # **************************************************************************
    Inputs      = 20
    Outputs     = 20
    Nodes       = Inputs + Outputs + 5
    Verbose     = True

    # initial starting graph
    InitGraph   = "Grid_Graph"
    Alpha       = 1.390525834
    Beta        = 82.42180956
    Gamma       = 16.33022077
    Delta       = 0
    BoundaryFlag = False
    GndNodeFlag  = True
    L       = FindFactors(Nodes)

    # # set the nodes and degree
    # Nodes  = 50
    # Degree = 4

    # **************************************************************************
    # profiling the code
    # **************************************************************************
    pr = cProfile.Profile()
    pr.enable()

    # create a random graph
    GraphClass = SmallWorldPowerLaw(InitGraph=InitGraph, L=L, Beta=Beta, Alpha=Alpha, \
            Gamma=Gamma, Delta=Delta, BoundaryFlag=BoundaryFlag, GndNodeFlag=GndNodeFlag, \
            Verbose=Verbose, MaxTries=2)

    pr.disable()
    pr.print_stats()
    # exit()
    # **************************************************************************

    # get the initial graph
    InitGraphName, InitialGraph = GraphClass.GetInitGraph()

    # get the device list
    DevList     = GraphClass.GetEdgeList()
    NumDev      = GraphClass.GetNumEdges()

    # exit()
    WireLength, WireRes = GraphClass.GetTotalWireInfo()

    print("WireLength   = ", WireLength)
    print("WireRes      = ", WireRes)
    print("NumDev       = ", NumDev)

    print("DevList")
    print(pd.DataFrame(DevList))

    RWires  = np.asarray([Rw for a, b, Rw, d in DevList if a != b])

    RWmin   = np.amin(RWires)
    RWmax   = np.amax(RWires)

    # print("RWmin    = ", GraphClass.GetRWmin())
    # print("RWmax    = ", RWmax)
    # exit()

    # get the initial graph
    InitGraphName, InitialGraph = GraphClass.GetInitGraph()

    # get the SW graph
    SWGraph = GraphClass.GetSWGraph()

    # get the net list
    DevList = GraphClass.GetEdgeList()

    # save the device list to a file
    FileName = "DevList"
    # np.savez_compressed(FileName, DevList=DevList)

    # SWGraph.add_edge(1,6)
    # SWGraph.add_edge(1,6)
    # SWGraph.add_edge(1,6)
    #
    # SWGraph.add_edge(6,1)
    # SWGraph.add_edge(6,1)

    # **************************************************************************
    # plot the initial graph
    # **************************************************************************
    GraphName = GraphClass.GetGraphName()

    # get number of nodes and edges
    Nodes, Edges = GraphClass.GetNumNodesAndEdgesInitGraph()

    # **************************************************************************
    # set the node labels
    # **************************************************************************
    NodeLabels  = np.arange(0, Nodes)

    # **************************************************************************
    # shuffle the labels
    # **************************************************************************
    np.random.shuffle(NodeLabels)

    # **************************************************************************
    # label the nodes
    # **************************************************************************
    Mapping = dict(zip(SWGraph.nodes(), NodeLabels))
    SWGraph = nx.relabel_nodes(SWGraph, Mapping)

    # plot the graph
    # sPlotGraph(InitialGraph, InitGraphName, Nodes, Edges, Label=True)

    # *************************************************************************
    # plot the small-world graph
    # *************************************************************************
    GraphName = GraphClass.GetGraphName()

    # get number of nodes and edges
    Nodes, Edges = GraphClass.GetNumNodesAndEdgesSWGraph()

    # **************************************************************************
    # set the figure name and title
    # **************************************************************************
    plt.figure(GraphName)
    GraphTitle = "Small-world power-law Network \n" + \
            r"$Node = %d, \alpha = %.2f, \beta = %.2f, \gamma = %d, \delta = %d$" % \
            (Nodes, Alpha, Beta/100.0, Gamma, Delta)
    plt.title(GraphTitle, color='b')

    # **************************************************************************
    # display the graph
    # **************************************************************************
    NodeSize = 300
    nx.draw_networkx(SWGraph, with_labels=True, alpha=1.0, node_color="b", font_color="w",\
            node_size=NodeSize)

    plt.tick_params(
        axis="both",       # changes apply to the x_axis
        which="both",      # both major and minor ticks are affected
        bottom="off",      # ticks along the bottom edge are off
        left="off",        # ticks along the bottom edge are off
        top="off",         # ticks along the top edge are off
        labelbottom="off",
        labelleft="off") # labels along the bottom edge are off

    plt.axis("off")
    plt.axis("tight")

    # **************************************************************************
    # set the file name
    # **************************************************************************
    # FileName = join("Figures", GraphName + ".eps")
    FileName = join("Figures", GraphName + ".jpg")
    FileName = join("Figures", GraphName + ".png")

    # **************************************************************************
    # save the figure
    # **************************************************************************
    print("...Saving figure to file = <%s> ..." % FileName)

    # **************************************************************************
    # save the figure
    # **************************************************************************
    plt.savefig(FileName)

    plt.show()
