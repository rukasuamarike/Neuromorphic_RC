# ********************************************************
# import modules
import networkx as nx
from os.path import join
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({"figure.autolayout": True})
# ********************************************************
# set the font family
# ********************************************************
FontSize = 5
font = {"family": "Times New Roman", "size": FontSize}
plt.rc("font", **font) # pass in the font dict as kwargs
# ********************************************************
# create a graph
# ********************************************************
GraphName = "Newman_Watts_Strogatz"
Nodes = 50
k = 2
p = 0.22
Graph = nx.newman_watts_strogatz_graph(Nodes, k, p)
# ********************************************************
# set graph name and title
# ********************************************************
plt.figure(GraphName)
plt.title(GraphName + " Graph")
# ********************************************************
# display the graph
# ********************************************************
nx.draw_networkx(Graph, node_size = 10, with_labels=True, alpha=1.0, node_color="r",
font_color="black")
# ********************************************************
# set the parameters for axes
# ********************************************************
plt.tick_params(
axis="both", # changes apply to the x_axis
which="both", # both major and minor ticks are affected
bottom="off", # ticks along the bottom edge are off
left="off", # ticks along the bottom edge are off
top="off", # ticks along the top edge are off
labelbottom="off",
labelleft="off") # labels along the bottom edge are off
plt.axis("off")
plt.axis("tight")
# ********************************************************
# set the file name
# ********************************************************
FileName = (GraphName + ".jpg")
print("...Saving figure to file = <%s> ..." % FileName)
# ********************************************************
# save the figure
# ********************************************************
# plt.savefig(FileName, bbox_inches='tight’)
plt.show()
