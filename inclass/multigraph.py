import networkx as nx
from os.path import join
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random

# Create a MultiGraph
G = nx.MultiGraph()

# Add nodes
G.add_nodes_from([1, 2, 3])

# Add a simple edge
G.add_edge(1, 2)

# Add an edge with a key and attribute
G.add_edge( 2, 3, key="special", weight=10)

# Add multiple edges with attributes (list of tuples)
edges_to_add = [(1, 3, {"label": "direct"}), (2, 1, {"weight": 2})]
G.add_edges_from(edges_to_add)

# Check the added edges
print(G.edges(data=True))


plt.figure(G.name)  # Use the combined graph's name
plt.title(f"Combined Graph ({G} )")

# Display the combined graph
# Adjust node colors or styles to differentiate between graphs (if applicable)
nx.draw_networkx(G, node_size=10, with_labels=True, alpha=1.0, font_color="black")

# Set the parameters for axes (same as before)
plt.tick_params(
    axis="both",
    which="both",
    bottom="off",
    left="off",
    top="off",
    labelbottom="off",
    labelleft="off",
)
plt.axis("off")
plt.axis("tight")

# Set the file name (update for combined graph)
FileName = f"Combined_{G}.jpg"
print(f"...Saving figure to file = <{FileName}> ...")

# Save and show the figure
plt.savefig(FileName, bbox_inches="tight")
plt.show()
