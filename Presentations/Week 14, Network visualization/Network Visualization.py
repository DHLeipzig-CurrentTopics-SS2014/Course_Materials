# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Network Visualization

# <headingcell level=2>

# Getting networks from texts

# <markdowncell>

# When using networks visualization for text analysis, the first step is to get a network from texts. This will not be covered here in detail, but I will outline some basic ideas.
# 
# The easiest way of getting a network from a text is a “cooccurrence network”. It uses words as nodes in the network and adds edges between nodes for all word cooccurrences. The edges can be weighted by log-likelihood. When creating the network, one will likely apply some sort of generalization, e.g. stemming or lemmatization, and some sort of filtering, e.g. by part of speech or using a stopword list.
# 
# Using pseudocode, the idea looks mainly like this (for sentence cooccurrences):
# 
# ```python
# graph = Graph()
# for sentence in sentences:
#     words = [generalize(word) for word in sentence.words if filter(word)]
#     for a, b in combinations(words, 2):
#         edge = graph.find_edge(a, b)
#         if edge:
#             edge['weight'] += 1
#         else:
#             graph.add_edge(a, b, weight=1)
# ```
# 
# In this example, the edge weight is the number of cooccurrences. In a second step, loglikelihood can be calculated from this information.

# <markdowncell>

# In this session, we will use a prepared graph. It is from the Mahābhārata, and ancient Indian epic, writing in Sanskrit and one of the central texts of Hinduism.

# <codecell>

from igraph import *
g = load("chap01-001.graphml")
summary(g)

# <markdowncell>

# So how can we use the cooccurrence network to learn more about the text? Of course by visualizing it.

# <codecell>

bbox = (1200, 1200)
plot(g, bbox=bbox, target='plot.svg')

# <markdowncell>

# Now since the network contains almost 1500 nodes and 7000 eges, it is hardly possible to see anything from this. An alternative is to look for information programmatically. So we can look for nodes with the hightes degree, i.e., the highest number of connected nodes:

# <codecell>

from collections import Counter
names_by_degree = Counter(dict(zip(g.vs['label'], g.degree())))
names_by_degree.most_common(10)

# <markdowncell>

# But this way, the relational information gets lost, which is one of the main advantages of using network visualization. In the following, we will use a small part of the network to discuss some main principles of network visualization, and then turn to problems of visualizing large networks.
# 
# First, we will look for the cooccurrence network of a specific term, “dharma”:

# <codecell>

node = g.vs.find(label='dharma')
g_dharma = g.subgraph(node.neighbors())
summary(g_dharma)

# <headingcell level=2>

# Network Visualization

# <markdowncell>

# Now what are basic principles of network visualization?

# <headingcell level=3>

# Layout

# <markdowncell>

# When plotting a network, one has to choose where to place the nodes. There is a variety of possibilities to do so:

# <codecell>

for layout_algo in ('random', 'circle', 'fruchterman_reingold'):
    layout = g_dharma.layout(layout_algo)
    plot(g_dharma, layout=layout, bbox=bbox, target='plot_' + layout_algo + '.svg')

# <markdowncell>

# For standard layouts, the main principles are these:
# 
# * the geometric distance between nodes in the plot should be aproximately proportional to the geodesic distance, i.e. the number of steps in the path between the two nodes.
# * When possible, edges should not cross.
# 
# The currently most frequently applied principle to generate layouts that fulfil these requirements are physics simulations based on what is called a “spring embedder model”. In all its variants, it is based on the central model that nodes push each other away, while edges draw nodes towards each other, like a spring.
# 
# These layouts thus are iteratively, they start with a random layout and then recalculate the layout for a given number of steps.

# <headingcell level=3>

# Properties

# <markdowncell>

# In addition to the layout itself, network visualizations allow to include additional information about the network into a single plot, making it a good source for visual interpretation by humans.
# 
# **Question:** Which information about the graph would you include into the network?
# 
# These might include:
# 
# * Part-of-speech
# * Centrality (degree, i.e., number of connections)
# * Edge weight
# 
# **Question:** What possibilities do we have to add information into the network visualization?
# 
# * Node size
# * Edge width
# * Node shape
# * Node color
# * Edge color
# 
# **Question:** Which plotting method is good to visualize which property?
# 
# * Size (node size, edge width): Good for metrical properties (edge weight, node centrality).
# * Shape: Good for categorial properties (word class).
# * Color: Good for both (gradiant vs. distinct colors).

# <codecell>

import numpy as np
def normalize(seq):
    seq = np.array(seq)
    try:
        return (seq - min(seq)) / (max(seq) - min(seq))
    except ZeroDivisionError:
        return [.5] * len(seq)

# <codecell>

degree = g_dharma.degree()
shape_map = {'noun': 'square', 'verb': 'triangle', 'adjective': 'circle'}
shape = [shape_map[pos] for pos in g_dharma.vs['type']]
width = normalize(g_dharma.es['weight']) * 19 + 1
size = normalize(degree) * 25 + 5
plot(g_dharma, layout=layout,
               bbox=bbox,
               target='plot_properties.svg',
               vertex_size=size,
               vertex_shape=shape,
               edge_width=width,
               )

# <headingcell level=3>

# Detecting structures

# <markdowncell>

# This visualization makes basic structures of the network topology visible. But there are also more subtile structures that we can visualize. E.g., when analyzing term context networks, we might want to find semantic clusters. These are areas in the network with higher connection density. In network analysis, finding these clusters is called “community detection”. There are several algorithms that allow to find communities, but most of the recent ones are based on the work of Girvan and Newman (2002).

# <codecell>

communities = g_dharma.community_multilevel()
print(communities)

# <markdowncell>

# Now we can add this information to the network plot.

# <codecell>

palette = RainbowPalette(len(communities))
color = [palette.get(i) for i in communities.membership]
plot(g_dharma, layout=layout,
               bbox=bbox,
               target='plot_communities.svg',
               vertex_size=size,
               vertex_shape=shape,
               vertex_color=color,
               edge_width=width,
               )

# <headingcell level=2>

# Large Graphs

# <markdowncell>

# Plotting a network of this size does not really work. It results in the well-known “hairball” plots.

# <markdowncell>

# The graph can contain several “components”, i.e. unconnected subgraphs. Since many calculations do not work (or make sense) on unconnected graphs, and these are usualy less important outliers, only the largest, “giant” component is used.

# <codecell>

components = g.clusters()
g = components.giant()
summary(g)

# <markdowncell>

# It would be helpful to disentangle the graph, making the internal structure more visible. The community structure of the graph can be used to re-position the nodes of the large graph.

# <codecell>

communities = g.community_multilevel()

# <markdowncell>

# A strategy for plotting a graph can be to plot it in a way that visualizes the community structure. This means that nodes belonging to the same community should be plotted close to each other, and in distance from other nodes.
# 
# **Question:** How could one implement such a layout algorithm?
# 
# **Hint:** There is a method that creates a graph that merges all nodes from a community, resulting in a higher-level community graph.

# <codecell>

contracted_graph = communities.cluster_graph()
outer_layout = contracted_graph.layout('auto')
outer_layout.scale(contracted_graph.vcount())

# <codecell>

plot(contracted_graph, layout=outer_layout, bbox=bbox, target='plot_outer.svg')

# <codecell>

outer_box = outer_layout.bounding_box()
from math import sqrt
r = sqrt(sum(outer_box.shape)/2)

# <codecell>

inner_layout = Layout([(0, 0) for _ in range(g.vcount())])
for comm, vertices in enumerate(communities):
    print('Plotting layout for community {} ...'.format(comm))
    comm_graph = g.induced_subgraph(vertices)
    comm_layout = g.layout('fruchterman_reingold')
    #comm_layout = g.layout('random')
    cx, cy = outer_layout[comm]
    inner_box = (cx-r, cy-r, cx+r, cy+r)
    comm_layout.fit_into(inner_box)
    for vertex, coords in zip(vertices, comm_layout):
        inner_layout[vertex] = coords

# <codecell>

degree = g.degree()
shape = [shape_map[pos] for pos in g.vs['type']]
width = normalize(g.es['weight']) * 38 + 2
size = normalize(degree) * 50 + 10
palette = RainbowPalette(len(communities))
color = [palette.get(i) for i in communities.membership]
plot(g, layout=inner_layout,
        bbox=(2400, 2400),
        target='plot_large.svg',
        vertex_size=size,
        vertex_shape=shape,
        vertex_color=color,
        edge_width=width,
        )

# <codecell>

g.vs["community"] = [str(m) for m in communities.membership]  # Trick gephi
for vertex, coords in enumerate(inner_layout):
    x, y = coords
    g.vs[vertex]["x"], g.vs[vertex]["y"] = x * 100, -y * 100  # Gephi uses flipped Y coordinate
save(g, "chap01-001-layout.graphml")

