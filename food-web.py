import networkx as nx
import matplotlib.pyplot as plt

G = nx.karate_club_graph()

print("Node Degree")

# for v in G:
#     print(G.nodes)

fig, ax = plt.subplots(figsize=(15, 9))
ax.axis("off")
plot_options = {"node_size": 10, "with_labels": True, "width": 0.15}
nx.draw_networkx(G, pos=nx.random_layout(G), ax=ax, **plot_options)
plt.show()
