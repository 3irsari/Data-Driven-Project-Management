import networkx as nx
import matplotlib.pyplot as plt

# Create graphs for Reactive and Proactive processes
G_reactive = nx.DiGraph()
G_proactive = nx.DiGraph()

# Define nodes and edges for Reactive (traditional) process
reactive_nodes = [
    "Project Event", "Issue Detected", "Manual Analysis", "Decision Made", "Action Taken"
]
G_reactive.add_nodes_from(reactive_nodes)
G_reactive.add_edges_from([
    ("Project Event", "Issue Detected"),
    ("Issue Detected", "Manual Analysis"),
    ("Manual Analysis", "Decision Made"),
    ("Decision Made", "Action Taken")
])

# Define nodes and edges for Proactive (AI-powered) process
proactive_nodes = [
    "Project Event", "AI Predictive Analysis", "Forecasting & Alerts", "Automated Decision", "Action Taken"
]
G_proactive.add_nodes_from(proactive_nodes)
G_proactive.add_edges_from([
    ("Project Event", "AI Predictive Analysis"),
    ("AI Predictive Analysis", "Forecasting & Alerts"),
    ("Forecasting & Alerts", "Automated Decision"),
    ("Automated Decision", "Action Taken")
])

# Visualize the graphs
fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# Reactive Process
pos_reactive = {node: (i, 0) for i, node in enumerate(reactive_nodes)}
nx.draw_networkx(
    G_reactive, pos=pos_reactive, ax=axes[0],
    node_color="lightcoral", node_size=8000, with_labels=True,
    font_size=9, font_color="black", arrowsize=15
)
axes[0].set_title("Reactive (Traditional) Decision-Making")

# Proactive Process
pos_proactive = {node: (i, 0) for i, node in enumerate(proactive_nodes)}
nx.draw_networkx(
    G_proactive, pos=pos_proactive, ax=axes[1],
    node_color="lightgreen", node_size=9000, with_labels=True,
    font_size=11, font_color="black", arrowsize=15
)
axes[1].set_title("Proactive (AI-Powered) Decision-Making")

# Show
plt.tight_layout()
plt.show()
