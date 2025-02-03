import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from textwrap import fill

# Initialize the graph
G = nx.DiGraph()

# Central node
central_node = "Integrating AI in Project Management"
G.add_node(central_node)

# Challenges and Opportunities
categories = ["Challenges", "Opportunities"]
G.add_nodes_from(categories)
G.add_edges_from([(central_node, category) for category in categories])

# Challenges sub-nodes
challenges = {
    "Data Quality and Availability": [
        "Inconsistent or incomplete data",
        "Data silos across organizations",
    ],
    "Stakeholder Resistance": [
        "Hesitation to trust AI-based decisions",
        "Concerns about job displacement",
    ],
    "Technical Limitations": [
        "Difficulty training accurate models",
        "Challenges integrating with tools",
    ],
    "Cost and Resources": [
        "High initial investment",
        "Need for skilled professionals",
    ],
    "Ethical and Privacy Concerns": [
        "Risks of bias in AI algorithms",
        "Compliance with regulations",
    ],
}

# Opportunities sub-nodes
opportunities = {
    "Improved Decision-Making": [
        "Data-driven forecasts",
        "Reduced uncertainty in planning",
    ],
    "Efficiency Gains": [
        "Automation of routine tasks",
        "Faster project adjustments",
    ],
    "Enhanced Stakeholder Confidence": [
        "Interactive project metrics visualizations",
        "Transparency in analytics results",
    ],
    "Scalability and Customization": [
        "Adaptable AI tools",
        "Continuous learning from diverse data",
    ],
    "Risk Mitigation": [
        "Early risk identification",
        "Better resource allocation",
    ],
}

# Add Challenges and Opportunities sub-nodes
for challenge, sub_challenges in challenges.items():
    G.add_node(challenge)
    G.add_edge("Challenges", challenge)
    for sub_challenge in sub_challenges:
        G.add_node(sub_challenge)
        G.add_edge(challenge, sub_challenge)

for opportunity, sub_opportunities in opportunities.items():
    G.add_node(opportunity)
    G.add_edge("Opportunities", opportunity)
    for sub_opportunity in sub_opportunities:
        G.add_node(sub_opportunity)
        G.add_edge(opportunity, sub_opportunity)

# Calculate node degrees
degrees = dict(G.degree())

# Normalize degrees for colormap
max_degree = max(degrees.values())
min_degree = min(degrees.values())
norm_degrees = [(degree - min_degree) / (max_degree - min_degree) for degree in degrees.values()]

# Choose a colormap
cmap = cm.get_cmap('Pastel1')

# Assign colors based on normalized degrees
node_colors = [cmap(norm_degree) for norm_degree in norm_degrees]

# Visualize the graph
plt.figure(figsize=(14, 14))  # Decrease figure size to reduce blank space

# Layout for nodes
pos = nx.spring_layout(G, seed=42, k=0.5)  # Adjust 'k' for tighter spacing
pos[central_node] = np.array([0, 0])  # Fix central node position at the center

# Wrap node labels
wrapped_labels = {node: fill(node, width=20) for node in G.nodes()}

# Draw nodes and edges with enhancements
nx.draw_networkx_nodes(G, pos, node_size=[3000 if node == central_node else 1500 for node in G.nodes()],
                       node_color=node_colors, alpha=0.9)
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", connectionstyle='arc3,rad=0.1')
nx.draw_networkx_labels(G, pos, labels=wrapped_labels, font_size=12, font_color="black", font_family="Arial")  # Increase font size

# Add a legend
plt.legend(['Central Node', 'Category', 'Sub-category'], loc='upper right')

plt.title("Challenges and Opportunities of Integrating AI in Project Management", fontsize=20)  # Increase title font size
plt.axis("off")
plt.show()
