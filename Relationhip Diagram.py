
import matplotlib.pyplot as plt
import networkx as nx

# Define the relationships between tables
relationships = [
    ("Project Progress Data", "Budget Financial Data"),
    ("Project Progress Data", "Resource Allocation Data"),
    ("Project Progress Data", "Risk Issue Logs"),
    ("Project Progress Data", "Stakeholder Feedback"),
    ("Project Progress Data", "Performance Metrics"),
    ("Performance Metrics", "Historical Data"),
    ("Predictive Analytics Outputs", "Project Progress Data"),
    ("Predictive Analytics Outputs", "Budget Financial Data"),
    ("Predictive Analytics Outputs", "Resource Allocation Data"),
    ("Predictive Analytics Outputs", "Risk Issue Logs"),
    ("Predictive Analytics Outputs", "Stakeholder Feedback"),
    ("Predictive Analytics Outputs", "Performance Metrics"),
]

# Create a directed graph
G = nx.DiGraph()

# Add relationships to the graph
G.add_edges_from(relationships)

# Draw the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42, k=2.2)  # Adjust 'k' for more spacing
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", edge_color="gray")
plt.title("Database Relationship Diagram", fontsize=16, fontweight="bold")
plt.show()