from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
from textwrap import fill

# Function to draw a single step in the pipeline
def draw_pipeline_step(ax, text, x, y, width, height, color):
    rect = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.3", edgecolor="black", facecolor=color, lw=1.5)
    ax.add_patch(rect)
    wrapped_text = fill(text, width=20)  # Wrap text to fit within the box
    ax.text(x + width / 2, y + height / 2, wrapped_text, ha="center", va="center", fontsize=10)

# Create the pipeline
fig, ax = plt.subplots(figsize=(12, 6))

# Define step details
steps = [
    "Data Collection",
    "Data Cleaning & Preprocessing",
    "Model Training",
    "Model Evaluation",
    "Prediction Generation",
    "Decision-Making"
]
colors = ["#FFDDC1", "#FFABAB", "#FFC3A0", "#D5AAFF", "#85E3FF", "#B9FBC2"]

# Draw each step
x_start = 1
y_start = 4
box_width = 3.5  # Increase box width to accommodate text
box_height = 1.5  # Increase box height for better text fit
spacing = 5  # Increase spacing between boxes
for i, step in enumerate(steps):
    draw_pipeline_step(ax, step, x_start + i * spacing, y_start, box_width, box_height, colors[i])

# Draw arrows between steps
for i in range(len(steps) - 1):
    ax.annotate("", xy=(x_start + (i + 1) * spacing - 0.5, y_start + box_height / 2),
                xytext=(x_start + i * spacing + box_width + 0.5, y_start + box_height / 2),
                arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.5))

# Set limits to ensure everything is visible
ax.set_xlim(0, x_start + len(steps) * spacing)
ax.set_ylim(0, y_start + box_height + 1)

# Add labels
plt.title("Pipeline: Applying Machine Learning to Project Management", fontsize=14, pad=20)
ax.axis("off")
plt.show()
