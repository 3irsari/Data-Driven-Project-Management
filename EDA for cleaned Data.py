import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = '*/cleaned_survey_data.csv'  # Replace with the correct file path
data = pd.read_csv(file_path)

# General Dataset Overview
print("Dataset Shape:", data.shape)
print("Columns in Dataset:", data.columns)
print("\nMissing Values:")
print(data.isnull().sum())

'''
def explode_multiple_choice(df, columns_valid_choices):

    expanded_rows = []

    for _,row in df.iterrows():
         base_row = row[['timestamp','department','job_title']].to_dict()  # Keep first three columns
         for column,valid_choices in columns_valid_choices.items():
             if pd.notna(row[column]):  # Check if the cell is not NaN
                matched = False
                for choice in valid_choices:
                     if choice in row[column]:  # Check if the valid choice is in the cell's text
                          new_row = base_row.copy()
                          new_row['question'] = column  # Add the question column
                          new_row['answer'] = choice  # Add the valid choice as the answer
                          expanded_rows.append(new_row)
                          matched = True

                 # If no match was found, classify as "Others"
                if not matched:
                     new_row = base_row.copy()
                     new_row['question'] = column
                     new_row['answer'] = 'Others'
                     expanded_rows.append(new_row)

    # Create a new DataFrame from expanded rows
    return pd.DataFrame(expanded_rows)


# Explode the data based on valid choices for multiple columns
expanded_df = explode_multiple_choice(df, columns_valid_choices)


'''

# Descriptive Statistics
print("\nDescriptive Statistics for Numeric Columns:")
pd.set_option('display.max_columns', None)
print(data.describe())

# Plot 1: Missing Data Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

# Plot 2: Categorical Variable Distributions
#categorical_columns = data.select_dtypes(include=['object']).columns
categorical_columns = [ 'pm_tools', 'success_assessment',
       'pa_future_adoption', 'department_group', 'experience_bin']
#print(categorical_columns)
'''
for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, y=column, order=data[column].value_counts().index, palette="crest")
    plt.title(f"Distribution of {column}")
    plt.xlabel("Count")
    plt.ylabel(column)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
'''

# Number of columns and rows for subplots
n_cols = 2  # Number of columns in the grid
n_rows = (len(categorical_columns) + n_cols - 1) // n_cols  # Calculate rows based on number of plots

# Create the subplot grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))  # Adjust figure size

# Flatten the axes array for easy iteration (handles grids of any size)
axes = axes.flatten()

# Plot each categorical column
for idx, column in enumerate(categorical_columns):
    sns.countplot(
        data=data,
        y=column,
        order=data[column].value_counts().index,
        palette="crest",
        ax=axes[idx]
    )
    axes[idx].set_title(f"Distribution of {column}")
    axes[idx].set_xlabel("Count")
    axes[idx].set_ylabel(column)
    axes[idx].tick_params(axis='x', labelsize=10)
    axes[idx].tick_params(axis='y', labelsize=10)

# Remove any unused subplots
for idx in range(len(categorical_columns), len(axes)):
    fig.delaxes(axes[idx])

# Adjust layout
plt.tight_layout()
plt.show()

# Plot 3: Histograms for Numeric Variables
numeric_columns = data.select_dtypes(include=['number']).columns
'''
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, bins=20, color='blue')
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
'''

# Number of columns and rows for subplots
n_cols = 3  # Number of columns in the grid
n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Calculate rows based on number of plots

# Create the subplot grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))  # Adjust figure size

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot each numeric column
for idx, column in enumerate(numeric_columns):
    sns.histplot(data[column], kde=True, bins=20, color='blue', ax=axes[idx])
    axes[idx].set_title(f"Distribution of {column}", fontsize=14)
    axes[idx].set_xlabel(column, fontsize=12)
    axes[idx].set_ylabel("Frequency", fontsize=12)
    axes[idx].tick_params(axis='x', labelsize=10)
    axes[idx].tick_params(axis='y', labelsize=10)

# Remove any unused subplots
for idx in range(len(numeric_columns), len(axes)):
    fig.delaxes(axes[idx])

# Adjust layout
plt.tight_layout()
plt.show()


# Plot 4: Pairwise Correlation Heatmap for Numeric Variables
plt.figure(figsize=(12, 8))
correlation_matrix = data[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()

# Plot 4.2: correlation more than |0.5|
mask = (np.abs(correlation_matrix) <= 0.60) & (np.abs(correlation_matrix) != 1)
np.fill_diagonal(mask.values, True)  # Set the diagonal to True to exclude it
filtered_correlation_matrix = correlation_matrix.mask(mask)
plt.figure(figsize=(12, 8))
sns.heatmap(filtered_correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, mask=mask)
plt.title("Filtered Correlation Heatmap (|correlation| > 0.5)")
plt.tight_layout()
plt.show()

# Plot 5: Boxplots of Numeric Columns Grouped by a Categorical Variable
grouping_column = 'department_group'  # Replace with a column for grouping (e.g., department or job title)
'''
if grouping_column in data.columns:
    for column in numeric_columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data, x=grouping_column, y=column, palette="pastel")
        plt.title(f"{column} Distribution by {grouping_column}")
        plt.xticks(rotation=45, fontsize=10)
        plt.xlabel(grouping_column)
        plt.ylabel(column)
        plt.show()
'''
# Number of columns and rows for subplots
n_cols = 3  # Number of columns in the grid
n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Calculate rows based on number of plots

# Create the subplot grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))  # Adjust figure size

# Flatten the axes array for easy iteration (handles grids of any size)
axes = axes.flatten()

# Plot each numeric column with a boxplot
for idx, column in enumerate(numeric_columns):
    sns.boxplot(data=data, x=grouping_column, y=column, palette="pastel", ax=axes[idx])
    axes[idx].set_title(f"{column} Distribution by {grouping_column}", fontsize=14)
    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, fontsize=10)
    axes[idx].set_xlabel(grouping_column, fontsize=12)
    axes[idx].set_ylabel(column, fontsize=12)
    axes[idx].tick_params(axis='x', labelsize=10)
    axes[idx].tick_params(axis='y', labelsize=10)

# Remove any unused subplots
for idx in range(len(numeric_columns), len(axes)):
    fig.delaxes(axes[idx])

# Adjust layout for better spacing between plots
plt.tight_layout()
plt.show()

# Plot 6: Scatter Plot of Two Key Numeric Columns
scatter_columns = ['confidence_influence', 'ai_confidence_increase']  # Replace with your columns
if all(col in data.columns for col in scatter_columns):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x=scatter_columns[0], y=scatter_columns[1], hue='department_group', palette="deep")
    plt.title(f"Scatter Plot: {scatter_columns[0]} vs {scatter_columns[1]}")
    plt.xlabel(scatter_columns[0])
    plt.ylabel(scatter_columns[1])
    plt.legend(title="Department Group",loc='upper left',bbox_to_anchor=(1,1))  # Legend outside the plot
    plt.tight_layout()
    plt.show()

# Plot 7: Pie Chart for a Key Categorical Variable
key_column = ['pm_tools', 'success_assessment', 'pa_future_adoption' ]
'''for column in key_column:
    plt.figure(figsize=(8, 8))
    data[column].value_counts().plot.pie(
        autopct='%1.1f%%',startangle=140,colors=sns.color_palette('pastel')    )
    plt.title(f"Distribution of {column}")
    plt.xlabel("Count")
    plt.ylabel(column)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()'''
# Number of columns and rows for subplots
n_cols = 2  # Number of columns in the grid
n_rows = (len(key_column) + n_cols - 1) // n_cols  # Calculate rows based on number of plots

# Create the subplot grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))  # Adjust figure size

# Flatten the axes array for easy iteration (handles grids of any size)
axes = axes.flatten()

# Plot each column as a pie chart
for idx, column in enumerate(key_column):
    # Calculate value counts and percentages
    value_counts = data[column].value_counts()
    total = value_counts.sum()
    percentages = value_counts / total * 100

    # Separate values into main categories and "Other"
    main_categories = percentages[percentages >= 8]
    other_categories = percentages[percentages < 8]
    other_sum = other_categories.sum()

    # Combine into a new series
    combined = main_categories.copy()
    if other_sum > 0:
        combined['Other'] = other_sum

    # Plot the pie chart
    combined.plot.pie(
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette('pastel'),
        ax=axes[idx]
    )
    axes[idx].set_title(f"Distribution of {column}", fontsize=14)
    axes[idx].set_ylabel("")  # Remove ylabel (unnecessary for pie chart)
    axes[idx].tick_params(axis='x', labelsize=10)
    axes[idx].tick_params(axis='y', labelsize=10)

# Remove any unused subplots
for idx in range(len(key_column), len(axes)):
    fig.delaxes(axes[idx])

# Adjust layout for better spacing between plots
plt.tight_layout()
plt.show()


# Summary Statistics for Categorical Variables
for column in categorical_columns:
 #   print(f"\n{column} Value Counts:")
    print(data[column].value_counts())

'''
# Define a function to create categorical distribution plots
def plot_categorical_distribution(column_name, title, xlabel, ylabel, color_palette="viridis"):
    """
    Plot the distribution of a categorical variable as a bar chart.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, y=column_name, palette=color_palette, order=data[column_name].value_counts().index)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

# Plot the distribution of departments (Professional Background)
plot_categorical_distribution(
    column_name="department",
    title="Distribution of Professional Backgrounds (Departments)",
    xlabel="Count",
    ylabel="Department",
    color_palette="crest"
)
'''
