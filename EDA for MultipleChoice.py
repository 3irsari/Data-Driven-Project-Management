import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# Load the dataset
data_path = '*/transformed_MultipleChoice_survey_responses.csv'
data = pd.read_csv(data_path)

# Ensure only relevant columns are used
if 'question' in data.columns and 'answer' in data.columns:
    # Initialize a dictionary to store answer counts for each question
    answer_counts_dict = {}

    # Count the answers for each distinct question
    for question in data['question'].unique():
        answer_counts = data[data['question'] == question]['answer'].value_counts()
        answer_counts_dict[question] = answer_counts

    # Convert the dictionary to a DataFrame for plotting
    answer_counts_df = pd.DataFrame(answer_counts_dict).fillna(0)

    # Plot each question's answer count side-by-side
    n_questions = len(answer_counts_df.columns)
    rows = (n_questions + 1) // 3  # Adjust rows for layout

    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 5))
    axes = axes.flatten()

    for i, question in enumerate(answer_counts_df.columns):
        # Filter out answers with zero counts
        filtered_data = answer_counts_df[[question]].loc[answer_counts_df[question] > 0].sort_values(by=question, ascending=False)
        if not filtered_data.empty:
            sns.barplot(x=filtered_data.index, y=filtered_data[question], ax=axes[i], palette='crest')
            axes[i].set_title(f"Answer Counts for {question}")
            axes[i].set_xlabel("Answers")
            axes[i].set_ylabel("Count")

            # Wrap and truncate x-axis labels
            labels = [
                textwrap.fill(label.split('(')[0].strip(),25)  # Split at '(' and take the first part
                for label in filtered_data.index
            ]
            axes[i].set_xticklabels(labels,rotation=45,ha='right')

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(hspace=1)  # Increase hspace to add more space between rows
    plt.show()

    # Drop the 'data_types_collected' column before computing the correlation matrix
    if 'data_types_collected' in answer_counts_df.columns:
        answer_counts_df = answer_counts_df.drop(columns=['data_types_collected'])
    correlation_matrix = answer_counts_df.corr()

    # Filter correlations with absolute value greater than 0.9
    high_correlation = correlation_matrix[(correlation_matrix.abs() > 0.09) & (correlation_matrix != 1.0)]

    # Plot the correlation heatmap
    plt.figure(figsize=(12,8))
    sns.heatmap(high_correlation,annot=True,cmap='coolwarm',fmt=".2f",cbar=True,linewidths=0.5)
    plt.title(" High Correlation Matrix of Answer Counts Between Questions")
    plt.xlabel("Questions")
    plt.ylabel("Questions")
    plt.xticks(rotation=45,ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("The dataset does not contain 'question' and 'answer' columns.")

# Descriptive Statistics
print("\nDescriptive Statistics for Numeric Columns:")
pd.set_option('display.max_columns', None)
print(data.describe())

# Summary Statistics for Categorical Variables
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
 #   print(f"\n{column} Value Counts:")
    print(data[column].value_counts())
