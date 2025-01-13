import pandas as pd

# Attempt to read the CSV file with a different encoding
df = pd.read_csv("*/survey_data.csv", encoding='latin1')

# Select first 4 and last 3 columns
selected_columns = df.iloc[:, list(range(4)) + list(range(-3, 0))]

# Create a new DataFrame with these columns
df_cleaned = pd.DataFrame(selected_columns)

# Drop rows with any null values in the selected columns
df_cleaned = df_cleaned.dropna(axis=0, how='any')

# Count rows after cleaning
row_count = len(df_cleaned)
print(f"Number of rows in the cleaned DataFrame: {row_count}")

# Save the cleaned DataFrame to a CSV file
df_cleaned.to_csv("*/Open_EndedQuestionAnswers.csv", index=False, encoding='utf-8')

print("New CSV file has been created with the selected columns, no null values, and fixed encoding.")
