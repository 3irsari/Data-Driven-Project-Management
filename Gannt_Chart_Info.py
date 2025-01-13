from faker import Faker
import pandas as pd
from datetime import datetime, timedelta
import random

# Initialize Faker and seed for reproducibility
faker = Faker()
Faker.seed(42)

# Load existing project data from Excel
file_path = "C:/Users/x/OneDrive - Berlin School of Business and Innovation (BSBI)/Desktop/BSBI/Dissertation/Data-Cleaning/Generated_Relational_Database.xlsx"
projects_df = pd.read_excel(file_path, sheet_name="Projects")

# Define project details (IDs, start dates, and durations)
projects = []
for _, row in projects_df.iterrows():
    projects.append({
        "Project ID": row["Project_ID"],
        "Start Date": datetime(2024, 11, 1),  # Example start date, adjust as needed
        "Duration Days": random.randint(120, 180)  # Example duration, adjust as needed
    })

# Predefined set of task names
task_names = ["Deployment", "Design", "Development", "Requirement Gathering", "Testing"]

# Generate task data for each project
tasks = []
total_tasks_needed = 100
tasks_per_project = total_tasks_needed // len(projects)

for project in projects:
    project_id = project["Project ID"]
    start_date = project["Start Date"]
    duration = project["Duration Days"]
    task_duration = duration // tasks_per_project

    for i in range(1, tasks_per_project + 1):
        task_start_date = start_date + timedelta(days=(i - 1) * task_duration)
        task_end_date = task_start_date + timedelta(days=task_duration - 1)
        tasks.append({
            "Task ID": f"T{project_id}{i}",
            "Project ID": project_id,
            "Task Name": random.choice(task_names),  # Choose from predefined task names
            "Task Start Date": task_start_date,
            "Task End Date": task_end_date,  # Allow end date to be later than today
            "Task Status": "Ongoing" if task_end_date >= datetime.today() else "Completed"
        })

# If there are any remaining tasks needed to reach 100, add them to the last project
remaining_tasks = total_tasks_needed - len(tasks)
if remaining_tasks > 0:
    project = projects[-1]
    project_id = project["Project ID"]
    start_date = project["Start Date"]
    duration = project["Duration Days"]
    task_duration = duration // (tasks_per_project + remaining_tasks)

    for i in range(tasks_per_project + 1, tasks_per_project + remaining_tasks + 1):
        task_start_date = start_date + timedelta(days=(i - 1) * task_duration)
        task_end_date = task_start_date + timedelta(days=task_duration - 1)
        tasks.append({
            "Task ID": f"T{project_id}{i}",
            "Project ID": project_id,
            "Task Name": random.choice(task_names),
            "Task Start Date": task_start_date,
            "Task End Date": task_end_date,  # Allow end date to be later than today
            "Task Status": "Ongoing" if task_end_date >= datetime.today() else "Completed"
        })

# Convert to a DataFrame
tasks_df = pd.DataFrame(tasks)

# Save to CSV for Tableau
output_path = "C:/Users/x/OneDrive - Berlin School of Business and Innovation (BSBI)/Desktop/BSBI/Dissertation/Data-Cleaning/gantt_chart_data.csv"
tasks_df.to_csv(output_path, index=False)

print("Gantt chart data saved successfully.")
