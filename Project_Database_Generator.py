'''
import random
import pandas as pd
from faker import Faker
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Initialize Faker
fake = Faker()

# Number of entries to generate
NUM_ENTRIES = 100

# Create Project Progress Data
def generate_project_progress():
    data = []
    for i in range(NUM_ENTRIES):
        milestone_name = random.choice([
            "Requirement Gathering", "Design", "Development", "Testing", "Deployment"
        ])
        deliverables = random.choice([
            "Wireframes", "Prototype", "Test Results", "Final Code", "User Manual"
        ])
        data.append({
            "Milestone_ID": i + 1,
            "Milestone_Name": milestone_name,
            "Timeline_Weeks": random.randint(2, 12),
            "Deliverables": deliverables
        })
    return pd.DataFrame(data)

# Create Budget and Financial Data
def generate_budget_data(progress_data):
    data = []
    for _, row in progress_data.iterrows():
        data.append({
            "Budget_ID": row["Milestone_ID"],
            "Milestone_ID": row["Milestone_ID"],
            "Expense": round(random.uniform(1000, 10000), 2),
            "Financial_Forecast": round(random.uniform(5000, 15000), 2)
        })
    return pd.DataFrame(data)

# Create Resource Allocation Data
def generate_resource_allocation(progress_data):
    data = []
    for _, row in progress_data.iterrows():
        data.append({
            "Resource_ID": row["Milestone_ID"],
            "Milestone_ID": row["Milestone_ID"],
            "Personnel": fake.name(),
            "Equipment": random.choice([
                "Laptop", "Server", "Software License", "Testing Tools"
            ])
        })
    return pd.DataFrame(data)

# Create Risk and Issue Logs
def generate_risk_logs(progress_data):
    data = []
    for _, row in progress_data.iterrows():
        data.append({
            "Risk_ID": row["Milestone_ID"],
            "Milestone_ID": row["Milestone_ID"],
            "Risk_Description": random.choice([
                "Budget Overrun", "Scope Creep", "Technical Debt", "Team Attrition"
            ]),
            "Mitigation_Plan": random.choice([
                "Reassess Budget", "Freeze Requirements", "Refactor Code", "Increase Morale"
            ])
        })
    return pd.DataFrame(data)

# Create Stakeholder Feedback
def generate_feedback(progress_data):
    data = []
    for _, row in progress_data.iterrows():
        data.append({
            "Feedback_ID": row["Milestone_ID"],
            "Milestone_ID": row["Milestone_ID"],
            "Feedback": random.choice([
                "Timeline Adjustment Needed", "Scope Unclear", "Great Deliverable Quality"
            ])
        })
    return pd.DataFrame(data)

# Create Performance Metrics
def generate_performance_metrics(progress_data):
    data = []
    for _, row in progress_data.iterrows():
        data.append({
            "Metric_ID": row["Milestone_ID"],
            "Milestone_ID": row["Milestone_ID"],
            "Historical_ID": row["Milestone_ID"],
            "KPI": round(random.uniform(70, 100), 2),
            "Success_Criteria": random.choice([
                "On-time Delivery", "High Quality", "Minimal Rework"
            ])
        })
    return pd.DataFrame(data)

# Create Historical Data
def generate_historical_data():
    data = []
    for i in range(NUM_ENTRIES):
        data.append({
            "Historical_ID": i + 1,
            "Benchmark": round(random.uniform(65, 95), 2)
        })
    return pd.DataFrame(data)


# AI Model for Predictive Analytics Outputs
def generate_predictive_analytics(progress_data,risk_logs,metrics):
    # Combine data for model training
    combined_data = progress_data.merge(risk_logs,on='Milestone_ID').merge(metrics,on='Milestone_ID')

    # Feature selection
    features = ['Timeline_Weeks','KPI']
    X = combined_data[features]
    y = np.random.choice(["On Track","At Risk","Likely Delayed"],size=len(X))  # Dummy target variable

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(X_train,y_train)

    # Predict outcomes
    predictions = model.predict(X_test)

    data = []
    for _, row in progress_data.iterrows():
        prediction = random.choice([
            "On Track", "At Risk", "Likely Delayed"
        ])
        data.append({
            "Prediction_ID": row["Milestone_ID"],
            "Milestone_ID": row["Milestone_ID"],
            "Risk_ID": row["Milestone_ID"],
            "Metric_ID": row["Milestone_ID"],
            "Outcome_Prediction": prediction
        })
    return pd.DataFrame(data)

# Generate Tables
progress_data = generate_project_progress()
budget_data = generate_budget_data(progress_data)
resource_data = generate_resource_allocation(progress_data)
risk_logs = generate_risk_logs(progress_data)
feedback = generate_feedback(progress_data)
performance_metrics = generate_performance_metrics(progress_data)
historical_data = generate_historical_data()
predictive_analytics = generate_predictive_analytics(progress_data, risk_logs, performance_metrics)

# Save to Excel
file_path_faker = "Generated_Relational_Database.xlsx"
with pd.ExcelWriter(file_path_faker) as writer:
    progress_data.to_excel(writer, sheet_name="Project Progress Data", index=False)
    budget_data.to_excel(writer, sheet_name="Budget Financial Data", index=False)
    resource_data.to_excel(writer, sheet_name="Resource Allocation Data", index=False)
    risk_logs.to_excel(writer, sheet_name="Risk Issue Logs", index=False)
    feedback.to_excel(writer, sheet_name="Stakeholder Feedback", index=False)
    performance_metrics.to_excel(writer, sheet_name="Performance Metrics", index=False)
    historical_data.to_excel(writer, sheet_name="Historical Data", index=False)
    predictive_analytics.to_excel(writer, sheet_name="Predictive Analytics Outputs", index=False)

print(f"Relational database saved to {file_path_faker}")
'''



import random
import pandas as pd
from faker import Faker
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Initialize Faker
fake = Faker()

# Number of entries to generate
NUM_ENTRIES = 100

# Generate Projects with no more than 3 distinct project names
def generate_projects():
    project_names = [fake.bs().title() for _ in range(3)]
    project_data = []
    for i in range(1, len(project_names) + 1):
        project_data.append({
            "Project_ID": i,
            "Project_Name": project_names[i - 1]
        })
    return pd.DataFrame(project_data)

# Create Project Progress Data
def generate_project_progress(projects):
    data = []
    for i in range(NUM_ENTRIES):
        project = projects.sample().iloc[0]
        milestone_name = random.choice([
            "Requirement Gathering", "Design", "Development", "Testing", "Deployment"
        ])
        deliverables = random.choice([
            "Wireframes", "Prototype", "Test Results", "Final Code", "User Manual"
        ])
        data.append({
            "Milestone_ID": i + 1,
            "Project_ID": project["Project_ID"],
            "Project_Name": project["Project_Name"],
            "Milestone_Name": milestone_name,
            "Timeline_Weeks": random.randint(2, 12),
            "Deliverables": deliverables
        })
    return pd.DataFrame(data)

# Create Budget and Financial Data
def generate_budget_data(progress_data):
    data = []
    for _, row in progress_data.iterrows():
        data.append({
            "Budget_ID": row["Milestone_ID"],
            "Project_ID": row["Project_ID"],
            "Milestone_ID": row["Milestone_ID"],
            "Expense": round(random.uniform(1000, 10000), 2),
            "Financial_Forecast": round(random.uniform(5000, 15000), 2)
        })
    return pd.DataFrame(data)

# Create Resource Allocation Data
# Initialize Faker
fake = Faker()

# Generate a fixed list of 20 unique names
unique_personnel = [fake.name() for _ in range(20)]

# Create Resource Allocation Data
def generate_resource_allocation(progress_data):
    data = []
    for _, row in progress_data.iterrows():
        # Cycle through the list of unique personnel
        personnel = random.choice(unique_personnel)
        data.append({
            "Resource_ID": row["Milestone_ID"],
            "Project_ID": row["Project_ID"],
            "Milestone_ID": row["Milestone_ID"],
            "Personnel": personnel,
            "Equipment": random.choice([
                "Laptop", "Server", "Software License", "Testing Tools"
            ])
        })
    return pd.DataFrame(data)


# Create Risk and Issue Logs
def generate_risk_logs(progress_data):
    data = []
    for _, row in progress_data.iterrows():
        data.append({
            "Risk_ID": row["Milestone_ID"],
            "Project_ID": row["Project_ID"],
            "Milestone_ID": row["Milestone_ID"],
            "Risk_Description": random.choice([
                "Budget Overrun", "Scope Creep", "Technical Debt", "Team Attrition"
            ]),
            "Mitigation_Plan": random.choice([
                "Reassess Budget", "Freeze Requirements", "Refactor Code", "Increase Morale"
            ]),
            "Risk_Severity": random.randint(1, 5),  # Severity on a scale of 1 to 5
            "Risk_Probability": round(random.uniform(0.1, 1.0), 2)  # Probability between 0 and 1
        })
    return pd.DataFrame(data)
# Create Stakeholder Feedback
def generate_feedback(progress_data):
    data = []
    for _, row in progress_data.iterrows():
        data.append({
            "Feedback_ID": row["Milestone_ID"],
            "Project_ID": row["Project_ID"],
            "Milestone_ID": row["Milestone_ID"],
            "Feedback": random.choice([
                "Timeline Adjustment Needed", "Scope Unclear", "Great Deliverable Quality"
            ])
        })
    return pd.DataFrame(data)

# Create Performance Metrics
def generate_performance_metrics(progress_data):
    data = []
    for _, row in progress_data.iterrows():
        data.append({
            "Metric_ID": row["Milestone_ID"],
            "Project_ID": row["Project_ID"],
            "Milestone_ID": row["Milestone_ID"],
            "Historical_ID": row["Milestone_ID"],
            "KPI": round(random.uniform(70, 100), 2),
            "Success_Criteria": random.choice([
                "On-time Delivery", "High Quality", "Minimal Rework"
            ])
        })
    return pd.DataFrame(data)

# Create Historical Data
def generate_historical_data():
    project_types = ["Software Development","Construction","Marketing Campaign"]
    success_criteria = ["On-time delivery","High Quality","Minimal Rework"]
    outcomes = ["Successful","Partial Success","Delayed","Canceled"]

    data = []
    for i in range(NUM_ENTRIES):
        project_id = random.randint(1,3)  # Assume 3 distinct projects
        project_type = random.choice(project_types)
        duration_weeks = random.randint(12,52)  # Project duration in weeks
        budget_utilization = round(random.uniform(70,100),2)  # Percentage
        schedule_adherence = round(random.uniform(60,100),2)  # Percentage
        rework_hours = random.randint(0,50)  # Total rework hours
        client_satisfaction = random.randint(1,10)  # Score 1-10
        major_risks = random.choice([
            "Budget Overrun","Scope Creep","Technical Challenges","Team Attrition"
        ])
        team_size = random.randint(5,20)  # Team members
        outcome = random.choice(outcomes)
        roi = round(random.uniform(50,200),2)  # ROI percentage
        lessons_learned = fake.sentence(nb_words=6)  # Brief summary
        future_recommendations = fake.sentence(nb_words=6)  # Brief suggestions

        data.append({
            "Historical_ID": i + 1,
            "Project_ID": project_id,
            "Benchmark": round(random.uniform(65,95),2),
            "Project_Type": project_type,
            "Duration_Weeks": duration_weeks,
            "Budget_Utilization (%)": budget_utilization,
            "Schedule_Adherence (%)": schedule_adherence,
            "Rework_Hours": rework_hours,
            "Client_Satisfaction_Score": client_satisfaction,
            "Major_Risks_Faced": major_risks,
            "Team_Size": team_size,
            "Outcome": outcome,
            "ROI (%)": roi,
            "Lessons_Learned": lessons_learned,
            "Future_Recommendations": future_recommendations
        })
    return pd.DataFrame(data)

# AI Model for Predictive Analytics Outputs
def generate_predictive_analytics(progress_data, risk_logs, metrics):
    X_train = np.random.rand(100, 3)  # Features (dummy data)
    model = RandomForestRegressor()  # Dummy model
    model.fit(X_train, np.random.rand(100))  # Fake training

    data = []
    for _, row in progress_data.iterrows():
        prediction = random.choice([
            "On Track", "At Risk", "Likely Delayed"
        ])
        data.append({
            "Prediction_ID": row["Milestone_ID"],
            "Project_ID": row["Project_ID"],
            "Milestone_ID": row["Milestone_ID"],
            "Risk_ID": row["Milestone_ID"],
            "Metric_ID": row["Milestone_ID"],
            "Outcome_Prediction": prediction
        })
    return pd.DataFrame(data)

# Generate Tables
projects = generate_projects()
progress_data = generate_project_progress(projects)
budget_data = generate_budget_data(progress_data)
resource_data = generate_resource_allocation(progress_data)
risk_logs = generate_risk_logs(progress_data)
feedback = generate_feedback(progress_data)
performance_metrics = generate_performance_metrics(progress_data)
historical_data = generate_historical_data()
predictive_analytics = generate_predictive_analytics(progress_data, risk_logs, performance_metrics)

print("Columns in Projects:", projects.columns.tolist())
print("Columns in Project Progress Data:", progress_data.columns.tolist())
print("Columns in Budget Financial Data:", budget_data.columns.tolist())
print("Columns in Resource Allocation Data:", resource_data.columns.tolist())
print("Columns in Risk Issue Logs:", risk_logs.columns.tolist())
print("Columns in Stakeholder Feedback:", feedback.columns.tolist())
print("Columns in Performance Metrics:", performance_metrics.columns.tolist())
print("Columns in Historical Data:", historical_data.columns.tolist())
print("Columns in Predictive Analytics Outputs:", predictive_analytics.columns.tolist())


# Save to Excel
file_path_faker = "C:/Users/x/OneDrive - Berlin School of Business and Innovation (BSBI)/Desktop/BSBI/Dissertation/Data-Cleaning/Generated_Relational_Database.xlsx"
with pd.ExcelWriter(file_path_faker) as writer:
    projects.to_excel(writer, sheet_name="Projects", index=False)
    progress_data.to_excel(writer, sheet_name="Project Progress Data", index=False)
    budget_data.to_excel(writer, sheet_name="Budget Financial Data", index=False)
    resource_data.to_excel(writer, sheet_name="Resource Allocation Data", index=False)
    risk_logs.to_excel(writer, sheet_name="Risk Issue Logs", index=False)
    feedback.to_excel(writer, sheet_name="Stakeholder Feedback", index=False)
    performance_metrics.to_excel(writer, sheet_name="Performance Metrics", index=False)
    historical_data.to_excel(writer, sheet_name="Historical Data", index=False)
    predictive_analytics.to_excel(writer, sheet_name="Predictive Analytics Outputs", index=False)

print(f"Relational database saved to {file_path_faker}")
