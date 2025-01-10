# Importing Required Libraries
import pandas as pd
from spellchecker import SpellChecker
import chardet


# ===================== Step 1: Data Loading =====================
def load_data(file_path, encoding='utf-8'):
    """Load survey data from CSV and remove last three columns."""
    df = pd.read_csv(file_path, encoding=encoding)
    df = df.iloc[:, :-3]  # Remove the last three columns
    # Print column names to verify
    #print("Columns in the DataFrame:",df.columns)

    return df
# ===================== Step 2: Column Renaming =====================
def rename_columns(df, aliases):
    """Rename columns based on the provided aliases."""
    df.rename(columns=aliases, inplace=True)
    return df

# ===================== Step 3: Text Preprocessing =====================
def correct_typos(text, spell):
    """Correct typos in text using a spell-checker."""
    if pd.isnull(text):
        return text
    words = text.split()
    corrected = [spell.correction(word) if word in spell else word for word in words]
    return " ".join(corrected)

def preprocess_text_columns(df, spell):
    """Apply typo correction to specific text columns."""
    df['department'] = df['department'].str.upper().apply(lambda x: correct_typos(x, spell))
    df['job_title'] = df['job_title'].str.upper().apply(lambda x: correct_typos(x, spell))
    return df

# ===================== Step 4: Filtering Data =====================
def filter_non_mandatory(df):
    """Filter out rows with missing values in non-mandatory columns."""
    non_mandatory_columns = df.columns[df.isnull().any()].tolist()
    print(f"Non-mandatory columns with missing values: {non_mandatory_columns}")
    return df

# ===================== Step 5: Grouping and Standardizing Responses =====================
def group_departments(df, department_groups):
    """Group departments into standardized categories."""
    def map_department(dept):
        for group, depts in department_groups.items():
            if dept in depts:
                return group
        return "Other"
    df['department_group'] = df['department'].apply(map_department)
    return df

def group_job_titles(df, job_title_groups):
    """Group job titles into standardized categories."""
    def map_job_title(title):
        for group, titles in job_title_groups.items():
            if title in titles:
                return group
        return "Uncategorized / Specialized Roles"
    df['job_title_group'] = df['job_title'].apply(map_job_title)
    return df

# ===================== Step 6: Bin Experience Years =====================
def bin_experience_years(df):
    """Bin experience years into predefined ranges."""
    bins = [0, 2, 5, 10, 15, float('inf')]
    labels = ['0-2 years', '3-5 years', '6-10 years', '11-15 years', '16+ years']
    df['experience_bin'] = pd.cut(df['experience_years'], bins=bins, labels=labels, right=False)
    return df

# ===================== Step 7: Explode Multiple Choice Columns =====================
def explode_multiple_choice(df, columns_valid_choices):
    """Explode multiple choice columns into separate rows."""
    expanded_rows = []
    for _, row in df.iterrows():
        base_row = row[['timestamp', 'department', 'job_title']].to_dict()  # Keep first three columns
        for column, valid_choices in columns_valid_choices.items():
            if pd.notna(row[column]):  # Check if the cell is not NaN
                for choice in valid_choices:
                    if choice in row[column]:  # Check if the valid choice is in the cell's text
                        new_row = base_row.copy()
                        new_row['question'] = column  # Add the question column
                        new_row['answer'] = choice  # Add the valid choice as the answer
                        expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)

# ===================== Step 8: Main Execution =====================
def main():
    file_path = "C:/Users/x/OneDrive - Berlin School of Business and Innovation (BSBI)/Desktop/BSBI/Dissertation/Data-Cleaning/survey_data.csv"
    aliases = {
        'Timestamp': 'timestamp',
        'Which department are you working in? ': 'department',
        'What is your job title? ': 'job_title',
        'How many years have you been working professionally ': 'experience_years',
        'What type of project management tools do you currently use? ': 'pm_tools',
        'How often do you use AI-based decision-making tools in your projects? (Rate from 1 = Never to 10 = Always) ': 'ai_usage_rate',
        'How do you currently assess and project the success of your projects? ': 'success_assessment',
        'What types of data do you collect during projects? (Select all that apply) ': 'data_types_collected',
        'What are the most common challenges you face in predicting project outcomes?  (Select all that apply) ': 'outcome_challenges',
        'Which aspects of project data are most critical in assessing project success? (Select all that apply) ': 'critical_data_aspects',
        'How frequently do you use predictive analytics in your projects? (Rate from 1 = Never to 10 = Always) ': 'pa_frequency',
        'What challenges have you faced integrating predictive analytics in project management? (Select all that apply) ': 'pa_integration_challenges',
        'How could predictive analytics be improved to support stakeholder confidence more effectively? (Select all that apply) ': 'pa_stakeholder_improvement_ideas',
        'What support or resources would help you use predictive analytics more effectively? (Select all that apply) ': 'pa_support_resources',
        'How do you rate the overall effectiveness of predictive analytics in improving project outcomes? ': 'pa_effectiveness',
        'Does using data-driven projections influence your confidence in project success? (Rate from 1 = Not at all to 10 = Very much) ': 'confidence_influence',
        'How effectively do you communicate data-driven projections to stakeholders? (Rate from 1 = Not effective to 10 = Very effective) ': 'stakeholder_communication_effectiveness',
        'To what degree do stakeholders trust projections from predictive analytics? (Rate from 1 = No trust to 10 = Full trust) ': 'stakeholder_trust',
        'What factors most influence stakeholders trust in predictive analytics? (Select all that apply) ': 'stakeholder_pa_trust_factors',
        'What are your concerns (if any) about using machine learning for project predictions?  (Select all that apply) ': 'ml_concerns',
        'What would encourage you from adopting predictive analytics tools in your projects?(Select all that apply) ': 'pa_adoption_encouragements',
        'What would discourage you from adopting predictive analytics tools in your projects?(Select all that apply) ': 'pa_adoption_discouragements',
        'Are you planning to adopt or expand predictive analytics in future projects? ': 'pa_future_adoption',
        'What factors most influence your confidence in estimates generated by predictive analytics? (Select all that apply) ': 'pa_confidence_factors',
        'Using AI-based tools has increased my confidence in project outcomes. (Rate from 1 = Not at all to 10 = Very much) ': 'ai_confidence_increase',
        'I feel that AI-based decision-making reduces uncertainties in project timelines. (Rate from 1 = Not at all to 10 = Very much) ': 'ai_uncertainty_reduction',
        'AI-based predictive analytics provide more accurate forecasts than traditional methods. (Rate from 1 = Not at all to 10 = Very much) ': 'ai_forecast_accuracy',
        'The efficiency of project management has improved since implementing AI tools. (Rate from 1 = Not at all to 10 = Very much) ': 'pm_efficiency_ai',
        'AI-based tools help in better risk assessment compared to traditional approaches. (Rate from 1 = Not at all to 10 = Very much) ': 'ai_risk_assessment',
        'Data visualization techniques in AI tools make it easier to understand project data. (Rate from 1 = Not at all to 10 = Very much) ': 'ai_data_visualization',
        'Overall, I prefer using AI-based tools over traditional project management methods. (Rate from 1 = Not at all to 10 = Very much) ': 'ai_tool_preference'
    }
    department_groups = {
        "Technology & IT": ["INFORMATION TECHNOLOGIES", "IT", "DEVELOPMENT TEAM", "DEVELOPMENT", "SOFTWARE",
                            "PAYMENT SYSTEMS", "IT DEPARTMENT (SOFTWARE DEVELOPMENT)", "COMPUTER SCIENCE",
                            "IT AND DIGITAL TRANSFORMATION", "SOFTWARE DEVELOPMENT"],
        "Analytics & Business Intelligence": ["DATA ANALYTICS", "SOLUTION DEVELOPMENT", "INTERNAL AUDIT",
                                              "DATA SCIENCE", "COMPANY", "MANAGEMENT", "EXECUTIVE MANAGEMENT"],
        "Project Management & Strategy": ["PROJECT MANAGEMENT OFFICE", "STRATEGIC PLANNING", "C-SUITE",
                                          "MANUFACTURING & AI INTEGRATION", "DATA SCIENCE", "PROJECT MANAGEMENT",
                                          "PMO SHOP", "PMO", "BOARD OF DIRECTORS", "GOVERNMENT"],
        "Customer-Facing & Sales": ["SALES", "TECHNICAL CUSTOMER SUPPORT", "HOSPITALITY", "ENTERTAINMENT",
                                    "EXPORT"],
        "Marketing & Communications": ["MARKETING", "COMMUNICATION", "WAS WORKING IN AN ADVERTISEMENT AGENCY"],
        "Research & Development": ["NEUROSCIENCE RESEARCH", "OKUMUYORUM", "UNIVERSITY", "ACADEMICS",
                                   "FACULTY OF ECONOMICS AND BUSINESS ADMINISTRATION", "R&D"],
        "Human Resources & People Services": ["HUMAN RESOURCES", "HUMAM RESOURCE", "PEOPLE SERVICES"],
        "Engineering & Specialized Roles": ["ENGINEERING", "CYBERSECURITY", "ENERGY TRADE DEPARTMENT",
                                            "INSTRUMENTATION", "SAP LOGISTICS AND E-INVOICING", "AGRICULTURAL SCIENCES",
                                            "MEDICINE", "YATIRIM DANIÅMANLIÄI"],
        "Product & Business Analysis": ["PRODUCT AND TECHNOLOGY", "BUSINESS DEVELOPMENT", "CONSULTANT",
                                        "OPERATIONS", "TRANSPORTATION"]
    }
    job_title_groups = {
        "Analytics & Business Intelligence": ["DATA ANALYST AND BUSINESS INTELLIGENCE SPECIALIST", "BI SPECIALIST",
                                              "INTERNAL AUDITOR DATA ANALYTICS", "DATA ANALYST", "BUSINESS ANALYST",
                                              "IT ANALYST", "PHENOTYPING AND DATA SPECIALIST", "DATA SPECIALIST"],
        "Customer Support & Operations": ["DIRECTOR OF CUSTOMER SUPPORT", "OPERATIONS MANAGER", "DISTRIBUTOR EXECUTIVE",
                                          "SALES SPECIALIST", "RECRUITMENT SPECIALIST", "LOGISTICS COORDINATOR"],
        "Engineering & Technology": ["ENERGY ENGINEER", "CYBERSECURITY CONSULTANT", "TEST ENGINEER",
                                     "PROGRAMMER", "SENIOR SOFTWARE ENGINEER", "DEVELOPER", "SENIOR SOFTWARE DEVELOPER",
                                     "SOFTWARE ENGINEER", "ARCHITECT"],
        "Project Management & Strategy": ["PROJECT MANAGER, AI SOLUTIONS", "PROJECT LEAD", "PROJECT MANAGER",
                                          "STRATEGIC PLANNING EXECUTIVE", "PRODUCT OWNER", "ASSISTANT PROJECT MANAGER",
                                          "IT DIRECTOR - SPECIAL PROJECTS", "SENIOR PROJECT MANAGER", "PMO", "SENIOR PM",
                                          "POLICY ANALYST"],
        "Marketing & Communications": ["MARKETER", "MARKETING MANAGER", "COMMUNICATION MANAGER","MARKETING ANALYST",
                                       "MARKETING SPECIALIST"],
        "Product & Business Development": ["PRODUCT MANAGER", "BUSINESS DEVELOPMENT", "OPERATIONS ANALYST"],
        "Consultancy & Coaching": ["PROFESSIONAL COACH", "CONSULTANT", "JUNIOR CONSULTANT", "PROFESSIONAL COACH",
                                   "SENIOR CONSULTANT", "SAP CONSULTANY", "KIDEMLI YATIRIM DANIÅMANI"],
        "Research & Academia": ["PHD STUDENT", "RESEARCH ASSISTANT", "SOSYOLOG", "ADJUNCT PROFESSOR", "SENIOR LECTURER",
                                "PHD", "LECTURER"],
        "Leadership & Executive Roles": ["VICE PRESIDENT", "GLOBAL HEAD OF ENGINEERING", "CEO", "CISO",
                                         "EXECUTIVE BOARD MEMBER", "SALES DIRECTOR", "COUNTRY DIRECTOR", "HOTEL MANAGER",
                                         "GENERAL MANAGER", "FOUNDER", "FILM PRODUCER", "EXPORT MANAGER"],
        "Uncategorized / Specialized Roles": ["YÃœKSEK ADLI BILIÅIM UZMANI", "ART DIRECTOR",  "ADMINISTRATOR",
                                              "PROGRAM TEAM LEADER", "INSTRUMENTATION TECHNICIAN", "MEDICAL DOCTOR" ]
    }

    # Detect the encoding of the file
    with open(file_path,'rb') as file:
        result = chardet.detect(file.read(10000))
        detected_encoding = result['encoding']
        print(f"Detected encoding: {detected_encoding}")

    # Step 1: Load data
    # Using the detected encoding to read the file
    df = load_data(file_path,encoding=detected_encoding)

    # Step 2: Rename columns
    df = rename_columns(df, aliases)

    # Step 3: Preprocess text columns
    spell = SpellChecker()
    df = preprocess_text_columns(df, spell)

    # Step 4: Filter non-mandatory columns
    df = filter_non_mandatory(df)

    # Step 5: Group and standardize responses
    df = group_departments(df, department_groups)
    df = group_job_titles(df, job_title_groups)

    # Step 6: Bin experience years
    df = bin_experience_years(df)

    # Step 7: Explode multiple choice columns
    columns_valid_choices = {
        'data_types_collected': [
            'Project progress data (e.g., milestones, timelines, and deliverables)',
            'Budget and financial data (e.g., costs, expenses, and financial forecasts)',
            'Resource allocation data (e.g., personnel, equipment, and material usage)',
            'Risk and issue logs (e.g., identified risks, mitigation plans, and resolutions)',
            'Stakeholder feedback (e.g., surveys, meeting notes, and communication records)',
            'Performance metrics (e.g., KPIs, success criteria, and variance analyses)',
            'Historical data (e.g., past project reports, lessons learned, and benchmarks)',
            'Predictive analytics outputs (e.g., project outcome predictions, trend analyses)',
            'Other:'
        ],
        'outcome_challenges': [
            'Timeline projections',
            'Budget',
            'Resource allocation',
            'Risk assessment',
            'Other:'
        ],
        'critical_data_aspects': [
            'Timelines',
            'Budget',
            'Resources',
            'Team performance',
            'Stakeholder satisfaction',
            'Other:'
        ],
        'pa_integration_challenges': [
            'Lack of expertise or training',
            'Difficulty in interpreting analytics results',
            'Resistance from stakeholders',
            'Limitations of available tools',
            'Other:'
        ],
        'pa_stakeholder_improvement_ideas': [
            'More user-friendly interfaces',
            'Increased model transparency',
            'Improved accuracy of projections',
            'Better integration with project management tools',
            'Other:'
        ],
        'pa_support_resources': [
            'Additional training',
            'Better software/tools',
            'Dedicated analytics support team',
            'Enhanced data sources',
            'Other:'
        ],
        'stakeholder_pa_trust_factors': [
            'Proven past accuracy of models',
            'Ease of understanding the model outputs',
            'Reliability of data sources',
            'Regular updates to projections',
            'Other:'
        ],
        'ml_concerns': [
            'Accuracy of predictions (e.g., fear of incorrect or misleading outputs)',
            'Data privacy and security (e.g., concerns about sensitive data being exposed)',
            'Complexity of implementation (e.g., difficulty integrating into current workflows)',
            'Lack of trust in the technology (e.g., skepticism about machine learning reliability)',
            'Cost of tools and systems (e.g., high upfront or maintenance costs)',
            'Dependence on data quality (e.g., concern that poor data may lead to poor predictions)',
            'Resistance to change (e.g., reluctance to move away from traditional methods)',
            'Limited understanding of machine learning (e.g., lack of technical expertise among stakeholders)',
            'Other:'
        ],
        'pa_adoption_encouragements': [
            'Demonstrated accuracy and reliability (e.g., proven success in similar projects)',
            'Ease of use and integration (e.g., tools that fit seamlessly into workflows)',
            'Clear ROI and cost-effectiveness (e.g., measurable benefits outweighing costs)',
            'Improved decision-making support (e.g., data-driven insights for critical decisions)',
            'Stakeholder buy-in and support (e.g., leadership or team endorsement)',
            'Comprehensive training and support (e.g., resources to ease adoption and use)',
            'Other:'
        ],
        'pa_adoption_discouragements': [
            'High initial investment costs (e.g., budget constraints)',
            'Lack of proven case studies (e.g., absence of successful real-world examples)',
            'Concerns about data governance (e.g., unclear policies or compliance risks)',
            'Fear of automation replacing human judgment (e.g., over-reliance on algorithms)',
            'Perceived complexity or technical barriers (e.g., challenging to understand or manage)',
            'Other:'
        ],
        'pa_confidence_factors': [
            'Accuracy of past predictions',
            'Complexity of the model used',
            'Clarity in results presentation',
            'Stakeholder feedback on accuracy',
            'Other:'
        ]
    }
    expanded_df = explode_multiple_choice(df, columns_valid_choices)

    # Save the transformed DataFrame
    expanded_df.to_csv('C:/Users/x/OneDrive - Berlin School of Business and Innovation (BSBI)/Desktop/BSBI/Dissertation/Data-Cleaning/transformed_MultipleChoice_survey_responses.csv', index=False)
    print("Data saved to transformed_MultipleChoice_survey_responses.csv file")

    # Display processed data
    pd.set_option('display.max_columns', None)

    # Save processed data to a new CSV file
    df.to_csv("C:/Users/x/OneDrive - Berlin School of Business and Innovation (BSBI)/Desktop/BSBI/Dissertation/Data-Cleaning/cleaned_survey_data.csv", index=False)
    print("Data saved to cleaned_survey_data.csv file")

if __name__ == "__main__":
    main()
