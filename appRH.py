import streamlit as st
import joblib
import pandas as pd

# Charger la pipeline sauvegardée
pipeline = joblib.load("project_full_pipeline.pkl")

st.title("Application de prédiction avec ta pipeline ML")

# Inputs pour les colonnes CONTINUES
age = st.number_input("Age", min_value=0, max_value=120, value=30)
total_working_years = st.number_input("TotalWorkingYears", min_value=0, max_value=50, value=10)
years_at_company = st.number_input("YearsAtCompany", min_value=0, max_value=50, value=5)
years_with_curr_manager = st.number_input("YearsWithCurrManager", min_value=0, max_value=50, value=3)
mean_presence_time = st.number_input("meanPresenceTime", min_value=0.0, max_value=24.0, value=8.0, format="%.2f")

# Inputs pour les colonnes DIST (catégorielles ou ordinales)
education_field = st.selectbox("EducationField", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
business_travel = st.selectbox("BusinessTravel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
environment_satisfaction = st.selectbox("EnvironmentSatisfaction", [1, 2, 3, 4])
marital_status = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])

# Inputs pour les colonnes FEAT_ENG (attention beaucoup de colonnes, je te donne un exemple pour les principales)
department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
distance_from_home = st.number_input("DistanceFromHome", min_value=0, max_value=100, value=10)
education = st.selectbox("Education", [1, 2, 3, 4, 5])
employee_count = st.number_input("EmployeeCount", min_value=0, max_value=1000, value=1)
employee_id = st.number_input("EmployeeID", min_value=1, max_value=10000, value=1001)
gender = st.selectbox("Gender", ["Male", "Female"])
job_level = st.number_input("JobLevel", min_value=1, max_value=5, value=2)
job_role = st.selectbox("JobRole", [
    "Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative",
    "Manager", "Sales Representative", "Research Director", "Human Resources"
])
monthly_income = st.number_input("MonthlyIncome", min_value=0, max_value=100000, value=5000)
num_companies_worked = st.number_input("NumCompaniesWorked", min_value=0, max_value=20, value=3)
over_18 = st.selectbox("Over18", ["Y", "N"])
percent_salary_hike = st.number_input("PercentSalaryHike", min_value=0, max_value=100, value=15)
standard_hours = st.number_input("StandardHours", min_value=0, max_value=100, value=40)
stock_option_level = st.number_input("StockOptionLevel", min_value=0, max_value=3, value=1)
training_times_last_year = st.number_input("TrainingTimesLastYear", min_value=0, max_value=20, value=3)
years_since_last_promotion = st.number_input("YearsSinceLastPromotion", min_value=0, max_value=50, value=1)
job_involvement = st.selectbox("JobInvolvement", [1, 2, 3, 4])
performance_rating = st.selectbox("PerformanceRating", [1, 2, 3, 4])
job_satisfaction = st.selectbox("JobSatisfaction", [1, 2, 3, 4])
work_life_balance = st.selectbox("WorkLifeBalance", [1, 2, 3, 4])
mean_start_time = st.number_input("meanStartTime", min_value=0.0, max_value=24.0, value=9.0, format="%.2f")
mean_leave_time = st.number_input("meanLeaveTime", min_value=0.0, max_value=24.0, value=18.0, format="%.2f")
number_of_absence_day = st.number_input("numberOfAbsenceDay", min_value=0, max_value=365, value=2)
number_of_presence_day = st.number_input("numberOfPresenceDay", min_value=0, max_value=365, value=250)

# Préparation du DataFrame d'entrée
input_data = pd.DataFrame({
    'Age': [age],
    'TotalWorkingYears': [total_working_years],
    'YearsAtCompany': [years_at_company],
    'YearsWithCurrManager': [years_with_curr_manager],
    'meanPresenceTime': [mean_presence_time],
    'EducationField': [education_field],
    'BusinessTravel': [business_travel],
    'EnvironmentSatisfaction': [environment_satisfaction],
    'MaritalStatus': [marital_status],
    'Age_feat_eng': [age],  # Si ta pipeline attend plusieurs fois 'Age' (sinon adapte)
    'BusinessTravel_feat_eng': [business_travel],  # pareil
    'Department': [department],
    'DistanceFromHome': [distance_from_home],
    'Education': [education],
    'EmployeeCount': [employee_count],
    'EmployeeID': [employee_id],
    'Gender': [gender],
    'JobLevel': [job_level],
    'JobRole': [job_role],
    'MaritalStatus_feat_eng': [marital_status],  # pareil
    'MonthlyIncome': [monthly_income],
    'NumCompaniesWorked': [num_companies_worked],
    'Over18': [over_18],
    'PercentSalaryHike': [percent_salary_hike],
    'StandardHours': [standard_hours],
    'StockOptionLevel': [stock_option_level],
    'TotalWorkingYears_feat_eng': [total_working_years],
    'TrainingTimesLastYear': [training_times_last_year],
    'YearsSinceLastPromotion': [years_since_last_promotion],
    'YearsWithCurrManager_feat_eng': [years_with_curr_manager],
    'JobInvolvement': [job_involvement],
    'PerformanceRating': [performance_rating],
    'EnvironmentSatisfaction_feat_eng': [environment_satisfaction],
    'JobSatisfaction': [job_satisfaction],
    'WorkLifeBalance': [work_life_balance],
    'meanStartTime': [mean_start_time],
    'meanLeaveTime': [mean_leave_time],
    'numberOfAbsenceDay': [number_of_absence_day],
    'numberOfPresenceDay': [number_of_presence_day]
})

# Ajuste les noms selon ta pipeline exacte, ici c'est un exemple

# Bouton de prédiction
if st.button("Prédire"):
    prediction = pipeline.predict(input_data)
    st.write(f"Résultat de la prédiction : {prediction[0]}")
