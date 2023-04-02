#!/usr/bin/env python3
import pandas as pd
import numpy as np



def load_dataset(file_path):
    """
    Load dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """
    Perform data cleaning and feature conversion on the given dataset.
    """
    
    # Check for missing values
    if data.isnull().sum().sum() > 0:
        data.dropna(inplace=True)
    # Check for duplicates
    if data.duplicated().sum() > 0:
        data.drop_duplicates(inplace=True)
    # Check if data is already clean
    if data.isnull().sum().sum() == 0 and data.duplicated().sum() == 0 and \
        data['sex'].isin(['male', 'female']).all() and \
        data['chest pain type'].isin(['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic']).all() and \
        data['target'].isin(['Heart risk', 'Normal']).all() and \
        data['resting ecg'].isin(['Normal', 'Abnormality in ST-T wave', 'Left ventricular hypertrophy']).all():
        print("Data is already clean.")
    else:
        # Feature conversions
        data.loc[data['sex'] == 1, 'sex'] = 'male'
        data.loc[data['sex'] == 0, 'sex'] = 'female'

        data.loc[data['chest pain type'] == 1, 'chest pain type'] = 'typical angina'
        data.loc[data['chest pain type'] == 2, 'chest pain type'] = 'atypical angina'
        data.loc[data['chest pain type'] == 3, 'chest pain type'] = 'non-anginal pain'
        data.loc[data['chest pain type'] == 4, 'chest pain type'] = 'asymptomatic'
        
        data.loc[data['resting ecg'] == 0, 'resting ecg'] = 'Normal'
        data.loc[data['resting ecg'] == 1, 'resting ecg'] = 'Abnormality in ST-T wave'
        data.loc[data['resting ecg'] == 2, 'resting ecg'] = 'Left ventricular hypertrophy'
        
        data.loc[data['target'] == 1, 'target'] = 'Heart risk'
        data.loc[data['target'] == 0, 'target'] = 'Normal'
        
        print("Data cleaning complete.")

    # Assign cleaned data to a new variable
    cleaned_data = data
    
    return cleaned_data

def analyze_age_distribution(cleaned_data):
    """
    Analyze the age distribution of patients in the dataset.
    """
    age_counts = cleaned_data['age'].value_counts()
    print("Age distribution for all patients:")
    print('Age:\tNumber of patients:')
    print(age_counts)
    
    age_counts_heart_risk = cleaned_data[cleaned_data['target'] == 1]['age'].value_counts()
    print("\nAge distribution for patients with heart risk:")
    print('Age:\tNumber of patients:')
    print(age_counts_heart_risk)
    
    max_age = cleaned_data['age'].max()
    min_age = cleaned_data['age'].min()
    avg_age = cleaned_data['age'].mean()
    std_age = cleaned_data['age'].std()
    print("\nAge statistics for all patients:")
    print(f"Max age: {max_age}")
    print(f"Min age: {min_age}")
    print(f"Average age: {avg_age:.2f}")
    print(f"Standard deviation of age: {std_age:.2f}")
    
    max_age_heart_risk = cleaned_data[cleaned_data['target'] == 1]['age'].max()
    min_age_heart_risk = cleaned_data[cleaned_data['target'] == 1]['age'].min()
    avg_age_heart_risk = cleaned_data[cleaned_data['target'] == 1]['age'].mean()
    std_age_heart_risk = cleaned_data[cleaned_data['target'] == 1]['age'].std()
    print("\nAge statistics for patients with heart risk:")
    print(f"Max age: {max_age_heart_risk}")
    print(f"Min age: {min_age_heart_risk}")
    print(f"Average age: {avg_age_heart_risk:.2f}")
    print(f"Standard deviation of age: {std_age_heart_risk:.2f}")


def analyze_gender_distribution(cleaned_data):
    """
    Analyze the gender distribution of patients in the cleaned dataset.
    """
    gender_counts = cleaned_data['sex'].value_counts()
    gender_percentage = gender_counts / gender_counts.sum() * 100
    print(f"\nGender:{gender_counts.to_frame('Number of Patients:')}\n")
    print(f"{gender_percentage.to_frame('Percentage (%)')}\n")
    
    male_data = cleaned_data.loc[cleaned_data['sex'] == 'male']
    female_data = cleaned_data.loc[cleaned_data['sex'] == 'female']
    
    male_risk_count = male_data['target'].sum()
    female_risk_count = female_data['target'].sum()
    
    male_risk_percentage = male_risk_count / male_data.shape[0] * 100
    female_risk_percentage = female_risk_count / female_data.shape[0] * 100
    
    print("Risk of getting heart disease by gender:")
    print(f"Male: {male_risk_count} out of {male_data.shape[0]} ({male_risk_percentage:.2f}%)")
    print(f"Female: {female_risk_count} out of {female_data.shape[0]} ({female_risk_percentage:.2f}%)")

def analyze_chest_pain(cleaned_data):
    """
    Analyze the chest pain distribution of patients in the dataset.
    """
    
    chest_pain_counts = cleaned_data['chest pain type'].value_counts()
    print("Chest pain distribution:")
    print('\nChest pain type:\tCount:')
    print(chest_pain_counts)

    # Calculate the percentage of patients with each chest pain type
    total_patients = len(cleaned_data)
    print('\nChest pain type:\tPercentage Count:')
    for chest_pain, count in chest_pain_counts.items():
        percentage = (count / total_patients) * 100
        print(f"{chest_pain}: \t{percentage:.2f}%")
    
    print('\n')
    # Calculate the percentage of patients with heart risk for each chest pain type
    for chest_pain in cleaned_data['chest pain type'].unique():
        risk_count = len(cleaned_data[(cleaned_data['chest pain type'] == chest_pain) & (cleaned_data['target'] == 1)])
        total_count = len(cleaned_data[cleaned_data['chest pain type'] == chest_pain])
        risk_percentage = (risk_count / total_count) * 100
        print(f"Percentage of patients with {chest_pain} chest pain type at risk of heart disease: {risk_percentage:.2f}%")

def analyze_blood_pressure(cleaned_data):
    """
    Analyze the levels of blood pressure in the dataset and its correlation with heart disease.
    """
    # Group the data by target variable and calculate the mean, median, min, and max blood pressure values
    bp_stats = cleaned_data.groupby('target')['resting bp s'].agg(['mean', 'median', 'min', 'max'])
    print("\nBlood Pressure Statistics:")
    print(bp_stats)

    # Calculate the correlation coefficient between blood pressure and heart disease
    correlation = cleaned_data['resting bp s'].corr(cleaned_data['target'])
    print("\nCorrelation between Blood Pressure and Heart Disease:")
    print(correlation)

def analyze_blood_pressure_and_cholesterol(cleaned_data):
    """
    Analyze the levels of blood pressure and cholesterol in the cleaned dataset.
    """
    # Get the mean values of blood pressure and cholesterol for patients with and without heart disease
    heart_disease_bp_mean = cleaned_data[cleaned_data['target'] == 1]['resting bp s'].mean()
    heart_disease_chol_mean = cleaned_data[cleaned_data['target'] == 1]['cholesterol'].mean()
    no_heart_disease_bp_mean = cleaned_data[cleaned_data['target'] == 0]['resting bp s'].mean()
    no_heart_disease_chol_mean = cleaned_data[cleaned_data['target'] == 0]['cholesterol'].mean()

    # Print the mean values
    print('\n')
    print("Mean blood pressure for patients with heart disease:", heart_disease_bp_mean)
    print("Mean blood pressure for patients without heart disease:", no_heart_disease_bp_mean)
    print("Mean cholesterol for patients with heart disease:", heart_disease_chol_mean)
    print("Mean cholesterol for patients without heart disease:", no_heart_disease_chol_mean)

    # Get the standard deviation of blood pressure and cholesterol for patients with and without heart disease
    heart_disease_bp_std = cleaned_data[cleaned_data['target'] == 1]['resting bp s'].std()
    heart_disease_chol_std = cleaned_data[cleaned_data['target'] == 1]['cholesterol'].std()
    no_heart_disease_bp_std = cleaned_data[cleaned_data['target'] == 0]['resting bp s'].std()
    no_heart_disease_chol_std = cleaned_data[cleaned_data['target'] == 0]['cholesterol'].std()

    print('\n')
    # Print the standard deviation
    print("Standard deviation of blood pressure for patients with heart disease:", heart_disease_bp_std)
    print("Standard deviation of blood pressure for patients without heart disease:", no_heart_disease_bp_std)
    print("Standard deviation of cholesterol for patients with heart disease:", heart_disease_chol_std)
    print("Standard deviation of cholesterol for patients without heart disease:", no_heart_disease_chol_std)
    
    print('\n')
    # Analyze the correlation between blood pressure, cholesterol, and heart disease
    correlation_matrix = cleaned_data[['resting bp s', 'cholesterol', 'target']].corr()
    print("Correlation between resting blood pressure, cholesterol, and heart disease:")
    print(correlation_matrix)

def analyze_fbs_distribution(cleaned_data):
    """
    Analyze the distribution of fasting blood sugar levels in relation to heart disease.
    """
    fbs_counts = cleaned_data['fasting blood sugar'].value_counts()
    fbs_percentages = round(fbs_counts / len(cleaned_data) * 100, 2)

    fbs_heart_disease_counts = cleaned_data.groupby('target')['fasting blood sugar'].value_counts()
    fbs_heart_disease_percentages = round(fbs_heart_disease_counts / len(cleaned_data) * 100, 2)

    print("Fasting Blood Sugar distribution:")
    print("------------------------------")
    print(f"Total count:\n{fbs_counts}\n")
    print(f"Percentage of total dataset:\n{fbs_percentages}\n")
    print("------------------------------")
    print(f"Counts by heart disease status:\n{fbs_heart_disease_counts}\n")
    print(f"Percentages by heart disease status:\n{fbs_heart_disease_percentages}")

def analyze_ecg_results(cleaned_data):
    """
    Analyze the Resting electrocardiogram results of patients in the cleaned_dataset.
    """
    ecg = cleaned_data['resting ecg']
    ecg_counts = ecg.value_counts()
    ecg_labels = {'Normal': 0, 'Abnormality in ST-T wave': 1, 'Left ventricular hypertrophy': 2}
    # ecg_counts.rename(index=ecg_labels, inplace=True)
    
    print("\nResting electrocardiogram distribution:")
    print(ecg_counts)
    
    ecg_by_target = cleaned_data.groupby(['resting ecg', 'target']).size().reset_index(name='count')
    ecg_by_target['Resting ecg'] = ecg_by_target['resting ecg'].map(ecg_labels)
    ecg_by_target['resting ecg'] = ecg_by_target['resting ecg'].map(ecg)
    ecg_by_target['target'] = ecg_by_target['target']
    
    print("\nResting electrocardiogram by target:")
    print(ecg_by_target)


# Main program
if __name__ == '__main__':
    # Load the dataset
    data = load_dataset('heart_statlog_cleveland_hungary_final.csv')
    print("Dataset loaded: heart_statlog_cleveland_hungary_final.csv\n")
    print("First 5 rows of the data:")
    print(data.head())

    while True:
        print("\nWhat would you like to do?")
        print("\t1. Clean the data")
        print("\t2. Explore the data")
        print("\t3. Exit")

        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            print("\nPerforming data cleaning...")
            cleaned_data = clean_data(data)
            print("First 5 rows of the cleaned data:")
            print(cleaned_data.head())
        elif choice == '2':
            print("\nExploring the data...")
            cleaned_data = clean_data(data)
            while True:
                print("\nWhat would you like to Explore/Analyze?")
                print("\t1. Age distribution")
                print("\t2. Gender distribution")
                print("\t3. Chest Pain Types")
                print("\t4. Blood Pressure Analysis")
                print("\t5. Blood pressure and Cholesterol")
                print("\t6. Fasting blood sugar analysis")
                print("\t7. Resting electrocardiogram results")
                
                print("\t8. Back to main")
                
                explore_choice = input("Enter your choice: ")
                
                if explore_choice == '1':
                    print("\n\t\tAnalyzing age distribution...")
                    analyze_age_distribution(cleaned_data)
                elif explore_choice == '2':
                    print("\n\t\tAnalyzing gender distribution...")
                    analyze_gender_distribution(cleaned_data)
                elif explore_choice == '3':
                    print('\n\t\tAnalyzing chest pain types...')
                    analyze_chest_pain(cleaned_data)
                elif explore_choice == '4':
                    print('Analyzing blood pressure...')
                    analyze_blood_pressure(cleaned_data)
                elif explore_choice == '5':
                    print('Analyzing blood pressure and cholesterol...')
                    analyze_blood_pressure_and_cholesterol(cleaned_data)
                elif explore_choice == '6':
                    print('Analyzing Fasting blood sugar...')
                    analyze_fbs_distribution(cleaned_data)
                elif explore_choice == '7':
                    print('Analyzing ecg results...')
                    analyze_ecg_results(cleaned_data)
                elif explore_choice == '8':
                    print("Returning to main menu...")
                    break
                else:
                    print("Invalid choice.")
        elif choice == '3':
            print("Exiting the program...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
