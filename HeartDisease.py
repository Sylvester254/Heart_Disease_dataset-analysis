#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
    if (data.isnull().sum().sum() == 0
        and data.duplicated().sum() == 0
        and data['sex'].isin(['male', 'female']).all()
        and data['chest pain type'].isin(['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic']).all()
        and data['resting ecg'].isin(['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy']).all()
        and data['exercise angina'].isin(['no', 'yes']).all()
        and data['target'].isin(['Normal', 'Heart risk']).all()
        and data["ST slope"].isin(["Normal", "Upsloping", "Flat", "Downsloping"]).all()):
        print("Data is already clean.")
    else:
        # Feature conversions
        data.loc[data['sex'] == 1, 'sex'] = 'male'
        data.loc[data['sex'] == 0, 'sex'] = 'female'

        data.loc[data['chest pain type'] == 1, 'chest pain type'] = 'typical angina'
        data.loc[data['chest pain type'] == 2, 'chest pain type'] = 'atypical angina'
        data.loc[data['chest pain type'] == 3, 'chest pain type'] = 'non-anginal pain'
        data.loc[data['chest pain type'] == 4, 'chest pain type'] = 'asymptomatic'
        
        data["resting ecg"] = data.loc[:, "resting ecg"].replace({0: "Normal",
                                                                1: "Abnormality in ST-T wave",
                                                                2: "Left ventricular hypertrophy"})
        
        data["exercise angina"] = data.loc[:, "exercise angina"].replace({0: "No",
                                                                        1: "Yes"})
        
        data["ST slope"] = data.loc[:, "ST slope"].replace({0: "Normal",
                                                        1: "Upsloping",
                                                        2: "Flat",
                                                        3: "Downsloping"})
        
        # Convert target variable
        data["target"] = data.loc[:, "target"].replace({0: "Normal",
                                                        1: "Heart risk"})
            
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

def analyze_max_heart_rate(cleaned_data):
    """
    Analyze the maximum heart rate achieved by patients in the cleaned_dataset.
    """
    # Summary statistics
    print("Maximum heart rate summary statistics:")
    print(cleaned_data['max heart rate'].describe())
    
    # Distribution of maximum heart rate
    print("\nMaximum heart rate distribution:")
    print(cleaned_data['max heart rate'].value_counts().sort_index())
    
    # Maximum heart rate by target
    max_hr_by_target = cleaned_data.groupby(['max heart rate', 'target']).size().reset_index(name='count')
    max_hr_by_target['target'] = max_hr_by_target['target']
    
    print("\nMaximum heart rate by target:")
    print(max_hr_by_target)

def analyze_exercise_angina(cleaned_data):
    """
    Analyze the incidence of exercise-induced angina in patients.
    """
    angina_counts = cleaned_data['exercise angina'].value_counts()
    
    print("Exercise-induced angina distribution:")
    print(angina_counts)
    
    angina_by_target = cleaned_data.groupby(['exercise angina', 'target']).size().reset_index(name='count')
    angina_by_target['exercise angina'] = angina_by_target['exercise angina'].str.capitalize()
    angina_by_target['target'] = angina_by_target['target'].str.capitalize()
    
    print("\nExercise-induced angina by target:")
    print(angina_by_target)

def analyze_oldpeak(cleaned_data):
    """
    Analyze the Exercise induced ST-depression in comparison with the state of rest of patients in the cleaned_dataset.
    """
    oldpeak_by_target = cleaned_data.groupby('target')['oldpeak'].mean()
    
    print("\nAverage exercise induced ST-depression by target:")
    print(oldpeak_by_target)
    
    oldpeak_counts = cleaned_data['oldpeak'].value_counts()
    oldpeak_counts.sort_index(inplace=True)
    
    print("\nExercise induced ST-depression distribution:")
    print(oldpeak_counts)
    
    oldpeak_by_ecg = cleaned_data.groupby(['resting ecg', 'exercise angina'])['oldpeak'].mean()
    
    print("\nAverage exercise induced ST-depression by resting ECG and exercise angina:")
    print(oldpeak_by_ecg)

def analyze_st_slope(cleaned_data):
    """
    Analyze the ST slope results of patients in the cleaned_dataset.
    """
    st_slope_counts = cleaned_data['ST slope'].value_counts()
    
    print("ST slope distribution:")
    print(st_slope_counts)
    
    st_slope_by_target = cleaned_data.groupby(['ST slope', 'target']).size().reset_index(name='count')
    st_slope_by_target['target'] = st_slope_by_target['target']
    
    print("\nST slope by target:")
    print(st_slope_by_target)

def correlation_matrix(data):
    """ 
    Create a correlation matrix to show the correlation coefficients
    between all pairs of variables in the dataset.
    """
    corr = data.corr(numeric_only=True)

    # Plot the correlation matrix
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

def feature_importance(data, k):
    """ 
    Use SelectKBest feature selection technique to identify the k most important
    features that are highly correlated with the target variable.
    """
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Min-Max Scaling
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=data.columns[:-1])


    if k > len(X.columns):
        print(f"Error: The maximum value for k is {len(X.columns)}")
        return
    # Use SelectKBest to select the k best features
    best_features = SelectKBest(score_func=chi2, k=k).fit(X, y)

    # Get the scores and p-values for each feature
    scores = best_features.scores_
    pvalues = best_features.pvalues_

    # Create a dataframe of feature scores and p-values
    feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores, 'P-value': pvalues})

    # Sort the dataframe by score in descending order
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)

    # Print the top k features with their scores and p-values
    print(f'Top {k} features:')
    print(feature_scores.head(k))

def clustering_analysis(data, n_clusters):
    """
    Perform clustering analysis on the given data to group similar patients based on their features.
    """
    # Separate features and target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Scale the data using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Add the cluster labels to the original dataset
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clusters

    # Analyze the target variable for each cluster separately
    for i in range(n_clusters):
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == i]
        cluster_data.loc[cluster_data['target'] == 1, 'target'] = 'Heart risk'
        cluster_data.loc[cluster_data['target'] == 0, 'target'] = 'Normal'
        print(f"Cluster {i} - Number of patients: {len(cluster_data)}")
        print(cluster_data['target'].value_counts())
        print()          

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
        print("\t3. Advanced Analysis")
        print("\tQ. Exit")

        choice = input("Enter your choice (1, 2, 3 or Q): ")

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
                print("\t5. Blood Pressure and Cholesterol")
                print("\t6. Fasting Blood Sugar Analysis")
                print("\t7. Resting Electrocardiogram Results")
                print("\t8. Heart Rate Analysis")
                print("\t9. Exercise Angina Analysis")
                print("\t10. Oldpeak Analysis")
                print("\t11. Analyze ST slope")
                print("\t12. Back to Main")
                
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
                    print('Analyzing max heart rate...')
                    analyze_max_heart_rate(cleaned_data)
                elif explore_choice == '9':
                    print('Analyzing exercise angina...')
                    analyze_exercise_angina(cleaned_data)
                elif explore_choice == '10':
                    print('Analyzing Oldpeak...')
                    analyze_oldpeak(cleaned_data)
                elif explore_choice == '11':
                    print("Analyzing the ST slope...")
                    analyze_st_slope(cleaned_data)
                elif explore_choice == '12':
                    print("Returning to main menu...")
                    break
                else:
                    print("Invalid choice.")
        elif choice == '3':
            print("\nAdvanced Analysis on the data...")
            while True:
                print("\nWhat would you like to Analyze?")
                print("\t1. Correlation Matrix")
                print("\t2. Most Important Features")
                print("\t3. Clustering")
                print("\t4. Back to Main")
                
                advanced_choice = input("Enter your choice: ")
                
                if advanced_choice == '1':
                    print("\n\t\tAnalyzing correlation matrix...")
                    correlation_matrix(data)
                elif advanced_choice == '2':
                    k = int(input("Enter the number of features to select: "))
                    print(f"\n\t\tAnalyzing the {k} most important features...")
                    feature_importance(data, k)
                elif advanced_choice == '3':
                    print("\n\t\tPerforming clustering analysis...")
                    n_clusters = int(input("Enter the number of clusters to create: "))
                    clustering_analysis(data, n_clusters)
                elif advanced_choice == '4':
                    print("Returning to main menu...")
                    break
                else:
                    print("Invalid choice.")
                
        elif choice == 'Q' or 'q':
            print("Exiting the program...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3 or Q.")
