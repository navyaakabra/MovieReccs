import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Sci-kit learn imports for modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Load datasets 
movies_df = pd.read_csv('/Users/Navyaa/Desktop/kaggle/Movies.csv')
film_details_df = pd.read_csv('/Users/Navyaa/Desktop/kaggle/FilmDetails.csv')
more_info_df = pd.read_csv('/Users/Navyaa/Desktop/kaggle/MoreInfo.csv')
#poster_path_df = pd.read_csv('/Users/Navyaa/Desktop/kaggle/PosterPath.csv')

def check_missing_values(dataframes):
    """
    Check for missing values across multiple DataFrames
    
    Parameters:
    dataframes (dict): Dictionary of DataFrames to check
    
    Returns:
    Dictionary of missing value information
    """
    missing_values = {}
    
    print("\n--- Missing Values Analysis ---")
    for name, df in dataframes.items():
        # Calculate missing values
        missing = df.isnull().sum()
        missing_percent = 100 * df.isnull().sum() / len(df)
        
        # Create missing values DataFrame
        missing_table = pd.concat([missing, missing_percent], axis=1, keys=['Missing Count', 'Missing Percent'])
        
        # Filter to show only columns with missing values
        missing_table = missing_table[missing_table['Missing Count'] > 0]
        
        if not missing_table.empty:
            print(f"\n{name} - Missing Values:")
            print(missing_table)
            missing_values[name] = missing_table
        else:
            print(f"\n{name} - No missing values")
    
    return missing_values

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    DataFrame with handled missing values
    """
    # Create a copy of the DataFrame
    cleaned_df = df.copy()
    
    # Strategy for different column types
    # Numeric columns: fill with median
    numeric_columns = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    
    # Categorical columns: fill with mode
    categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    # Date columns: fill with most frequent date
    date_columns = cleaned_df.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    return cleaned_df

def check_duplicate_data(dataframes):
    """
    Check for duplicate data across DataFrames
    
    Parameters:
    dataframes (dict): Dictionary of DataFrames to check
    
    Returns:
    Dictionary of duplicate data information
    """
    duplicate_info = {}
    
    print("\n--- Duplicate Data Analysis ---")
    for name, df in dataframes.items():
        # Check total duplicates
        total_duplicates = df.duplicated().sum()
        
        # Check duplicates for specific columns
        # You might want to adjust these based on your specific dataset
        column_duplicates = {}
        key_columns = ['title', 'release_date']  # Modify as needed
        
        for col in key_columns:
            if col in df.columns:
                dup_in_col = df.duplicated(subset=[col], keep=False).sum()
                column_duplicates[col] = dup_in_col
        
        print(f"\n{name}:")
        print(f"Total duplicate rows: {total_duplicates}")
        
        if column_duplicates:
            print("Duplicates by column:")
            for col, count in column_duplicates.items():
                print(f"  - Duplicates in {col}: {count}")
        
        duplicate_info[name] = {
            'total_duplicates': total_duplicates,
            'column_duplicates': column_duplicates
        }
    
    return duplicate_info

def remove_duplicates(df):
    """
    Remove duplicate data from DataFrame
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    DataFrame with duplicates removed
    """
    # Create a copy of the DataFrame
    cleaned_df = df.copy()
    
    # Print initial number of rows
    print(f"\nInitial number of rows: {len(cleaned_df)}")
    
    # Remove total duplicates
    cleaned_df.drop_duplicates(inplace=True)
    
    # Remove duplicates based on specific columns (if applicable)
    # Modify columns as needed for your dataset
    key_columns = ['title', 'release_date']
    key_columns = [col for col in key_columns if col in cleaned_df.columns]
    
    if key_columns:
        cleaned_df.drop_duplicates(subset=key_columns, keep='first', inplace=True)
    
    # Print final number of rows
    print(f"Number of rows after removing duplicates: {len(cleaned_df)}")
    
    return cleaned_df

# Combine DataFrames for analysis
dataframes = {
    'Movies': movies_df,
    'Film Details': film_details_df,
    'More Info': more_info_df,
    'Poster Path': poster_path_df
}

# Perform data cleaning steps
# 1. Check missing values
missing_values = check_missing_values(dataframes)

# 2. Handle missing values
movies_df_cleaned = handle_missing_values(movies_df)

# 3. Check and remove duplicates
duplicate_info = check_duplicate_data(dataframes)
movies_df_final = remove_duplicates(movies_df_cleaned)

# Convert 'release_date' to datetime format
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')

def calculate_descriptive_statistics(df):
    """
    Calculate descriptive statistics for numeric columns in the movie dataset.
    
    Parameters:
    df (pandas.DataFrame): Input movie dataset
    
    Returns:
    Prints and returns a dictionary of descriptive statistics
    """
    # Select numeric columns
    numeric_columns = ['user_score', 'critic_score', 'box_office']
    
    # Filter for actually existing numeric columns
    existing_numeric_columns = [col for col in numeric_columns if col in df.columns]
    
    # Create a dictionary to store results
    stats_results = {}
    
    print("\n--- Descriptive Statistics ---")
    
    for column in existing_numeric_columns:
        # Calculate statistics
        mean_val = df[column].mean()
        median_val = df[column].median()
        mode_val = df[column].mode().values
        
        # Store results
        stats_results[column] = {
            'mean': mean_val,
            'median': median_val,
            'mode': mode_val,
            'std_dev': df[column].std(),
            'min': df[column].min(),
            'max': df[column].max()
        }
        
        # Print results
        print(f"\n{column.replace('_', ' ').title()} Statistics:")
        print(f"Mean: {mean_val:.2f}")
        print(f"Median: {median_val:.2f}")
        print(f"Mode: {mode_val}")
        print(f"Standard Deviation: {stats_results[column]['std_dev']:.2f}")
        print(f"Min: {stats_results[column]['min']:.2f}")
        print(f"Max: {stats_results[column]['max']:.2f}")
    
    # Additional categorical column statistics
    if 'language' in df.columns:
        print("\n--- Language Distribution ---")
        language_counts = df['language'].value_counts()
        print(language_counts)
    
    # Genre distribution
    if 'genres' in df.columns:
        print("\n--- Genre Distribution ---")
        genre_counts = df['genres'].str.get_dummies(sep=', ').sum().sort_values(ascending=False)
        print(genre_counts.head(10))
    
    return stats_results

descriptive_stats = calculate_descriptive_statistics(movies_df)

def detect_outliers_iqr(df, columns=None):
    """
    Detect outliers using Interquartile Range (IQR) method
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    columns (list): Numeric columns to check for outliers (optional)
    
    Returns:
    Dictionary of outlier information
    """
    # If no columns specified, select all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Filter for actually existing columns
    existing_columns = [col for col in columns if col in df.columns]
    
    outliers_info = {}
    
    print("\n--- Outlier Detection Using IQR Method ---")
    
    # Create figure for box plots
    fig, axes = plt.subplots(len(existing_columns), 1, figsize=(10, 4*len(existing_columns)))
    fig.suptitle('Box Plots for Numeric Columns', fontsize=16)
    
    # Adjust for single column case
    if len(existing_columns) == 1:
        axes = [axes]
    
    for i, column in enumerate(existing_columns):
        # Calculate Q1, Q3, and IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Detect outliers
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        # Store outlier information
        outliers_info[column] = {
            'total_outliers': len(outliers),
            'percent_outliers': (len(outliers) / len(df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        # Print outlier details
        print(f"\n{column}:")
        print(f"  Total Outliers: {len(outliers)}")
        print(f"  Percent of Outliers: {outliers_info[column]['percent_outliers']:.2f}%")
        print(f"  Lower Bound: {lower_bound:.2f}")
        print(f"  Upper Bound: {upper_bound:.2f}")
        
        # Create box plot
        sns.boxplot(x=df[column], ax=axes[i])
        axes[i].set_title(f'Box Plot for {column}')
    
    plt.tight_layout()
    plt.show()
    
    return outliers_info

def handle_outliers(df, columns=None, method='clip'):
    """
    Handle outliers using different methods
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    columns (list): Numeric columns to handle outliers
    method (str): Method to handle outliers ('clip', 'remove', 'median')
    
    Returns:
    DataFrame with handled outliers
    """
    # Create a copy of the DataFrame
    cleaned_df = df.copy()
    
    # If no columns specified, select all numeric columns
    if columns is None:
        columns = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Filter for actually existing columns
    existing_columns = [col for col in columns if col in cleaned_df.columns]
    
    print(f"\n--- Handling Outliers Using {method.upper()} Method ---")
    print(f"Original DataFrame shape: {cleaned_df.shape}")
    
    for column in existing_columns:
        # Calculate Q1, Q3, and IQR
        Q1 = cleaned_df[column].quantile(0.25)
        Q3 = cleaned_df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if method == 'clip':
            # Clip values to the bounds
            cleaned_df[column] = cleaned_df[column].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'remove':
            # Remove outliers
            cleaned_df = cleaned_df[
                (cleaned_df[column] >= lower_bound) & 
                (cleaned_df[column] <= upper_bound)
            ]
        elif method == 'median':
            # Replace outliers with median
            median_val = cleaned_df[column].median()
            cleaned_df.loc[
                (cleaned_df[column] < lower_bound) | 
                (cleaned_df[column] > upper_bound), 
                column
            ] = median_val
    
    print(f"Cleaned DataFrame shape: {cleaned_df.shape}")
    
    return cleaned_df

def visualize_outlier_impact(original_df, cleaned_df, columns=None):
    """
    Visualize the impact of outlier handling
    
    Parameters:
    original_df (pandas.DataFrame): Original DataFrame
    cleaned_df (pandas.DataFrame): Cleaned DataFrame
    columns (list): Columns to visualize
    """
    # If no columns specified, select all numeric columns
    if columns is None:
        columns = original_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Filter for actually existing columns
    existing_columns = [col for col in columns if col in original_df.columns]
    
    # Reduced figure size - 2 columns per 1 unit of height
    fig, axes = plt.subplots(len(existing_columns), 2, figsize=(12, 2*len(existing_columns)))
    fig.suptitle('Distribution Before and After Outlier Handling', fontsize=12)
    
    # Adjust for single column case
    if len(existing_columns) == 1:
        axes = [axes]
    
    for i, column in enumerate(existing_columns):
        # Original distribution
        sns.histplot(original_df[column], kde=True, ax=axes[i][0], color='blue', alpha=0.7)
        axes[i][0].set_title(f'{column} - Original Distribution')
        
        # Cleaned distribution
        sns.histplot(cleaned_df[column], kde=True, ax=axes[i][1], color='green', alpha=0.7)
        axes[i][1].set_title(f'{column} - Cleaned Distribution')
    
    plt.tight_layout()
    plt.show()

# Print available columns to verify
print("Available columns:", list(movies_df.columns))

# Dynamically select numeric columns
numeric_columns = list(movies_df.select_dtypes(include=['float64', 'int64']).columns)
print("Numeric columns:", numeric_columns)

# Detect outliers
outliers_info = detect_outliers_iqr(movies_df, columns=numeric_columns)

# Handle outliers using different methods
# 1. Clipping method
movies_df_clipped = handle_outliers(movies_df, columns=numeric_columns, method='clip')

# 2. Removal method
movies_df_removed = handle_outliers(movies_df, columns=numeric_columns, method='remove')

# 3. Median replacement method
movies_df_median = handle_outliers(movies_df, columns=numeric_columns, method='median')

# Visualize the impact of outlier handling
visualize_outlier_impact(movies_df, movies_df_clipped, columns=numeric_columns)


# Data Visualization
# User Score Distribution Histogram
plt.figure(figsize=(10, 6))
sns.histplot(movies_df['user_score'], bins=20, kde=True)
plt.title('Distribution of User Scores')
plt.xlabel('User Score')
plt.ylabel('Frequency')
plt.show()

# Top 10 Movies Bar Plot
genre_counts = movies_df['genres'].str.get_dummies(sep=', ').sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
genre_counts.head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Movie Genres')
plt.xlabel('Genres')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.show()

# Language Distribution Pie Chart
language_counts = movies_df['language'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(language_counts, labels=language_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Movies by Language')
plt.axis('equal')
plt.show()

def prepare_correlation_data(df):
    """
    Prepare data for correlation analysis by encoding categorical variables
    
    Parameters:
    df (pandas.DataFrame): Input movie dataset
    
    Returns:
    Prepared DataFrame for correlation analysis
    """
    # Create a copy of the dataframe
    data = df.copy()
    
    # Extract year from release date
    data['release_year'] = data['release_date'].dt.year
    
    # Encode categorical variables
    # Language encoding
    le_language = LabelEncoder()
    data['language_encoded'] = le_language.fit_transform(data['language'])
    
    # Genre encoding (multi-label)
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(data['genres'].str.split(', '))
    genre_columns = [f'genre_{genre}' for genre in mlb.classes_]
    genre_df = pd.DataFrame(genre_encoded, columns=genre_columns)
    
    # Combine features
    correlation_data = pd.concat([
        data[['user_score', 'language_encoded', 'release_year']], 
        genre_df
    ], axis=1)
    
    return correlation_data

def create_correlation_heatmap(correlation_data):
    """
    Create and visualize correlation matrix heatmap
    
    Parameters:
    correlation_data (pandas.DataFrame): Prepared data for correlation analysis
    """
    # Compute correlation matrix
    correlation_matrix = correlation_data.corr()
    
    # Create a large figure for better readability
    plt.figure(figsize=(16, 12))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, 
                annot=True,  # Show numeric correlation values
                cmap='coolwarm',  # Color map (blue for negative, red for positive)
                center=0,  # Center color scale at 0
                vmin=-1, 
                vmax=1,
                square=True,  # Make plot square
                linewidths=0.5,  # Add lines between cells
                cbar_kws={"shrink": .8})  # Slightly shrink the colorbar
    
    plt.title('Correlation Heatmap of Movie Features', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print top correlations
    print("\nTop Positive Correlations:")
    top_positive = correlation_matrix.unstack().sort_values(ascending=False)
    top_positive = top_positive[top_positive < 1]  # Exclude self-correlations
    print(top_positive.head())
    
    print("\nTop Negative Correlations:")
    top_negative = correlation_matrix.unstack().sort_values()
    top_negative = top_negative[top_negative > -1]  # Exclude self-correlations
    print(top_negative.head())

# Prepare data and create correlation heatmap
correlation_data = prepare_correlation_data(movies_df)
create_correlation_heatmap(correlation_data)


# Prepare data for predictive modeling
def prepare_data_for_prediction(df):
    """
    Prepare the dataset for predictive modeling
    
    Parameters:
    df (pandas.DataFrame): Input movie dataset
    
    Returns:
    Prepared features and target variable
    """
    # Create a copy of the dataframe
    data = df.copy()
    
    # Encode categorical variables
    # Language encoding
    le_language = LabelEncoder()
    data['language_encoded'] = le_language.fit_transform(data['language'])
    
    # Genre encoding (multi-label)
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(data['genres'].str.split(', '))
    genre_columns = [f'genre_{genre}' for genre in mlb.classes_]
    genre_df = pd.DataFrame(genre_encoded, columns=genre_columns)
    
    # Extract year from release date
    data['release_year'] = data['release_date'].dt.year
    
    # Combine features
    features = pd.concat([
        data[['language_encoded', 'release_year']], 
        genre_df
    ], axis=1)
    
    # Target variable
    target = data['user_score']
    
    return features, target

# Predictive Model: Linear Regression
def linear_regression_model(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Linear Regression model for predicting user scores
    
    Parameters:
    X_train, X_test: Feature matrices
    y_train, y_test: Target variables
    
    Returns:
    Trained Linear Regression model
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = lr_model.predict(X_test_scaled)
    
    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nLinear Regression Results:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Feature importance visualization
    feature_names = X_train.columns
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lr_model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['Feature'], feature_importance['Coefficient'])
    plt.title('Linear Regression - Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    return lr_model

# Predictive Model: Decision Tree Regression
def decision_tree_model(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Decision Tree Regression model for predicting user scores
    
    Parameters:
    X_train, X_test: Feature matrices
    y_train, y_test: Target variables
    
    Returns:
    Trained Decision Tree model
    """
    # Train Decision Tree model
    dt_model = DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_split=10)
    #dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = dt_model.predict(X_test)
    
    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Decision Tree Regression Results:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Feature importance visualization
    feature_names = X_train.columns
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': dt_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['Feature'], feature_importance['Importance'])
    plt.title('Decision Tree - Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    return dt_model
    

# Main prediction workflow
def run_movie_predictions():
    """
    Run predictive modeling on movie dataset
    """
    # Prepare features and target
    X, y = prepare_data_for_prediction(movies_df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Run Linear Regression
    linear_regression_model(X_train, X_test, y_train, y_test)
    
    # Run Decision Tree Regression
    decision_tree_model(X_train, X_test, y_train, y_test)

# Run the prediction models
run_movie_predictions()

# Recommendation function
def recommend_movies(genre=None, min_score=0):
    filtered_movies = movies_df[(movies_df['user_score'] >= min_score)]
    if genre:
        filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(genre)]
    return filtered_movies[['title', 'user_score', 'release_date']].head(10)

# Example usage of recommendation
print("\nRecommended Horror Movies:")
print(recommend_movies(genre='Horror', min_score=8))