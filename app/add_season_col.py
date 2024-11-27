import pandas as pd

# Load data (assuming you have the file path)
df = pd.read_csv('restaurant.csv')

# Convert 'DateTime' column to datetime type, keeping the original intact
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')

# Function to determine the season based on the updated Lesotho seasonal calendar
def get_lesotho_season(date):
    month = date.month
    if month in [11, 12, 1]:
        return 'Summer'
    elif month in [2, 3, 4]:
        return 'Autumn'
    elif month in [5, 6, 7]:
        return 'Winter'
    else:
        return 'Spring'

# Add a new 'Season' column without changing the 'DateTime' column
df['Season'] = df['DateTime'].apply(get_lesotho_season)

# Save the updated dataframe back to CSV, without replacing the original columns
df.to_csv('dataset.csv', index=False)

# Preview the dataframe to ensure the new 'Season' column is added
print(df.head())
