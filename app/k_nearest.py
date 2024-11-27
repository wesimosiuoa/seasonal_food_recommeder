import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

# Load the data
data = pd.read_csv('dataset.csv')

# Dictionary to map input strings to corresponding codes
daypart_map = {'Night': 3, 'Morning': 2, 'Afternoon': 1, 'Evening': 0}
daytype_map = {'Weekend': 0, 'Weekday': 1}
season_map = {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}

# Preprocess the data
data['Daypart'] = data['Daypart'].map(daypart_map)
data['DayType'] = data['DayType'].map(daytype_map)
data['Season'] = data['Season'].map(season_map)

# Fill missing values in 'Daypart', 'DayType', and 'Season' with their respective mode
data['Daypart'] = data['Daypart'].fillna(data['Daypart'].mode()[0])
data['DayType'] = data['DayType'].fillna(data['DayType'].mode()[0])
data['Season'] = data['Season'].fillna(data['Season'].mode()[0])

# Feature selection
X = data[['Daypart', 'DayType', 'Season']]
y = data['Items']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Create an oversampler
ros = RandomOverSampler(random_state=42)

# Fit and resample
X_resampled, y_resampled = ros.fit_resample(X, y_encoded)

# Check new class distribution
print("New class distribution:\n", pd.Series(y_resampled).value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=16)

# Initialize and train the KNN model
model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Function to recommend best-selling items
def recommend_best_selling(daypart, daytype, season, top_n):
    # Convert inputs to codes
    daypart_code = daypart_map[daypart]
    daytype_code = daytype_map[daytype]
    season_code = season_map[season]

    # Create input data for prediction
    input_data = pd.DataFrame([[daypart_code, daytype_code, season_code]], columns=['Daypart', 'DayType', 'Season'])

    # Predict for the input data
    predictions = model.predict(input_data)

    # Create a DataFrame to hold all predictions and their corresponding features
    prediction_df = pd.DataFrame({
        'Predicted_Items': model.predict(X),
        'Daypart': data['Daypart'],
        'DayType': data['DayType'],
        'Season': data['Season']
    })

    # Get the nearest neighbors based on the input features
    distances, indices = model.kneighbors(input_data, n_neighbors=top_n)

    # Retrieve the predicted items for the nearest neighbors
    recommended_items = prediction_df.iloc[indices.flatten()]['Predicted_Items'].value_counts().head(top_n).index.tolist()

    return recommended_items

# Example usage
# recommended_items = recommend_best_selling('Morning', 'Weekday', 'Winter', top_n=5)
# print(f'Recommended best-selling items: {recommended_items}')