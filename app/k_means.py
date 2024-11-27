import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os



# Load the data
data = pd.read_csv('app/dataset.csv')


# Dictionary to map input strings to corresponding codes
daypart_map = {'Night': 3, 'Morning': 2, 'Afternoon': 1, 'Evening': 0}
daytype_map = {'Weekend': 0, 'Weekday': 1}
season_map = {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}

# Preprocess the data
data['Daypart'] = data['Daypart'].map(daypart_map)
data['DayType'] = data['DayType'].map(daytype_map)
data['Season'] = data['Season'].map(season_map)

# Fill missing values with their respective mode
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

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Determine the optimal number of clusters (optional)
# You can use silhouette score or elbow method to find the best k
kmeans = KMeans(n_clusters=5, random_state=42)  # Change n_clusters as needed
kmeans.fit(X_scaled)

# Assign clusters to the original data
data['Cluster'] = kmeans.predict(scaler.transform(X))
# silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
# print(f'Silhouette Score: {silhouette_avg}')

#persist the model with pickle
with open('kmeans_model.pk1', 'wb') as model_file : 
    pickle.dump (kmeans, model_file)
with open ('scaler.pk1', 'wb') as scaler_file: 
    pickle.dump(scaler, scaler_file)

#resuse model
with open ('kmeans_model.pk1', 'rb') as model_file: 
    kmeans_loaded = pickle.load(model_file)
with open ('scaler.pk1', 'rb') as scaler_file : 
    scaler_loaded = pickle.load(scaler_file)
# Function to recommend items based on clustering
def recommend_items(daypart, daytype, season, top_n=5):
    # Convert inputs to codes
    daypart_code = daypart_map[daypart]
    daytype_code = daytype_map[daytype]
    season_code = season_map[season]

    
    input_data = pd.DataFrame([[daypart_code, daytype_code, season_code]], columns=['Daypart', 'DayType', 'Season'])
    

    input_scaled = scaler_loaded.transform(input_data)

    cluster_label = kmeans_loaded.predict(input_scaled)[0]

    # Recommend items from the same cluster
    recommended_items = data[data['Cluster'] == cluster_label]['Items'].value_counts().head(top_n).index.tolist()

    return recommended_items

# Example usage
# recommended_items = recommend_items('Afternoon', 'Weekend', 'Spring', top_n=5)
# print(f'Recommended best-selling items: {recommended_items}')
plt.switch_backend('Agg')

def Seasonal_Preference_Analysis():
    # Load data
    df = pd.read_csv('app/dataset.csv')
    
    # Data preparation
    seasonal_sales = df.groupby(['Season', 'Items']).size().reset_index(name='Total_Sales')
    seasonal_sales_sorted = seasonal_sales.sort_values(by=['Season', 'Total_Sales'], ascending=[True, False])
    top_items_per_season = seasonal_sales_sorted.groupby('Season').head(5)
    
    print('Top items per season:\n', top_items_per_season)
    
    # Adjust palette to handle more items
    unique_items = top_items_per_season['Items'].nunique()
    palette = sns.color_palette("Set2", unique_items)

    # 1. Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_items_per_season, x='Season', y='Total_Sales', hue='Items', palette=palette)
    plt.title('Top 5 Items per Season')
    plt.xlabel('Season')
    plt.ylabel('Total Sales')
    plt.legend(title='Items', bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig('app/static/images/bar_chart.png')  # Save image
    plt.close()

    # 2. Line Graph
    plt.figure(figsize=(10, 6))
    top_items = ['Coffee', 'Bread', 'Tea', 'Cake', 'Sandwich', 'Alfajores', 'Scone']
    for item in top_items:
        item_data = top_items_per_season[top_items_per_season['Items'] == item]
        plt.plot(item_data['Season'], item_data['Total_Sales'], marker='o', label=item)
    plt.title('Seasonal Trend of Top Items')
    plt.xlabel('Season')
    plt.ylabel('Total Sales')
    plt.legend(title='Items')
    plt.savefig('app/static/images/line_graph.png')  # Save image
    plt.close()

    # 3. Stacked Bar Chart
    season_sales_pivot = top_items_per_season.pivot(index='Season', columns='Items', values='Total_Sales').fillna(0)
    season_sales_pivot.plot(kind='bar', stacked=True, figsize=(10, 6), color=palette)
    plt.title('Seasonal Contribution of Each Item')
    plt.xlabel('Season')
    plt.ylabel('Total Sales')
    plt.legend(title='Items', bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig('app/static/images/stacked_bar_chart.png')  # Save image
    plt.close()

    # 4. Pie Charts
    seasons = top_items_per_season['Season'].unique()
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Proportion of Each Item per Season')
    for ax, season in zip(axs.flatten(), seasons):
        season_data = top_items_per_season[top_items_per_season['Season'] == season]
        pie_palette = sns.color_palette("Set2", season_data.shape[0])
        ax.pie(season_data['Total_Sales'], labels=season_data['Items'], autopct='%1.1f%%', startangle=90, colors=pie_palette)
        ax.set_title(season)
    plt.tight_layout()
    plt.savefig('app/static/images/pie_charts.png')  # Save image
    plt.close()
# Seasonal_Preference_Analysis()

