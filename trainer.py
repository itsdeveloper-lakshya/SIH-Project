import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
try:
    with open('./ISLdata.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("Data file not found. Ensure the correct path to the pickle file.")
    exit()

# Initialize lists to store consistent data
consistent_data = []
consistent_labels = []

# Iterate over the data to check for consistency
for i, data_point in enumerate(data_dict['data']):
    if len(data_point) == 84:  # Check for two hands (42 landmarks for each hand, x and y coordinates)
        consistent_data.append(data_point)
        consistent_labels.append(data_dict['labels'][i])
    else:
        print(f"Inconsistent data at index {i}: {len(data_point)} features")

# Convert to numpy arrays
data = np.asarray(consistent_data)
labels = np.asarray(consistent_labels)

# Check if data is empty after filtering
if len(data) == 0:
    print("No consistent data found. Exiting.")
    exit()

# Proceed with train-test split and training
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Added random_state for reproducibility
model.fit(x_train, y_train)

# Make predictions and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the model
with open('ISLmodel.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print("Model saved successfully!")