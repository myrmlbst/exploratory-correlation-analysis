# import libraries
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC # Import SVM classifier
from sklearn.ensemble import RandomForestClassifier # Import Random Forest classifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from pprint import pprint
# load the dresses-sales sample dataset from OpenML
dataset = fetch_openml('dresses-sales', version=2)
# Organize data
label_names = dataset.target_names
labels = dataset.target
feature_names = dataset.feature_names
features = dataset.data
attribute_name = {
 'V2': 'Style',
 'V3': 'Price',
 'V4': 'Rating',
 'V5': 'Size',
 'V6': 'Season',
 'V7': 'NeckLine',
 'V8': 'NeckLine',
 'V9': 'Waistline',
 'V10': 'Material',
 'V11': 'Fabric Type',
 'V12': 'Decoration',
 'V13': 'Pattern Type'
}
label_counts = labels.value_counts().sort_index()
feature_counts = {}
for feature in features.columns:
 unique_values = features[feature].unique()
 if feature not in feature_counts:
 feature_counts[feature] = {}
 for value in unique_values:
 feature_counts[feature][value] = {
 'class1': 0,
 'class2': 0
 }
for index, row in features.iterrows():
 for column, value in row.items():
 if labels[index] == '1':
 feature_counts[column][value]['class1'] += 1
 elif labels[index] == '2':
 feature_counts[column][value]['class2'] += 1
# Print labels
print("Class Labels: " + str(label_names))
print("Feature Labels: " + str(feature_names))
print("")
# Plot Classes value each occurrences
labels_names = label_counts.index
class1 = label_counts[labels_names[0]]
class2 = label_counts[labels_names[1]]
# Define colors for each bar
colors = ['blue', 'orange']
plt.bar([labels_names[0], labels_names[1]], [class1, class2], color=colors)
plt.xlabel("Class")
plt.ylabel("Number of Occurrences")
plt.title("Occurrences of Each Class")
plt.show()
# Plot each feature in relation to the label
for feature_name in feature_names:
 X = feature_counts[feature_name].keys()
 X = list(X)
 class1 = []
 class2 = []
 for feature in feature_counts[feature_name]:
 class1.append(feature_counts[feature_name][feature]['class1'])
 class2.append(feature_counts[feature_name][feature]['class2'])
 X_axis = np.arange(len(X))
 plt.figure(figsize=(14, 6))

 plt.bar(X_axis - 0.2, class1, 0.4, label='1', color=colors)
 plt.bar(X_axis + 0.2, class2, 0.4, label='2', color=colors)
 plt.xticks(X_axis, X)
 plt.xlabel(feature_name)
 plt.ylabel("Number of Occurrences")
 plt.title(f"Occurrences of Each Class in {feature_name}")
 plt.legend()
 plt.show()
features = pd.get_dummies(features, columns=['V2', 'V3', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13'])
# Split data into training and testing 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20)
# Train a model using Decision Tree algorithm
dtree = DecisionTreeClassifier(max_depth=3, random_state=101) # Adjust parameters as needed
dtree.fit(X_train, y_train)
# Train a model using KNN algorithm
knn = KNeighborsClassifier(n_neighbors=5) # Adjust parameters as needed
knn.fit(X_train, y_train)
# Train a model using SVM with linear kernel
svm_linear = SVC(kernel='linear') # Using linear kernel
svm_linear.fit(X_train, y_train)
# Train a model using SVM with polynomial kernel
svm_poly = SVC(kernel='poly') # Using polynomial kernel
svm_poly.fit(X_train, y_train)
# Train a model using Random Forest
rfm = RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=1, random_state=101, max_features=None,
 min_samples_leaf=30)
rfm.fit(X_train, y_train)
# Predict using the trained Decision Tree model
y_pred_dtree = dtree.predict(X_test)
# Predict using the trained KNN model
y_pred_knn = knn.predict(X_test)
# Predict using the trained SVM with linear kernel model
y_pred_svm_linear = svm_linear.predict(X_test)
# Predict using the trained SVM with polynomial kernel model
y_pred_svm_poly = svm_poly.predict(X_test)
# Predict using the trained Random Forest model
y_pred_rfm = rfm.predict(X_test)
# Validate the trained models
accuracy_dtree = accuracy_score(y_test, y_pred_dtree)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_svm_linear = accuracy_score(y_test, y_pred_svm_linear)
accuracy_svm_poly = accuracy_score(y_test, y_pred_svm_poly)
accuracy_rfm = accuracy_score(y_test, y_pred_rfm)
print("Decision Tree Accuracy Score: " + str(accuracy_dtree))
print("KNN Accuracy Score: " + str(accuracy_knn))
print("SVM with Linear Kernel Accuracy Score: " + str(accuracy_svm_linear))
print("SVM with Polynomial Kernel Accuracy Score: " + str(accuracy_svm_poly))
print("Random Forest Accuracy Score: " + str(accuracy_rfm))
print("")
# Predict a classification based on the user input
inputs = []
for feature_name in feature_names:
 if feature_name != 'V4':
 list_attribute = list(feature_counts[feature_name])
 list_attribute = list(filter(lambda x: not (isinstance(x, float) and np.isnan(x)), list_attribute))
 print(f'Please enter one of the following {attribute_name[feature_name]} {", ".join(list_attribute)}')
 attribute = input(f"Enter {attribute_name[feature_name]}: ").strip().lower() # Convert to lowercase
 print()
 if attribute.lower() not in [attr.lower() for attr in
 list_attribute]: # Convert list to lowercase for comparison
 print("Invalid input. Please try again.")
 break
 inputs.append(attribute)
 else:
 while True:
 attribute = input(f'Enter {attribute_name[feature_name]}(which could be a number from 1 to 5): ')
 print()
 if not attribute.isdigit():
 print("Invalid input. Please enter a number.")
 continue
 attribute = float(attribute)

 if attribute < 1 or attribute > 5:
 print("Invalid input. Please enter a number between 1 and 5.")
 continue
 inputs.append(attribute)
 break
if len(inputs) != len(feature_names):
 print("Input error. Please try again.")
else:
 # Create DataFrame with one row
 df_inputs = pd.DataFrame([inputs], columns=feature_names)
 input_feature = pd.get_dummies(df_inputs,
 columns=['V2', 'V3', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13'])
 input_feature = input_feature.reindex(columns=X_train.columns, fill_value=False)
 predict_class_dtree = dtree.predict(input_feature)
 predict_class_knn = knn.predict(input_feature)
 predict_class_svm_linear = svm_linear.predict(input_feature)
 predict_class_svm_poly = svm_poly.predict(input_feature)
 predict_class_rfm = rfm.predict(input_feature)
 print("Class of this dress using Decision Tree is: " + str(predict_class_dtree[0]))
 print("Class of this dress using KNN is: " + str(predict_class_knn[0]))
 print("Class of this dress using SVM with Linear Kernel is: " + str(predict_class_svm_linear[0]))
 print("Class of this dress using SVM with Polynomial Kernel is: " + str(predict_class_svm_poly[0]))
 print("Class of this dress using Random Forest is: " + str(predict_class_rfm[0]))
