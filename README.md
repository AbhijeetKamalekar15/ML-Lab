# ML-Lab

# Linear Regression

# Import necessary libraries
import numpy as np  # Used for numerical operations, arrays, etc.
import pandas as pd  # Used for data manipulation and analysis (e.g., reading CSV)
import matplotlib.pyplot as plt  # Used for creating visualizations like scatter plots
# Read the CSV file containing the dataset into a DataFrame
file = pd.read_csv('/content/Linear Regression - Sheet1.csv')
# Display the first 5 rows of the dataset to get an overview
file.head()
# Import the LinearRegression class from scikit-learn and rename it as 'lr'
from sklearn.linear_model import LinearRegression as lr
# Create an instance of the Linear Regression model
model = lr()
# Define 'x' as the DataFrame without the 'X' column (all other columns are features)
x = file.drop(['X'], axis=1)
# Define 'y' as the 'X' column of the DataFrame (the target variable)
y = file['X']
# Fit the Linear Regression model to the data (x = features, y = target)
model.fit(x, y)
# Get the coefficients (slope) of the model and print them (the importance of each feature)
model.coef_
# Get the intercept (the y-axis intercept of the regression line) and print it
model.intercept_
# Predict the target value when the input feature value is 10
model.predict([[10]])
# Predict the target value again for the input value of 10 (duplicate line, same as above)
model.predict([[10]])
# Predict the target values for feature values of 30 and 4.5 (array of inputs)
model.predict([[30], [4.5]])
# Create a scatter plot of the actual data points
plt.scatter(x, y)


# Logistic Regression

# Importing necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and reading files
import matplotlib.pyplot as plt  # For creating plots
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.metrics import accuracy_score  # To evaluate model performance
from sklearn.impute import SimpleImputer # For handling missing values

# Load the CSV file containing the dataset into a DataFrame
file = pd.read_csv('/content/framingham.csv')

# Display the first few rows of the dataset to inspect its structure
file.head()

# Define 'X' as the feature columns we will use for prediction (all columns except 'diabetes')
x = file[['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp']]

# Define 'y' as the target column (whether the person has diabetes or not)
y = file['diabetes']

# Create an instance of the SimpleImputer to fill missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the feature data and transform it
x = imputer.fit_transform(x)

# Create an instance of the Logistic Regression model
model = LogisticRegression()

# Fit the Logistic Regression model to the training data
model.fit(x, y)

# Predict the target variable for the test set
y_pred = model.predict(x) # Use the imputed data for predictions

# Print the accuracy score of the model
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy}")

# Create a scatter plot of actual vs predicted values (for two selected features)
plt.scatter(x[:, 1], y, color='blue', label='Actual') # Use the imputed data for plotting
plt.scatter(x[:, 1], y_pred, color='red', label='Predicted') # Use the imputed data and predictions for plotting
plt.title('Logistic Regression: Actual vs Predicted')
plt.xlabel('Age')
plt.ylabel('Diabetes (0: No, 1: Yes)')
plt.legend()
plt.show()

# Display model coefficients (weights) and intercept
print(f"Model Coefficient (Weights): {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

# Polynomial Regression

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (Framingham dataset)
df = pd.read_csv('framingham.csv')

# For this example, let's predict cigsPerDay (number of cigarettes per day) based on age and male.
# You can replace this with other features and target based on your need.

# Select the features and the target variable
X = df[['age', 'male']]  # Independent variables
y = df['cigsPerDay']      # Dependent variable (Target)

# Drop rows with missing values in either X or y
df = df.dropna(subset=['age', 'male', 'cigsPerDay']) # Drop rows with missing values from the DataFrame

# Select the features and the target variable after dropping missing values
X = df[['age', 'male']]  # Independent variables
y = df['cigsPerDay']      # Dependent variable (Target)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply polynomial transformation to the features
degree = 2  # Degree of the polynomial regression
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions
y_pred = model.predict(X_test_poly)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Visualizing the result (only works with one feature, so let's just use 'age' for this)
plt.scatter(X_test['age'], y_test, color='blue', label='Actual')
plt.scatter(X_test['age'], y_pred, color='red', label='Predicted')
plt.title('Polynomial Regression (degree = {})'.format(degree))
plt.xlabel('Age')
plt.ylabel('Cigarettes Per Day')
plt.legend()
plt.show()

# KNN classifier on appropriate classifier

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generating synthetic data with blobs
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=1.5, random_state=4)

# Setting up the plot style (note: seaborn style deprecated in matplotlib 3.6+)
plt.style.use('seaborn')

# Plotting the blobs in a figure with a size of 10x10
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y, marker='*', s=100, edgecolors='black')
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Generate synthetic data using make_blobs
X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1.5, random_state=4)

# Plot the dataset
plt.style.use('seaborn')
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y, marker='*', s=100, edgecolors='black')
plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Train a KNeighborsClassifier with k=5
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)

# Train a KNeighborsClassifier with k=1 for comparison
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, y_train)

# Predict the labels for the test set using k=5 and k=1 classifiers
y_pred_5 = knn5.predict(X_test)
y_pred_1 = knn1.predict(X_test)

# Calculate and print the accuracy of both classifiers
from sklearn.metrics import accuracy_score
print("Accuracy with k=5:", accuracy_score(y_test, y_pred_5))
print("Accuracy with k=1:", accuracy_score(y_test, y_pred_1))

# Confusion Matrix with appropriate example

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# 1. Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = knn.predict(X_test)

# 5. Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# 6. Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 7. Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Support Vector Machine

from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm  # Importing SVM

# Generating synthetic data
X, y = datasets.make_classification(n_samples=100, n_features=10, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Training an SVM model
clf = svm.SVC()  # Using SVC for Support Vector Classifier
clf.fit(X_train, y_train)

# Making predictions
y_pred = clf.predict(X_test)

# Evaluating performance
print(metrics.classification_report(y_test, y_pred))
acu = metrics.accuracy_score(y_test, y_pred)
print(acu)
conf = metrics.confusion_matrix(y_test, y_pred)
print(conf)

# Visualizing results
plt.scatter(X_test[:, 0], y_pred)
plt.show()

# Import necessary libraries
from sklearn import metrics, datasets, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data
X, y = datasets.make_classification(n_samples=100, n_features=10, n_classes=2)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Visualize the training data (first two features for simplicity)
plt.scatter(X_train[:, 0], X_train[:, 1])
plt.xlabel("Feature 1")  # Label for x-axis
plt.ylabel("Feature 2")  # Label for y-axis
plt.title("Training Data Scatter Plot")  # Title of the plot
plt.show()

# Initialize and train an SVM model
clf = svm.SVC()
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
print(metrics.classification_report(y_test, y_pred))

# Calculate accuracy and confusion matrix
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Visualize the predictions (first feature vs predicted class)
plt.scatter(X_test[:, 0], y_pred)
plt.xlabel("Feature 1")  # Label for x-axis
plt.ylabel("Predicted Class")  # Label for y-axis
plt.title("Predicted Values Scatter Plot")  # Title of the plot
plt.show()


# K-means Clustering Algorithm

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
x=[4,5,10,4,3,11,14,6,10,12]
y=[21,19,24,17,16,25,24,22,21,21]
data=list(zip(x,y))
print(data)

inertias=[]

for i in range(1,11):
  kmeans=KMeans(n_clusters=i)
  kmeans.fit(data)
  inertias.append(kmeans.inertia_)

plt.plot(range(1,11),inertias,marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

kmeans=KMeans(n_clusters=3)
kmeans.fit(data)

plt.scatter(x,y,c=kmeans.labels_)
plt.show()

import pandas as pd
df = pd.read_csv("xclara.csv")
X = df[["V1", "V2"]]
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

labels = kmeans.labels_
df["cluster"] = labels
print(df.groupby("cluster").mean())
print(df.head())
plt.scatter(df["V1"], df["V2"], c=df["cluster"])
plt.xlabel("V1")
plt.ylabel("V2")
plt.show()

# Anomaly detection model to identify unusual patterns

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
n_samples=500
X, y= make_blobs(n_samples=n_samples, centers=5, random_state=7,cluster_std=0.6)
anomalies=np.array([[5, 5], [6, 6], [7, 7]])
X= np.vstack([X, anomalies])
plt.scatter(X[:, 0],X[:, 1], c='b', marker='o', s=25)
plt.title("Synthetic Dataset")
plt.show()
dbscan = DBSCAN(eps=0.5, min_samples=30)
labels=dbscan.fit_predict(X)
anomalies= X[labels == -1]
anomalies =X[labels == -1]
plt.scatter(X[:, 0],X[:, 1], c='b', marker='o', s=25)
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='r', marker='x', s=50, label='Anomalies')
plt.title("Anomaly Detection with DBSCAN (Anomalies Outside Clusters)")
plt.legend()
plt.show()
print("Identified Anomalies:")
print(anomalies)

# single layer and multilayer perceptron

x_input = [0.1, 0.5, 0.2]
w_input = [0.4, 0.3, 0.6]
b_input = 0.5
threshold =0.3

def step (weighted_sum):
  if weighted_sum > threshold:
    return 1
  else:
    return 0

def perceptron():
  weighted_sum = 0
  for x, w in zip(x_input, w_input):
    weighted_sum += x*w
    weighted_sum += b_input
    return step(weighted_sum)

output=perceptron()
print("output:" + str(output))
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neural_network
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X, y = datasets.make_classification(n_samples=100, n_features=10, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape, X_test.shape)

plt.scatter(X_train[:, 0], X_train[:, 1])
plt.show()
clf = neural_network.MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred))

acu = metrics.accuracy_score(y_test, y_pred)
print(acu)
conf = metrics.confusion_matrix(y_test, y_pred)
print(conf)

# CNN on suitable dataset

# Import TensorFlow
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    #which is why we need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

# Adam is the best among the adaptive optimizers in most of the cases
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# An epoch means training the neural network with all the
# training data for one cycle. Here I use 10 epochs
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))


plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,
                                     test_labels,
                                     verbose=2)

print('Test Accuracy is',test_acc)
