import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

st.title('Streamlit Example')

st.write("""
# Explore different classifier and datasets
Which one is the best?
""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Pedestrian detection data', 'Baldness Prediction dataset', 'Sign Language MNIST')
)


st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)
def get_dataset(name):
    data = None
    if name == 'Pedestrian detection data':
        data = datasets.load_iris()
    elif name == 'Baldness Prediction dataset':
        data = datasets.load_wine()
    elif name == 'Sign Language MNIST':
        # Correct dataset name and use fetch_openml
        data = datasets.fetch_openml(name="sign_mnist")
    else:
        # Handle other datasets or raise an error
        raise ValueError("Unknown dataset name")
    
    X = data.data
    y = data.target
    return X, y

import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming you have a function get_dataset() that returns X, y

# Function to add UI parameters based on the chosen classifier
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

# Function to get the selected classifier based on user input
def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                      max_depth=params['max_depth'], random_state=1234)
    return clf

# Streamlit UI
st.title("Classifier Training App")



# Display dataset information
X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('Number of classes:', len(np.unique(y)))

# Add UI parameters for the selected classifier
params = add_parameter_ui(classifier_name)

# Button to trigger training and evaluation
if st.button("Train and Evaluate"):
    # Get the selected classifier
    clf = get_classifier(classifier_name, params)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate and display results
    acc = accuracy_score(y_test, y_pred)
    st.write(f'Classifier = {classifier_name}')
    st.write(f'Accuracy =', acc)
    st.button("Train and Evaluate", key="reset_button")  # Reset button color after evaluation

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming you have a function get_dataset() that returns X, y

# Streamlit UI
st.title("Dataset Visualization")

# Display dataset information
X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('Number of classes:', len(np.unique(y)))

#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot
scatter = ax1.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')
ax1.set_title('Scatter Plot')

# Add colorbar to the entire figure
cbar = plt.colorbar(scatter, ax=ax1)

# Histogram
ax2.hist(x1, bins=30, color='blue', alpha=0.7, label='PC1')
ax2.hist(x2, bins=30, color='orange', alpha=0.7, label='PC2')
ax2.set_xlabel('Principal Components')
ax2.set_ylabel('Frequency')
ax2.set_title('Histogram')
ax2.legend()

# Adjust layout
plt.tight_layout()

# Display the plot using st.pyplot()
st.pyplot(fig)


