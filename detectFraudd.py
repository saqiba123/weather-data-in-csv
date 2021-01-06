#import imblearn
from imblearn.over_sampling import SMOTE
import numpy as np
# import matplotlib library
import matplotlib.pyplot as plt
# Import pandas and read csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report, confusion_matrix
# This is the pipeline module we need for this from imblearn
from imblearn.pipeline import Pipeline
import seaborn as sns

df = pd.read_csv("creditcard.csv")
# Explore the features available in your dataframe
#print(df.info())
print(df.head())

# Count the occurrences of fraud and no fraud and print them ,only 492 fraud transactions
occ = df['Class'].value_counts()
print(occ)
# Print the ratio of fraud cases
print(occ / len(df))

# Convert the DataFrame into two variable
# X: data columns (V1 - V28)
# y: lable column


def prep_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray):

    X = df.iloc[:, 2:30].values
    y = df.Class.values
    return X, y


# Define a function to create a scatter plot of our data and labels
def plot_data(X, y):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    return plt.show()


# Create X and y from the prep_data function
X, y = prep_data(df)
# Plot our data by running our plot data function on X and y
plot_data(X, y)
# Run the prep_data function
X, y = prep_data(df)

# Define the resampling method
method = SMOTE(sampling_strategy="auto")

# Create the resampled feature set
X_resampled, y_resampled = method.fit_sample(X, y)
# Plot the resampled data
plot_data(X_resampled, y_resampled)
# Print the value_counts on the original labels y
print(pd.value_counts(pd.Series(y)))
# Print the value_counts
print(pd.value_counts(pd.Series(y_resampled)))

# Run compare_plot


def compare_plot(X: np.ndarray, y: np.ndarray, X_resampled: np.ndarray, y_resampled: np.ndarray, method: str):
    plt.subplot(1, 2, 1)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.title('Original Set')
    plt.subplot(1, 2, 2)
    plt.scatter(X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1], label="Class #0", alpha=0.5,
                linewidth=0.15)
    plt.scatter(X_resampled[y_resampled == 1, 0], X_resampled[y_resampled == 1, 1], label="Class #1", alpha=0.5,
                linewidth=0.15, c='r')
    plt.title(method)
    plt.legend()
    plt.show()


compare_plot(X, y, X_resampled, y_resampled, method='SMOTE')
# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Fit a logistic regression model to our data
model = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)
# Obtain model predictions
predicted = model.predict(X_test)
# Print the classification report and confusion matrix
print('Classification report:\n', classification_report(y_test, predicted))
class_names = ['not_fraud', 'fraud']
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)

# Create pandas data Frame
df1 = pd.DataFrame(conf_mat, index=class_names, columns=class_names)
# Create heat Map
sns.heatmap(df1, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

# Define which ReSampling method and which ML model to use in the pipeline
resampling = SMOTE(sampling_strategy='borderline2')
model = LogisticRegression()
# Define the pipeline, tell it to combine SMOTE with the Logistic Regression model
pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model)])