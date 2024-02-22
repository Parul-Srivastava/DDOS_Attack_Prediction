pip install lime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
accuracy_score, f1_score, recall_score, precision_score,
confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve,
auc
)
import matplotlib.pyplot as plt
from lime import lime_tabular
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/ML/dosattack.csv')
# Load dataset
data_path = '/content/drive/MyDrive/ML/dosattack.csv'
data = pd.read_csv(data_path)
data = data.dropna()
print(data)
# Print size and features of dataset
print("Shape of dataset:", data.shape)
print("Columns: ", data.columns)
# Select features and target
X = data[['pktcount', 'bytecount', 'dur', 'dur_nsec', 'tot_dur', 'flows',
'packetins',
'pktperflow', 'byteperflow', 'pktrate', 'Pairflow', 'Protocol',
'port_no',
'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps']]
y = data['label']
X_encoded = pd.get_dummies(X, columns=['Protocol'])
# Split data for testing and training
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y,
test_size=0.3, random_state=42)
print(X)
# Perform ANOVA feature selection
k_best = SelectKBest(score_func=f_classif, k=4)
X_train_best = k_best.fit_transform(X_train, y_train)
X_test_best = k_best.transform(X_test)
# Get the selected feature names
selected_feature_names = X_encoded.columns[k_best.get_support()]
print(k_best)
print(X_train_best)
print(X_test_best)
print(selected_feature_names)
from sklearn.neighbors import KNeighborsClassifier
Knn = KNeighborsClassifier(n_neighbors=9)
Knn.fit(X_train_best,y_train)
y_pred= Knn.predict(X_test_best)
print("Test set predictions: {}", format(Knn.predict(X_test_best)))
# Model Evaluation
accuracy = Knn.score(X_test_best, y_test)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print("Accuracy Score:", accuracy)
print("F1-Score:", f1)
print("Recall:", recall)
print("Precision:", precision)
# Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)
# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# ROC Curve
y_prob = Knn.predict_proba(X_test_best)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area =
%0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve
(area = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()
# Lime Explainer
explainer = lime_tabular.LimeTabularExplainer(X_train_best,
feature_names=selected_feature_names.tolist(),
class_names=['Normal',
'Attack'], discretize_continuous=True)
instance_to_explain = X_test_best[0]
# Explain the prediction
lime_explanation = explainer.explain_instance(instance_to_explain,
Knn.predict_proba)
# Visualize the explanation
lime_explanation.show_in_notebook()
joblib.dump(Knn, '/content/drive/My Drive/ML/Knn.pkl')