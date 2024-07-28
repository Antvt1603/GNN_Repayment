import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

# Accessing the dataset
dataset_path = "loan.csv"

# Reading the dataset
df_read = pd.read_csv(dataset_path, low_memory=False)
print("Shape of the dataset:", df_read.shape)
df_read.head()

df_read.info()

# Data Cleaning
percentage_missing = df_read.isnull().sum() / len(df_read) * 100
# Create a new DataFrame with columns from df and index set to None
new_df = pd.DataFrame(columns=df_read.columns, index=None)
pd.set_option('display.max_columns', None)

# Creating new Percentage index
new_df.loc['Percentage'] = percentage_missing.values
new_df

features_to_keep = df_read.columns[(df_read.isnull().sum() / len(df_read) * 100 < 20)].to_list()
print("Total features before:",len(df_read.columns))
print("Total features now:",len(features_to_keep))

df1 = df_read[features_to_keep]
df1.shape

lucky_features = ['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length', 
                  'home_ownership', 'annual_inc', 'verification_status', 'purpose', 'dti', 'delinq_2yrs', 
                  'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 
                  'last_pymnt_amnt', 'loan_status']
df = df1[lucky_features]
print("Shape of the dataset:", df.shape)
df.head()

df.describe()

df_read['loan_status'].unique()

# Filter target loan status
target_loan = ["Fully Paid", "Charged Off"]
df = df[df["loan_status"].isin(target_loan)]
print(df.shape)

df.isnull().sum()

# Handling null values
df['emp_length'] = df['emp_length'].fillna(df['emp_length'].mode()[0])
df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(20, 10))

# Loan status count by term
sns.countplot(data=df, x="loan_status", hue="term", palette='dark', ax=axes[0, 0])
axes[0, 0].set(xlabel='Status', ylabel='')
axes[0, 0].set_title('Loan status count by term', size=20)

# Loan status count by verification status
sns.countplot(data=df, x="loan_status", hue="verification_status", palette='coolwarm', ax=axes[0, 1])
axes[0, 1].set(xlabel='Status', ylabel='')
axes[0, 1].set_title('Loan status count by verification status', size=20)

# Loan status count by employment length
sns.countplot(data=df, x="emp_length", palette='spring', ax=axes[1, 0])
axes[1, 0].set(xlabel='Length of Employment', ylabel='')
axes[1, 0].set_title('Loan status count by employment length', size=20)
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=90)

# Loan Grades count
sns.countplot(data=df, y="grade", palette='rocket', ax=axes[1, 1])
axes[1, 1].set_title('Loan Grades count', size=20)

# Adjust layout
plt.tight_layout()
plt.show()

# Combine two barplots into one figure
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# Purpose vs Loan Amount
sns.barplot(data=df, x="purpose", y='loan_amnt', palette='spring', ax=axes[0])
axes[0].set(xlabel='Purpose', ylabel='Amount')
axes[0].set_title('Purpose vs Loan Amount', size=20)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)

# Home Ownership vs Annual Income
sns.barplot(data=df, x="home_ownership", y='annual_inc', palette='viridis', ax=axes[1])
axes[1].set(xlabel='Home Ownership', ylabel='Annual Income')
axes[1].set_title('Home Ownership vs Annual Income', size=20)

# Adjust layout
plt.tight_layout()
plt.show()

# Plotting a heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()

# Dividing our features into categorical and numerical
categorical=[feature for feature in df.columns if df[feature].dtype=='object']
numerical=[feature for feature in df.columns if feature not in categorical]
print("Categorical columns:",categorical)
print("Numerical columns:",numerical)

# Histplot for each variable in numerical list
def histplot_visual(data,column):
    fig, ax = plt.subplots(3,5,figsize=(15,6))
    fig.suptitle('Histplot for each variable',y=1, size=20)
    ax=ax.flatten()
    for i,feature in enumerate(column):
        sns.histplot(data=data[feature],ax=ax[i], kde=True)
histplot_visual(data=df,column=numerical)
plt.tight_layout()

# Boxplot for each variable in numerical list
def boxplots_visual(data,column):
    fig, ax = plt.subplots(3,5,figsize=(15,6))
    fig.suptitle('Boxplot for each variable',y=1, size=20)
    ax=ax.flatten()
    for i,feature in enumerate(column):
        sns.boxplot(data=data[feature],ax=ax[i], orient='h')
        ax[i].set_title(feature+ ', skewness is: '+str(round(data[feature].skew(axis = 0, skipna = True),2)),fontsize=10)
        ax[i].set_xlim([min(data[feature]), max(data[feature])])
boxplots_visual(data=df,column=numerical)
plt.tight_layout()


# Preprocessing
df['term'] = df['term'].str.replace(' months', '').astype(int)
df['emp_length'] = df['emp_length'].str.replace(r'[a-zA-Z]', '', regex=True).str.replace(' ', '', regex=False)

# Label Encoding
df['grade'] = df['grade'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
df['sub_grade'] = df['sub_grade'].map({'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 'A5': 4,
                                      'B1': 5, 'B2': 6, 'B3': 7, 'B4': 8, 'B5': 9,
                                      'C1': 10, 'C2': 11, 'C3': 12, 'C4': 13, 'C5': 14,
                                      'D1': 15, 'D2': 16, 'D3': 17, 'D4': 18, 'D5': 19,
                                      'E1': 20, 'E2': 21, 'E3': 22, 'E4': 23, 'E5': 24,
                                      'F1': 25, 'F2': 26, 'F3': 27, 'F4': 28, 'F5': 29,
                                      'G1': 30, 'G2': 31, 'G3': 32, 'G4': 33, 'G5': 34})
df['emp_length'] = df['emp_length'].map({'<1': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 
                                        '6': 6, '7': 7, '8': 8, '9': 9, '10+': 10})
df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

# One hot encoding
df = pd.get_dummies(df, columns=['home_ownership', 'verification_status', 'purpose'], drop_first=True)
df.head()

# Model Training
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Splitting our dataset between training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

# Define a function to evaluate model performance
def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        test_preds = []
        test_labels = []
        for features, labels in test_loader:
            outputs = model(features).squeeze()
            preds = (outputs >= 0.5).float()
            test_preds.extend(preds.tolist())
            test_labels.extend(labels.tolist())

        test_accuracy = accuracy_score(test_labels, test_preds)
        test_precision = precision_score(test_labels, test_preds, zero_division=0)
        test_recall = recall_score(test_labels, test_preds, zero_division=0)
        test_f1 = f1_score(test_labels, test_preds, zero_division=0)
        test_roc_auc = roc_auc_score(test_labels, test_preds) if len(set(test_labels)) > 1 else None

        test_roc_auc_str = f"{test_roc_auc:.4f}" if test_roc_auc is not None else "N/A"
        conf_matrix = confusion_matrix(test_labels, test_preds)
        class_report = classification_report(test_labels, test_preds)
        
        return test_accuracy, test_precision, test_recall, test_f1, test_roc_auc_str, conf_matrix, class_report

# Define the custom dataset class
class LoanDefaultDataset(Dataset):
    def __init__(self, features, labels):
        self.features = np.array(features)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Define the GNN model 
class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]
hidden_dim = 16
output_dim = 1
model = SimpleGNN(input_dim, hidden_dim, output_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define a function to train and evaluate the model
def train_and_evaluate(X_train, y_train, X_test, y_test):
    train_dataset = LoanDefaultDataset(X_train, y_train)
    test_dataset = LoanDefaultDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train the model
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate the model
    return evaluate_model(model, test_loader)

# MinMax Scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

df['loan_status'].value_counts()
print("Fully Paid:",df['loan_status'].value_counts()[0]/len(df['loan_status'])*100)
print("Charged Off:",df['loan_status'].value_counts()[1]/len(df['loan_status'])*100)

# Save the scaler
scaler_path = 'GNN_Scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# Check if the scaler file has been saved
if os.path.exists(scaler_path):
    print(f"{scaler_path} has been saved successfully.")
else:
    print(f"Failed to save {scaler_path}.")


# Biểu đồ: Trước khi xử lý mất cân bằng dữ liệu
plt.figure(figsize=(8, 6))
plt.title('Before Balancing')
df['loan_status'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Handling imbalanced data and evaluating performance

# Over-sampling using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
accuracy_smote, precision_smote, recall_smote, f1_smote, roc_auc_smote, conf_matrix_smote, class_report_smote = train_and_evaluate(X_train_smote, y_train_smote, X_test_scaled, y_test)

# Under-sampling using NearMiss
nearmiss = NearMiss()
X_train_nm, y_train_nm = nearmiss.fit_resample(X_train_scaled, y_train)
accuracy_nm, precision_nm, recall_nm, f1_nm, roc_auc_nm, conf_matrix_nm, class_report_nm = train_and_evaluate(X_train_nm, y_train_nm, X_test_scaled, y_test)

# Combined over- and under-sampling using SMOTETomek
smotetomek = SMOTETomek(random_state=42)
X_train_smotetomek, y_train_smotetomek = smotetomek.fit_resample(X_train_scaled, y_train)
accuracy_smotetomek, precision_smotetomek, recall_smotetomek, f1_smotetomek, roc_auc_smotetomek, conf_matrix_smotetomek, class_report_smotetomek = train_and_evaluate(X_train_smotetomek, y_train_smotetomek, X_test_scaled, y_test)

# Displaying results
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_smote)))
print(f"SMOTE - Accuracy: {accuracy_smote:.4f}, Precision: {precision_smote:.4f}, Recall: {recall_smote:.4f}, F1-score: {f1_smote:.4f}, ROC AUC: {roc_auc_smote}")
print(f"Confusion Matrix:\n{conf_matrix_smote}")
print(f"Classification Report:\n{class_report_smote}")
print()
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_nm)))
print(f"NearMiss - Accuracy: {accuracy_nm:.4f}, Precision: {precision_nm:.4f}, Recall: {recall_nm:.4f}, F1-score: {f1_nm:.4f}, ROC AUC: {roc_auc_nm}")
print(f"Confusion Matrix:\n{conf_matrix_nm}")
print(f"Classification Report:\n{class_report_nm}")
print()
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_smotetomek)))
print(f"SMOTETomek - Accuracy: {accuracy_smotetomek:.4f}, Precision: {precision_smotetomek:.4f}, Recall: {recall_smotetomek:.4f}, F1-score: {f1_smotetomek:.4f}, ROC AUC: {roc_auc_smotetomek}")
print(f"Confusion Matrix:\n{conf_matrix_smotetomek}")
print(f"Classification Report:\n{class_report_smotetomek}")

# Choose the best method based on Accuracy, F1-score, and ROC AUC
methods = ['SMOTE', 'NearMiss', 'SMOTETomek']
recall=[recall_smote,recall_nm,recall_smotetomek]
f1_scores = [f1_smote, f1_nm, f1_smotetomek]
roc_auc_scores = [roc_auc_smote, roc_auc_nm, roc_auc_smotetomek]
accuracies = [accuracy_smote, accuracy_nm, accuracy_smotetomek]
# Define weights for each metric
weight_recall = 0.4
weight_f1 = 0.3
weight_roc_auc = 0.2
weight_accuracy = 0.1

# Ensure all roc_auc_scores are floats
roc_auc_scores = [float(score) for score in roc_auc_scores]

# Compute scores for each method
scores = []
for i in range(len(methods)):
    score = (weight_recall * recall[i]+ 
             weight_f1 * f1_scores[i] +
             weight_roc_auc * roc_auc_scores[i] + 
             weight_accuracy * accuracies[i])
    scores.append(score)

# Output the scores
for method, score in zip(methods, scores):
    print(f"{method}: {score:.4f}")

# Find the method with the highest score
best_index = np.argmax(scores)
best_method = methods[best_index]
best_recall = methods[best_index]
best_f1 = f1_scores[best_index]
best_roc_auc = roc_auc_scores[best_index]
best_accuracy = accuracies[best_index]

print(f'Best method: {best_method}')
print(f'Accuracy: {best_accuracy:.4f}, Precision: {precision_smote:.4f} (or corresponding precision), Recall: {recall_smote:.4f} (or corresponding recall), F1-score: {best_f1:.4f}, ROC-AUC: {best_roc_auc}')

# Final model training using the best method
if best_method == 'SMOTE':
    X_train_best, y_train_best = X_train_smote, y_train_smote
elif best_method == 'NearMiss':
    X_train_best, y_train_best = X_train_nm, y_train_nm
elif best_method == 'SMOTETomek':
    X_train_best, y_train_best = X_train_smotetomek, y_train_smotetomek

train_dataset = LoanDefaultDataset(X_train_best, y_train_best)
test_dataset = LoanDefaultDataset(X_test_scaled, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Tạo lại DataFrame sau khi cân bằng
df_balanced = pd.DataFrame(X_train_best, columns=df.columns.drop('loan_status'))
df_balanced['loan_status'] = y_train_best

# Biểu đồ: Sau khi xử lý mất cân bằng dữ liệu
plt.figure(figsize=(8, 6))
plt.title('After Balancing')
df_balanced['loan_status'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

print(df_balanced)

# Re-initialize the model, loss function, and optimizer
model = SimpleGNN(input_dim, hidden_dim, output_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
best_model_path = 'best_model.pth'
best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Evaluation on the test set
    test_accuracy, test_precision, test_recall, test_f1, test_roc_auc_str, conf_matrix, class_report = evaluate_model(model, test_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1-score: {test_f1:.4f}, Test ROC-AUC: {test_roc_auc_str}')

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'Saving best model with accuracy: {best_accuracy:.4f}')

print(f'Best model saved with accuracy: {best_accuracy:.4f}')


# Load and evaluate the saved model
model.load_state_dict(torch.load(best_model_path))
model.eval()
with torch.no_grad():
    test_preds = []
    test_labels = []
    for features, labels in test_loader:
        outputs = model(features).squeeze()
        test_preds.extend(outputs.tolist())  # Dự đoán xác suất thay vì nhãn
        test_labels.extend(labels.tolist())

# Tính toán và vẽ ROC Curve
fpr, tpr, thresholds = roc_curve(test_labels, test_preds)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


