import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import warnings
import matplotlib.pyplot as plt
import torch.optim as optim
import networkx as nx
import seaborn as sns
from io import StringIO

import gdown

warnings.filterwarnings(action='ignore')

# Define the GNN model (simplified for this example)
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
    
# # # Thay thế URL dưới đây bằng URL chia sẻ của file Google Drive
# url = 'https://drive.google.com/uc?id=1EO-Wb6hPnNnv8KSaVBexbcaqa5LBb-pq'
# data_path = 'loan.csv'  # Tên file bạn muốn lưu

# gdown.download(url, data_path, quiet=False)

# Load the scaler and model
scaler_path = 'GNN_Scaler.pkl'
model_path = 'best_model.pth'
# data_path = 'data/loan.csv'

# url = 'https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv'
# df = pd.read_csv(url, index_col=0)
# print(df.head(5))

try:
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()
    
input_dim = 37  # Update this if you have a different number of features
hidden_dim = 16
output_dim = 1
model = SimpleGNN(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

def preprocess_data(df):
    # Tiền xử lý dữ liệu
    df = df.copy()

    # Xử lý giá trị thiếu
    df['emp_length'] = df['emp_length'].fillna(df['emp_length'].mode()[0])
    df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())

    # Chuyển đổi cột 'term' và 'emp_length'
    df['term'] = df['term'].str.replace(' months', '').astype(int)
    df['emp_length'] = df['emp_length'].str.replace(r'[a-zA-Z]', '', regex=True).str.replace(' ', '')

    # Label encoding cho các biến phân loại
    df['grade'] = df['grade'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
    df['sub_grade'] = df['sub_grade'].map({f'{ch}{num}': idx for idx, (ch, num) in enumerate([(ch, num) for ch in 'ABCDEFG' for num in range(1, 6)])})
    df['emp_length'] = df['emp_length'].map({'<1': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10+': 10})
    df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

    # One hot encoding cho các biến phân loại
    df = pd.get_dummies(df, columns=['home_ownership', 'verification_status', 'purpose'], drop_first=True)

    return df

# Tạo container cho sidebar
sidebar = st.sidebar.container()

# Thêm widget vào sidebar
with sidebar:
    st.header('Navigation')
    nav_item = st.radio('Go to', ('Home', 'Loan Predict', 'Data Visualization'))

# Tạo container cho nội dung chính
content = st.container()

# Khởi tạo biến df_predict rỗng
df_predict = pd.DataFrame()

# Thêm nội dung vào container chính
with content:
    if nav_item == 'Home':
        st.markdown("<h1 style='font-size: 35px;'><u>Lending-Club Loan Prediction App</u></h1>", unsafe_allow_html=True)
        st.markdown("<h4>Using the power of Artificial Neural Networks to make informed financial decisions</h4>", unsafe_allow_html=True)
        message = """
        **Welcome to the Loan Status Predictor App!🥳**\n
        With just a few inputs, our model can predict whether a loan is Fully Paid or Charged Off.\n
        Thank you for using our Loan Status Predictor App!🤞
        """
        st.markdown(message)

    elif nav_item == 'Loan Predict':
        st.title('Upload CSV File')
        
        # Tải file CSV từ người dùng
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            # Đọc dữ liệu từ file CSV
            df = pd.read_csv(uploaded_file)
            st.write('Data from CSV file:')
            # st.dataframe(df)  # Hiển thị toàn bộ dữ liệu
            df_display = df[['id', 'member_id']].copy()
            df_display['ID'] = df_display['id'].astype(str).str.replace(',', '')
            df_display['MemberID'] = df_display['member_id'].astype(str).str.replace(',', '')

            # Xóa các cột gốc sau khi đã tạo cột mới
            df_display = df_display.drop(columns=['id', 'member_id'])

            # Hiển thị DataFrame chỉ với các cột mới
            st.dataframe(df_display)

            # Chọn ID khoản vay cần dự đoán hoặc dự đoán tất cả
            prediction_option = st.radio('Prediction Option', ['Predict All Loans', 'Predict Specific Loan'])
            
            if prediction_option == 'Predict Specific Loan':
                loan_id = st.selectbox('Select Loan ID', df['id'].unique())
                if loan_id:
                    if loan_id in df['id'].values:
                        df_predict = df[df['id'] == loan_id]
                        member_id = df_predict['member_id'].values[0]
                        zip_code = df_predict['zip_code'].values[0]
                        addr_state = df_predict['addr_state'].values[0]
                        
                        st.write(f"Loan ID: {loan_id}")
                        st.write(f"Member ID: {member_id}")
                        st.write(f"Zip Code: {zip_code}")
                        st.write(f"Address State: {addr_state}")

                        if not df_predict.empty:
                            # Nút dự đoán cho khoản vay cụ thể
                            if st.button('Predict Specific Loan'):
                                # Chuẩn bị dữ liệu cho dự đoán
                                df_predict = preprocess_data(df_predict)

                                # Đảm bảo các cột trùng khớp với khi huấn luyện
                                feature_names = scaler.feature_names_in_  # Update this if needed
                                for col in feature_names:
                                    if col not in df_predict.columns:
                                        df_predict[col] = 0
                                df_predict = df_predict[feature_names]

                                # Scale các đặc trưng
                                df_predict_scaled = scaler.transform(df_predict)

                                # Dự đoán
                                with torch.no_grad():
                                    input_tensor = torch.tensor(df_predict_scaled, dtype=torch.float32)
                                    predictions = model(input_tensor).numpy().flatten()
                                    predictions = (predictions > 0.5).astype(int)

                                # Hiển thị dự đoán
                                st.write(f"Loan ID: {loan_id} - Prediction: {'Fully Paid' if predictions[0] == 0 else 'Charged Off'}")

                                # Vẽ đồ thị
                                st.write("**Visualizing Input Data as Graph**")

                                # Create a graph
                                G = nx.Graph()

                                # Add nodes with feature values from the first row of df_predict
                                for col in df_predict.columns:
                                    if col != 'loan_status':  # Exclude target variable
                                        G.add_node(col, value=df_predict[col].values[0])

                                # Add edges (connections between features)
                                features = list(df_predict.columns)
                                for i in range(len(features)):
                                    for j in range(i + 1, len(features)):
                                        G.add_edge(features[i], features[j])
                                
                                # Draw the graph
                                pos = nx.spring_layout(G)
                                node_labels = nx.get_node_attributes(G, 'value')
                                plt.figure(figsize=(12, 12))
                                nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
                                nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): '' for u, v in G.edges()}, font_color='red')
                                plt.title('Input Data Graph')
                                st.pyplot(plt)

            elif prediction_option == 'Predict All Loans':
                if st.button('Predict All Loans'):
                    df_predict = preprocess_data(df)

                    # Đảm bảo các cột trùng khớp với khi huấn luyện
                    feature_names = scaler.feature_names_in_  # Update this if needed
                    for col in feature_names:
                        if col not in df_predict.columns:
                            df_predict[col] = 0
                    df_predict = df_predict[feature_names]

                    # Scale các đặc trưng
                    df_predict_scaled = scaler.transform(df_predict)

                    # Dự đoán
                    with torch.no_grad():
                        input_tensor = torch.tensor(df_predict_scaled, dtype=torch.float32)
                        predictions = model(input_tensor).numpy().flatten()
                        predictions = (predictions > 0.5).astype(int)
                        predictions = np.where(predictions == 1, 'Charged Off', 'Fully Paid')  # Thay đổi 0-1 thành 'Fully Paid' hoặc 'Charged Off'
                    
                    # Thêm dự đoán vào DataFrame
                    df['loan_status_pred'] = predictions
                    st.write("**Predictions for All Loans:**")
                    st.dataframe(df[['id', 'loan_status_pred']])

                    # Nút xuất kết quả
                    st.write("**Export Predictions to CSV**")

                    # Tạo file CSV với dự đoán
                    output = StringIO()
                    df[['id', 'loan_status_pred']].to_csv(output, index=False)
                    st.download_button(
                        label="Download CSV",
                        data=output.getvalue(),
                        file_name='loan_predictions.csv',
                        mime='text/csv'
                    )

                    # Vẽ đồ thị mô hình
                    st.write("**Visualizing Model as Graph**")

                    # Create a graph for the model
                    G = nx.Graph()

                    # Add nodes with feature names
                    for feature in feature_names:
                        G.add_node(feature)

                    # Add edges (connections between features)
                    features = list(feature_names)
                    for i in range(len(features)):
                        for j in range(i + 1, len(features)):
                            G.add_edge(features[i], features[j])
                    
                    # Draw the graph
                    pos = nx.spring_layout(G)
                    plt.figure(figsize=(12, 12))
                    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightgreen', font_size=10, font_weight='bold')
                    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): '' for u, v in G.edges()}, font_color='red')
                    plt.title('Model Feature Graph')
                    st.pyplot(plt)

    # elif nav_item == 'Data Visualization':
    #     st.title('Data Visualization')
    #     st.write('Visualize the relationship between loan features and loan status.')

    #     # Đọc dữ liệu từ file loan.csv
    #     df_read = pd.read_csv(data_path,low_memory=False)
    #     # Calculate the percentage of missing values in df_read
    #     percentage_missing = df_read.isnull().sum() / len(df_read) * 100

    #     # Create a new DataFrame with columns from df and index set to None
    #     new_df = pd.DataFrame(columns=df_read.columns, index=None)
    #     pd.set_option('display.max_columns', None)

    #     # Creating new Percentage index
    #     new_df.loc['Percentage'] = percentage_missing.values
    #     new_df

    #     # Keeping only those features with less than 20% of missing values
    #     features_to_keep = df_read.columns[((df_read.isnull().sum()/len(df_read))*100 < 20)].to_list()
    #     # print("Total features before:",len(df_read.columns))
    #     # print("Total features now:",len(features_to_keep))

    #     df1=df_read[features_to_keep]
    #     lucky_features=['loan_amnt','term', 'int_rate', 'installment', 'grade', 'sub_grade','emp_length','home_ownership',
    #             'annual_inc','verification_status','purpose','dti','delinq_2yrs','inq_last_6mths','open_acc',
    #             'pub_rec','revol_bal','revol_util','total_acc','last_pymnt_amnt','loan_status']
    #     # print(len(lucky_features))

    #     df=df1[lucky_features]
    #     print("Shape of the dataset:",df.shape)
    #     df.head()

    #     df.describe()

    #     df_read['loan_status'].unique()

    #     target_loan= ["Fully Paid","Charged Off"]
    #     df=df[df["loan_status"].isin(target_loan)]
    #     print(df.shape)

    #     df.isnull().sum()

    #     df['emp_length']=df['emp_length'].fillna(df['emp_length'].mode()[0])
    #     df['revol_util']=df['revol_util'].fillna(df['revol_util'].median())

    #     # Biểu đồ Loan Status Count theo Term
    #     fig, ax = plt.subplots(figsize=(10, 4))
    #     sns.countplot(data=df, x="loan_status", hue="term", palette='dark', ax=ax)
    #     ax.set(xlabel='Status', ylabel='')
    #     ax.set_title('Loan status count', size=20)
    #     st.pyplot(fig)

    #     # Biểu đồ Loan Status Count theo Verification Status
    #     fig, ax = plt.subplots(figsize=(10, 4))
    #     sns.countplot(data=df, x="loan_status", hue="verification_status", palette='coolwarm', ax=ax)
    #     ax.set(xlabel='Status', ylabel='')
    #     ax.set_title('Loan status count', size=20)
    #     st.pyplot(fig)

    #     # Biểu đồ Loan Status Count theo Employment Length
    #     fig, ax = plt.subplots(figsize=(10, 4))
    #     sns.countplot(data=df, x="emp_length", palette='spring', ax=ax)
    #     ax.set(xlabel='Length of Employment', ylabel='')
    #     ax.set_title('Loan status count', size=20)
    #     plt.xticks(rotation='vertical')
    #     st.pyplot(fig)

    #     # Biểu đồ Loan Grades Count
    #     fig, ax = plt.subplots(figsize=(10, 4))
    #     sns.countplot(data=df, y="grade", palette='rocket', ax=ax)
    #     ax.set_title('Loan Grades count', size=20)
    #     st.pyplot(fig)

    #     # Biểu đồ Purpose vs Loan Amount
    #     fig, ax = plt.subplots(figsize=(10, 4))
    #     sns.barplot(data=df, x="purpose", y='loan_amnt', palette='spring', ax=ax)
    #     ax.set(xlabel='Purpose', ylabel='Amount')
    #     ax.set_title('Purpose vs Loan Amount', size=20)
    #     plt.xticks(rotation='vertical')
    #     st.pyplot(fig)

    #     # Biểu đồ Home Ownership vs Annual Income
    #     fig, ax = plt.subplots(figsize=(10, 4))
    #     sns.barplot(data=df, x="home_ownership", y='annual_inc', palette='viridis', ax=ax)
    #     ax.set(xlabel='Home Ownership', ylabel='Annual Income')
    #     ax.set_title('Home Ownership vs Annual Income', size=20)
    #     st.pyplot(fig)

    #     # Biểu đồ Heatmap
    #     plt.figure(figsize=(15, 10))
    #     sns.heatmap(df.corr(), annot=True)
    #     plt.title('Heatmap of Features')
    #     st.pyplot(plt)

    #     # Chia các đặc trưng thành categorical và numerical
    #     categorical = [feature for feature in df.columns if df[feature].dtype == 'object']
    #     numerical = [feature for feature in df.columns if feature not in categorical]

    #     # st.write(f"Categorical columns: {categorical}")
    #     # st.write(f"Numerical columns: {numerical}")

    #     # Biểu đồ Histplot cho mỗi biến trong danh sách numerical
    #     def histplot_visual(data, column):
    #         fig, ax = plt.subplots(3, 5, figsize=(15, 6))
    #         fig.suptitle('Histplot for each variable', y=1, size=20)
    #         ax = ax.flatten()
    #         for i, feature in enumerate(column):
    #             sns.histplot(data=data[feature], ax=ax[i], kde=True)
    #     histplot_visual(data=df, column=numerical)
    #     plt.tight_layout()
    #     st.pyplot(plt)

    #     # Biểu đồ Boxplot cho mỗi biến trong danh sách numerical
    #     def boxplots_visual(data, column):
    #         fig, ax = plt.subplots(3, 5, figsize=(15, 6))
    #         fig.suptitle('Boxplot for each variable', y=1, size=20)
    #         ax = ax.flatten()
    #         for i, feature in enumerate(column):
    #             sns.boxplot(data=data[feature], ax=ax[i], orient='h')
    #             ax[i].set_title(feature + ', skewness is: ' + str(round(data[feature].skew(axis=0, skipna=True), 2)), fontsize=10)
    #             ax[i].set_xlim([min(data[feature]), max(data[feature])])
    #     boxplots_visual(data=df, column=numerical)
    #     plt.tight_layout()
    #     st.pyplot(plt) 

    #     def histplot_visual(data,column):
    #         fig, ax = plt.subplots(3,5,figsize=(15,6))
    #         fig.suptitle('Histplot for each variable',y=1, size=20)
    #         ax=ax.flatten()
    #         for i,feature in enumerate(column):
    #             sns.histplot(data=data[feature],ax=ax[i], kde=True)
    #     histplot_visual(data=df,column=numerical)
    #     plt.tight_layout()
    #     def histplot_visual(data,column):
    #         fig, ax = plt.subplots(3,5,figsize=(15,6))
    #         fig.suptitle('Histplot for each variable',y=1, size=20)
    #         ax=ax.flatten()
    #         for i,feature in enumerate(column):
    #             sns.histplot(data=data[feature],ax=ax[i], kde=True)
    #     histplot_visual(data=df,column=numerical)
    #     plt.tight_layout()

    #      # Chọn các cột cho trực quan hóa
    #     loan_info_cols = ['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade']
    #     borrower_info_cols = ['emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'purpose']
    #     loan_details_cols = ['dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'last_pymnt_amnt']

    #     st.subheader('Loan Information')
    #     fig, axs = plt.subplots(len(loan_info_cols), figsize=(10, 30))
    #     for i, col in enumerate(loan_info_cols):
    #         if col in df.columns:
    #             sns.histplot(data=df, x=col, hue='loan_status', multiple='stack', ax=axs[i])
    #             axs[i].set_title(f'{col} Distribution by Loan Status')
    #     st.pyplot(fig)

    #     st.subheader('Borrower Information')
    #     fig, axs = plt.subplots(len(borrower_info_cols), figsize=(10, 30))
    #     for i, col in enumerate(borrower_info_cols):
    #         if col in df.columns:
    #             sns.histplot(data=df, x=col, hue='loan_status', multiple='stack', ax=axs[i])
    #             axs[i].set_title(f'{col} Distribution by Loan Status')
    #     st.pyplot(fig)

    #     st.subheader('Loan Details')
    #     fig, axs = plt.subplots(len(loan_details_cols), figsize=(10, 45))
    #     for i, col in enumerate(loan_details_cols):
    #         if col in df.columns:
    #             sns.histplot(data=df, x=col, hue='loan_status', multiple='stack', ax=axs[i])
    #             axs[i].set_title(f'{col} Distribution by Loan Status')
    #     st.pyplot(fig)

    #     # Histplot for each variable in numerical list
    #     def histplot_visual(data,column):
    #         fig, ax = plt.subplots(3,5,figsize=(15,6))
    #         fig.suptitle('Histplot for each variable',y=1, size=20)
    #         ax=ax.flatten()
    #         for i,feature in enumerate(column):
    #             sns.histplot(data=data[feature],ax=ax[i], kde=True)
    #     histplot_visual(data=df,column=numerical)
    #     plt.tight_layout()