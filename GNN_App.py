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
import requests

import gdown

warnings.filterwarnings(action='ignore')
# Hàm tải file từ Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

# Define the function to convert emp_length
def emp_length_to_num(emp_length):
    if emp_length == '<1':
        return 0
    elif emp_length == '10+':
        return 10
    else:
        return int(emp_length)

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

# Load the scaler and model
scaler_path = 'GNN_Scaler.pkl'
model_path = 'best_model.pth'
# data_path = 'data/loan.csv'

with open('dataset.pkl', 'rb') as file:
    dfi = pickle.load(file)

# Convert to a DataFrame if it's not already one
if not isinstance(dfi, pd.DataFrame):
    dfi = pd.DataFrame(dfi)

try:
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()
feature_names = scaler.feature_names_in_
# Initialize session state variables if not already present
if 'Loan_Information' not in st.session_state:
    st.session_state['Loan_Information'] = {
        'loan_amnt': 2400,
        'term': 36,
        'int_rate': 15.96,
        'installment': 84.33,
    }

if 'Borrower_Information' not in st.session_state:
    st.session_state['Borrower_Information'] = {
        'grade': 'B',
        'sub_grade': 'C5',
        'emp_length': '10+',
        'home_ownership': 'RENT',
        'annual_inc': 12252.0,
        'verification_status': 'Verified',
    }

if 'Loan_Details' not in st.session_state:
    st.session_state['Loan_Details'] = {
        'purpose': 'small_business',
        'dti': 8.72,
        'delinq_2yrs': 0.0,
        'inq_last_6mths': 2.0,
        'open_acc': 2.0,
        'pub_rec': 0.0,
        'revol_bal': 2956.0,
        'revol_util': 98.5,
        'total_acc': 10.0,
        'last_pymnt_amnt': 649.91
    }

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
    # df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

    # One hot encoding cho các biến phân loại
    df = pd.get_dummies(df, columns=['home_ownership', 'verification_status', 'purpose'], drop_first=True)

    return df

# CSS để thay đổi màu nền
background_css = """
<style>
body {
    background-color: #ADD8E6;  /* Màu nền xanh nhạt */
}
</style>
"""

# Chèn CSS vào ứng dụng
st.markdown(background_css, unsafe_allow_html=True)

# Tạo container cho sidebar
sidebar = st.sidebar.container()

# Thêm widget vào sidebar
with sidebar:
    st.header('Navigation')
    nav_item = st.radio('Go to', ('Home', 'Loan Predict from files', 'Loan Predict'))

# Tạo container cho nội dung chính
content = st.container()

# Khởi tạo biến df_predict rỗng
df_predict = pd.DataFrame()

# Thêm nội dung vào container chính
with content:
    if nav_item == 'Home':
        st.markdown("<h1 style='font-size: 35px;'><u>PREDICTING THE ABILITY OF INDIVIDUAL CUSTOMERS TO REPAY DEBT TO BANKS</u></h1>", unsafe_allow_html=True)
        st.markdown("<h4>USING THE GRAPH NEURAL NETWORK METHOD</h4>", unsafe_allow_html=True)
        message = """
        **Welcome to the Loan Status Predictor App!**\n
        Just fill in the input information or upload files, this model can predict whether a loan is Fully Paid or Charged Off.\n
        Thank you for using App!
        """
        st.markdown(message)

    elif nav_item == 'Loan Predict from files':

        # ID của file Google Drive
        file_id = '1uZwcoNFHiJb6gtthmGh-Ev-_n8X4vatv'
        destination = 'Data-Template.csv'

        # Tải file xuống
        download_file_from_google_drive(file_id, destination)

        # Nút tải xuống file mẫu
        st.markdown("### Download Sample CSV File")
        st.download_button(
            label="Download Sample CSV",
            data=open(destination, 'rb').read(),
            file_name=destination,
            mime='text/csv'
        )
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

    elif nav_item == 'Loan Predict':
        st.title('Loan Information')
        st.write('Enter Loan Information here.')
        
        loan_col1, loan_col2 = st.columns(2)
          
        with loan_col1:
            loan_amnt = st.number_input('Loan Amount (in $)', min_value=0, step=1000)
            term = st.selectbox('Term', [36, 60])
          
        with loan_col2:
            int_rate = st.number_input('Interest Rate', min_value=0.0, step=5.0, format="%.2f")
            installment = st.number_input('Monthly Payment', min_value=0.0, step=50.0, format="%.2f")

        st.session_state['Loan_Information'] = {
            'loan_amnt': loan_amnt,
            'term': term,
            'int_rate': int_rate,
            'installment': installment,
        }
        
        st.title('Borrower Information')
        st.write('Enter Borrower Information here.')
        
        borrower_col1, borrower_col2 = st.columns(2)
          
        with borrower_col1:
            grade = st.selectbox('Grade', np.sort(dfi['grade'].unique(), kind='mergesort'))
            sub_grade = st.selectbox('Sub-Grade', np.sort(dfi['sub_grade'].unique(), kind='mergesort'))
            emp_length = st.selectbox('Employment Length (in years)', ['<1', '1', '2', '3',
                                                    '4', '5', '6', '7',
                                                    '8', '9', '10+'])
        with borrower_col2:
            home_ownership = st.selectbox('Home Ownership', np.sort(dfi['home_ownership'].unique(), kind='mergesort'))
            home_ownership = home_ownership.replace(" ", "_")
            annual_inc = st.number_input('Annual Income', min_value=0.0, step=1000.0)
            verification_status = st.selectbox('Income Verification Status', np.sort(dfi['verification_status'].unique(), kind='mergesort'))
        
        st.session_state['Borrower_Information'] = {
            'grade': grade,
            'sub_grade': sub_grade,
            'emp_length': emp_length,
            'home_ownership': home_ownership,
            'annual_inc': annual_inc,
            'verification_status': verification_status,        
        }

        st.title('Loan Details')
        st.write('Enter Loan details here.')

        loan_detail_col1, loan_detail_col2 = st.columns(2)

        with loan_detail_col1:
            purpose = st.selectbox('Purpose for the loan', np.sort(dfi['purpose'].unique(), kind='mergesort'))
            dti = st.number_input('Debt-to-Income Ratio (DTI)', min_value=0.0, step=5.0)
            delinq_2yrs = st.number_input('Delinquency count in the past 2 years', min_value=0.0, step=1.0, max_value=50.0)
            inq_last_6mths = st.number_input('Number of inquiries in the last 6 months', min_value=0.0, step=1.0, max_value=50.0)
            open_acc = st.number_input('Number of open credit lines', min_value=0.0, step=1.0, max_value=50.0)

        with loan_detail_col2:
            pub_rec = st.number_input('Number of derogatory public records', min_value=0.0, step=1.0, max_value=50.0)
            revol_bal = st.number_input('Revolving balance', min_value=0.0, step=100.0)
            revol_util = st.number_input('Revolving line utilization rate (%)', min_value=0.0, step=5.0)
            total_acc = st.number_input('Total number of credit lines', min_value=0.0, step=1.0, max_value=50.0)
            last_pymnt_amnt = st.number_input('Last payment amount', min_value=0.0, step=100.0)

        st.session_state['Loan_Details'] = {
            'purpose': purpose,
            'dti': dti,
            'delinq_2yrs': delinq_2yrs,
            'inq_last_6mths': inq_last_6mths,
            'open_acc': open_acc,
            'pub_rec': pub_rec,
            'revol_bal': revol_bal,
            'revol_util': revol_util,
            'total_acc': total_acc,
            'last_pymnt_amnt': last_pymnt_amnt,
        }
        
        if st.button('Predict'):
            # Chuẩn bị dữ liệu đầu vào cho dự đoán
            loan_info = st.session_state['Loan_Information']
            borrower_info = st.session_state['Borrower_Information']
            loan_details = st.session_state['Loan_Details']
            input_data = {**loan_info, **borrower_info, **loan_details}
            input_df = pd.DataFrame([input_data])

            # Chuyển đổi và scale dữ liệu
            input_df['emp_length'] = input_df['emp_length'].apply(emp_length_to_num)
            input_df = pd.get_dummies(input_df)
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_names]
            input_scaled = scaler.transform(input_df)
            
            # Dự đoán
            with torch.no_grad():
                input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
                prediction = model(input_tensor).numpy().flatten()
                prediction = (prediction > 0.5).astype(int)
            
            st.write(f"Prediction: {'Fully Paid' if prediction[0] == 0 else 'Charged Off'}")
            
            # Vẽ đồ thị
            st.write("**Visualizing Input Data as Graph**")   

            # Create a graph
            G = nx.Graph()

            # Add nodes with feature values from input_df
            for col in input_df.columns:
                 if col != 'loan_status':  # Exclude target variable
                     G.add_node(col, value=input_df[col].values)
            # Add edges (connections between features)
            features = list(input_df.columns)
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
