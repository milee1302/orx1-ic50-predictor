import streamlit as st
import numpy as np
import pickle
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
import gdown
import os
import zipfile
import glob # Thêm thư viện để tự động tìm file

# --- GIAO DIỆN CHÍNH ---
st.title("Dự đoán ORX1 IC50 bằng Ensemble XGBoost 🧬")
st.write("Ứng dụng sử dụng mô hình Ensemble XGBoost (50 models) và đặc trưng ECFP4 để dự đoán hoạt tính ức chế.")

# --- TẢI MÔ HÌNH TỪ GOOGLE DRIVE ---
@st.cache_resource
def load_model_from_drive():
    file_id = '1_cCTY3euT-yPtBsp1wYW9P0yOadVB9Db'
    url = f'https://drive.google.com/uc?id={file_id}'
    zip_path = "model_orx1.zip"
    extract_folder = "model_extracted"
    
    # 1. Tải file ZIP (nếu chưa tải)
    if not os.path.exists(zip_path):
        with st.spinner("Đang tải file ZIP từ Drive và giải nén (68MB)... Chờ tí nhé bro!"):
            gdown.download(url, zip_path, quiet=False)
            
            # 2. Giải nén file ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder) 
                
    # 3. Tự động quét tìm file .pkl sau khi giải nén (bất chấp file nằm trong thư mục con)
    pkl_files = glob.glob(f"{extract_folder}/**/*.pkl", recursive=True)
    if not pkl_files:
        raise FileNotFoundError("Giải nén xong nhưng không tìm thấy file .pkl nào bên trong cục ZIP của bro!")
        
    pkl_path = pkl_files[0] # Lấy file pkl đầu tiên tìm được
    
    # 4. Đọc model
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model_from_drive()
    st.success("Tải và giải nén mô hình thành công! Đã sẵn sàng dự đoán. 🚀")
except Exception as e:
    st.error(f"Lỗi khi tải mô hình. Vui lòng kiểm tra lại. Lỗi chi tiết: {e}")
    st.stop()

# --- XỬ LÝ HÓA HỌC: SMILES -> ECFP4 ---
def smiles_to_ecfp4(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp).reshape(1, -1)

# --- KHUNG NHẬP LIỆU & DỰ ĐOÁN ---
st.markdown("---")
smiles_input = st.text_input("Nhập chuỗi SMILES của hợp chất vào đây:", "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F")

if st.button("Dự đoán IC50", type="primary"):
    if smiles_input.strip() == "":
        st.warning("Bro chưa nhập chuỗi SMILES kìa!")
    else:
        with st.spinner("Đang tính toán..."):
            features = smiles_to_ecfp4(smiles_input)
            
            if features is not None:
                prediction = model.predict(features)
                st.metric(label="Giá trị dự đoán (IC50 / pIC50)", value=f"{prediction[0]:.4f}")
                st.balloons()
            else:
                st.error("Chuỗi SMILES không hợp lệ. Bro kiểm tra lại cấu trúc hóa học nhé.")
