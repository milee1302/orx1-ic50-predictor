import streamlit as st
import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
import gdown
import os
import zipfile
import glob
import joblib 

# --- GIAO DIỆN CHÍNH ---
st.title("Dự đoán ORX1 pIC50 bằng Ensemble XGBoost 🧬")
st.write("Ứng dụng sử dụng mô hình Ensemble XGBoost (50 models) và đặc trưng ECFP4. Áp dụng chiến thuật dự đoán đồng thuận (Consensus).")

# --- TẢI VÀ GIẢI NÉN MÔ HÌNH TỪ GOOGLE DRIVE ---
@st.cache_resource
def load_model_from_drive():
    file_id = '1_cCTY3euT-yPtBsp1wYW9P0yOadVB9Db'
    url = f'https://drive.google.com/uc?id={file_id}'
    zip_path = "model_orx1.zip"
    extract_folder = "model_extracted"
    
    if not os.path.exists(zip_path):
        with st.spinner("Đang tải file ZIP từ Drive và giải nén (68MB)... Chờ tí nhé bro!"):
            gdown.download(url, zip_path, quiet=False)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder) 
                
    pkl_files = glob.glob(f"{extract_folder}/**/*.pkl", recursive=True)
    if not pkl_files:
        raise FileNotFoundError("Không tìm thấy file .pkl nào bên trong cục ZIP!")
        
    ensemble_data = joblib.load(pkl_files[0])
    
    preprocessor = ensemble_data['preprocessor']
    models = ensemble_data['models']
    
    return preprocessor, models

try:
    preprocessor, models = load_model_from_drive()
    st.success(f"Tải thành công Bộ tiền xử lý và {len(models)} mô hình XGBoost! Đã sẵn sàng. 🚀")
except Exception as e:
    st.error(f"Lỗi khi tải mô hình. Lỗi chi tiết: {e}")
    st.stop()

# --- XỬ LÝ HÓA HỌC: SMILES -> ECFP4 ---
def smiles_to_ecfp4(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp).reshape(1, -1)

# --- KHUNG NHẬP LIỆU & DỰ ĐOÁN ĐỒNG THUẬN ---
st.markdown("---")
smiles_input = st.text_input("Nhập chuỗi SMILES của hợp chất vào đây:", "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F")

if st.button("Dự đoán pIC50", type="primary"):
    if smiles_input.strip() == "":
        st.warning("Bro chưa nhập chuỗi SMILES kìa!")
    else:
        with st.spinner("Đang cho 50 mô hình chạy đồng thuận..."):
            features_raw = smiles_to_ecfp4(smiles_input)
            
            if features_raw is not None:
                try:
                    features_proc = preprocessor.transform(features_raw)
                    
                    predictions = []
                    for sub_model in models:
                        pred = sub_model.predict(features_proc)[0]
                        predictions.append(pred)
                        
                    final_pIC50 = np.mean(predictions)
                    uncertainty = np.std(predictions)
                    
                    col1, col2 = st.columns(2)
                    col1.metric(label="pIC50 (Consensus Score)", value=f"{final_pIC50:.4f}")
                    col2.metric(label="Độ bất định (Std Dev)", value=f"{uncertainty:.4f}")
                    
                    st.balloons()
                except Exception as e:
                    st.error(f"Lỗi trong quá trình tính toán đặc trưng: {e}")
            else:
                st.error("Chuỗi SMILES không hợp lệ. Bro kiểm tra lại cấu trúc hóa học nhé.")
