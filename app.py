import streamlit as st
import numpy as np
import pickle
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
import gdown
import os

# --- GIAO DIỆN CHÍNH ---
st.title("Dự đoán ORX1 IC50 bằng Ensemble XGBoost 🧬")
st.write("Ứng dụng sử dụng mô hình Ensemble XGBoost (50 models) và đặc trưng ECFP4 để dự đoán hoạt tính ức chế.")

# --- TẢI MÔ HÌNH TỪ GOOGLE DRIVE ---
@st.cache_resource
def load_model_from_drive():
    # MÃ FILE ID CỦA BRO ĐÃ ĐƯỢC ĐIỀN SẴN Ở ĐÂY:
    file_id = '1-DNXbXsKLlD6wyXB2XoZq38jtHPhq3tg'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = "Ensemble_50_XGBoost_Raw_ECFP4_MAX.pkl"
    
    # Kiểm tra nếu chưa có file trên server Streamlit thì mới tải
    if not os.path.exists(output):
        with st.spinner("Đang tải mô hình từ Google Drive (68MB)... Lần đầu chạy sẽ mất khoảng 10-20 giây nhé bro!"):
            gdown.download(url, output, quiet=False)
            
    with open(output, "rb") as f:
        return pickle.load(f)

# Gọi hàm load model (có bọc try-except để bắt lỗi nếu quên mở quyền Public Drive)
try:
    model = load_model_from_drive()
    st.success("Tải mô hình thành công! Đã sẵn sàng dự đoán. 🚀")
except Exception as e:
    st.error(f"Lỗi khi tải mô hình. Vui lòng kiểm tra lại quyền chia sẻ file trên Google Drive. Lỗi chi tiết: {e}")
    st.stop()

# --- XỬ LÝ HÓA HỌC: SMILES -> ECFP4 ---
def smiles_to_ecfp4(smiles, radius=2, n_bits=2048):
    """
    Lưu ý: Nếu lúc train model bro dùng radius hay n_bits khác, hãy sửa lại số ở đây cho khớp nhé!
    """
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
                # Dự đoán với mô hình đã tải
                prediction = model.predict(features)
                
                # Hiển thị kết quả nổi bật
                st.metric(label="Giá trị dự đoán (IC50 / pIC50)", value=f"{prediction[0]:.4f}")
                st.balloons() # Thêm hiệu ứng bóng bay cho ngầu
            else:
                st.error("Chuỗi SMILES không hợp lệ. Bro kiểm tra lại cấu trúc hóa học nhé.")