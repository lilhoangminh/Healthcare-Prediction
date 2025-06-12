import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import re

st.set_page_config(
    page_title="Dự đoán bệnh từ triệu chứng",
    page_icon="🏥",
    layout="wide"
)

def load_data():
    dataset_path = "dataset/Disease Symptom Prediction/dataset.csv"
    symptom_desc_path = "dataset/Disease Symptom Prediction/symptom_Description.csv"
    symptom_precaution_path = "dataset/Disease Symptom Prediction/symptom_precaution.csv"
    doctor_disease_path = "dataset/Doctor's Specialty Recommendation/Doctor_Versus_Disease.csv"
    disease_desc_path = "dataset/Doctor's Specialty Recommendation/Disease_Description.csv"
    
    # read CSV file
    df_symptoms = pd.read_csv(dataset_path)
    df_symptom_desc = pd.read_csv(symptom_desc_path)
    df_precaution = pd.read_csv(symptom_precaution_path)
    df_doctor = pd.read_csv(doctor_disease_path)
    df_disease_desc = pd.read_csv(disease_desc_path)
    
    return df_symptoms, df_symptom_desc, df_precaution, df_doctor, df_disease_desc

def prepare_data(df_symptoms):    
    # get all diseases list
    diseases = df_symptoms['Disease'].unique()
    disease_to_idx = {disease: i for i, disease in enumerate(diseases)}
    idx_to_disease = {i: disease for i, disease in enumerate(diseases)}
    
    # get all symptons from dataset
    all_symptoms = set()
    for col in df_symptoms.columns:
        if col.startswith('Symptom_'):
            symptoms = df_symptoms[col].dropna().unique()
            all_symptoms.update(symptoms)
    
    # convert to list and remove NaN value
    all_symptoms = [s for s in all_symptoms if isinstance(s, str)]
    symptom_to_idx = {symptom: i for i, symptom in enumerate(all_symptoms)}
    
    X = []  # list of symptons
    y = []  # disease label

    for _, row in df_symptoms.iterrows():
        disease = row['Disease']
        disease_idx = disease_to_idx[disease]
        
        # get symptons from this disease
        symptoms = []
        for col in df_symptoms.columns:
            if col.startswith('Symptom_') and pd.notna(row[col]):
                symptoms.append(row[col])

        # append to train data
        X.append(symptoms)
        y.append(disease_idx)
    
    return X, y, all_symptoms, symptom_to_idx, disease_to_idx, idx_to_disease

def symptoms_to_input(symptoms, all_symptoms):
    input_vector = [0] * len(all_symptoms)    
    for i, symptom in enumerate(all_symptoms):
        if symptom in symptoms:
            input_vector[i] = 1
    
    return np.array([input_vector])  

# create and train model
def create_and_train_model(X_processed, y, input_size, output_size):
    """Tạo và huấn luyện mô hình neural network đơn giản"""
    
    # neuron network model
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_size,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(output_size, activation='softmax')
    ])
    
    # compile model 
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # train model
    model.fit(X_processed, y, epochs=20, batch_size=8, verbose=0)
    
    return model

# handle input value 
def process_user_input(user_input, all_symptoms):
    # convert to lower case
    user_input = user_input.lower()
    user_input = re.sub(r'[^\w\s]', ' ', user_input)
    
    # split input value
    words = user_input.split()
    
    # find suitable symptons
    matched_symptoms = []
    
    for symptom in all_symptoms:
        symptom_words = symptom.lower().replace('_', ' ').split()
        
        # check if input value contains symptons
        for symptom_word in symptom_words:
            if any(symptom_word in word or word in symptom_word for word in words):
                matched_symptoms.append(symptom)
                break
    
    return matched_symptoms

# get disease info
def get_disease_info(disease, df_precaution, df_disease_desc, df_doctor):
    """Lấy thông tin về bệnh, biện pháp phòng ngừa và bác sĩ chuyên khoa"""
    
    # get disease description
    description = ""
    desc_row = df_disease_desc[df_disease_desc['Disease'] == disease]
    if not desc_row.empty and 'Description' in desc_row.columns:
        description = desc_row['Description'].iloc[0]
    
    # get preventive measures
    precautions = []
    precaution_row = df_precaution[df_precaution['Disease'] == disease]
    if not precaution_row.empty:
        for i in range(1, 5):
            col_name = f'Precaution_{i}'
            if col_name in precaution_row.columns:
                precaution = precaution_row[col_name].iloc[0]
                if pd.notna(precaution):
                    precautions.append(precaution)
    
    # get all doctor specialtity
    doctor_specialist = "Bác sĩ đa khoa"
    for col in df_doctor.columns:
        if disease in df_doctor[col].values:
            doctor_specialist = col
            break
    
    return description, precautions, doctor_specialist

def main():
    st.title("🏥 Dự đoán bệnh từ triệu chứng")
    st.write("Nhập các triệu chứng bạn đang gặp phải để nhận dự đoán về bệnh và lời khuyên từ bác sĩ.")
    st.markdown("---")
    
    # read data
    with st.spinner("Đang tải dữ liệu..."):
        df_symptoms, df_symptom_desc, df_precaution, df_doctor, df_disease_desc = load_data()
    
    with st.spinner("Đang chuẩn bị dữ liệu..."):
        X, y, all_symptoms, symptom_to_idx, disease_to_idx, idx_to_disease = prepare_data(df_symptoms)
        
        # to data to train
        X_processed = []
        for symptoms in X:
            X_processed.append([1 if s in symptoms else 0 for s in all_symptoms])
        X_processed = np.array(X_processed)
    
    # train model
    with st.spinner("Đang huấn luyện mô hình..."):
        model = create_and_train_model(X_processed, y, len(all_symptoms), len(disease_to_idx))
    
    st.success("✅ Mô hình đã sẵn sàng!")
    
    # sidebar
    with st.sidebar:
        st.header("📋 Thông tin")
        st.write(f"Số lượng triệu chứng: {len(all_symptoms)}")
        st.write(f"Số loại bệnh: {len(disease_to_idx)}")
        
        st.header("💡 Hướng dẫn")
        st.write("1. Nhập các triệu chứng vào ô bên dưới")
        st.write("2. Nhấn nút 'Dự đoán bệnh'")
        st.write("3. Xem kết quả và lời khuyên")
        
        st.header("⚠️ Lưu ý")
        st.warning("Đây chỉ là công cụ hỗ trợ, không thay thế cho việc khám bác sĩ!")
    
    
    st.header("💬 Nhập triệu chứng của bạn")
    
    # show some examples
    with st.expander("🔍 Xem ví dụ triệu chứng"):
        st.write("• sốt cao, ho, đau đầu, mệt mỏi")
        st.write("• đau bụng, buồn nôn, tiêu chảy")
        st.write("• khó thở, đau ngực, tim đập nhanh")
        st.write("• đau khớp, sưng khớp, cứng khớp")
    
    # symptons input
    user_input = st.text_area(
        "Mô tả các triệu chứng bạn đang gặp phải:",
        placeholder="Ví dụ: sốt cao, ho khan, đau đầu...",
        height=100
    )
    
    # predict button
    predict_button = st.button("🔍 Dự đoán bệnh", type="primary")
    
    # predict button handle
    if predict_button and user_input.strip():
        with st.spinner("Đang phân tích triệu chứng..."):
            # user's input handle
            matched_symptoms = process_user_input(user_input, all_symptoms)
            
            if not matched_symptoms:
                st.warning("⚠️ Không nhận diện được triệu chứng nào. Vui lòng thử lại với mô tả chi tiết hơn.")
            else:
                # sympton recognize
                st.success(f"✅ Đã nhận diện {len(matched_symptoms)} triệu chứng:")
                
                # sympton recognize
                symptom_display = [s.replace('_', ' ').title() for s in matched_symptoms]
                st.write(", ".join(symptom_display))
                
                # input -> model
                input_vector = symptoms_to_input(matched_symptoms, all_symptoms)
                
                # predict disease
                prediction = model.predict(input_vector, verbose=0)
                predicted_idx = np.argmax(prediction[0])
                confidence = prediction[0][predicted_idx]
                
                # get disease name from index
                predicted_disease = idx_to_disease[predicted_idx]
                
                # show result prediction
                st.markdown("---")
                st.header("🎯 Kết quả dự đoán")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Bệnh dự đoán", predicted_disease)
                with col2:
                    st.metric("Độ tin cậy", f"{confidence:.1%}")
                
                # get disease info
                description, precautions, doctor_specialist = get_disease_info(
                    predicted_disease, df_precaution, df_disease_desc, df_doctor
                )
                
                # disease info
                st.markdown("---")
                st.header("📖 Thông tin về bệnh")
                
                if description:
                    st.write(description)
                else:
                    st.write("Không có thông tin chi tiết về bệnh này.")
                
                # recommend method
                if precautions:
                    st.markdown("---")
                    st.header("🛡️ Biện pháp phòng ngừa")
                    
                    for i, precaution in enumerate(precautions, 1):
                        st.write(f"{i}. {precaution.replace('_', ' ').title()}")
                
                # doctor speciality recommend
                st.markdown("---")
                st.header("👨‍⚕️ Bác sĩ chuyên khoa")
                st.info(f"Bạn nên đến khám: **{doctor_specialist.replace('_', ' ').title()}**")
                
                # Attention note
                st.markdown("---")
                st.warning("⚠️ **Lưu ý quan trọng:** Kết quả này chỉ mang tính chất tham khảo. Vui lòng tham khảo ý kiến bác sĩ để có chẩn đoán chính xác.")
    
    elif predict_button:
        st.warning("⚠️ Vui lòng nhập triệu chứng trước khi dự đoán!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>🏥 Hệ thống Dự đoán Bệnh | Được phát triển bằng TensorFlow & Streamlit</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
