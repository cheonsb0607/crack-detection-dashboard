import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# 학습된 모델 불러오기
def load_trained_model(model_path='autoencoder_model.h5'):
    model = load_model(model_path)
    return model

# 이미지 예측 함수
def predict_image(model, image):
    # 이미지 전처리
    image = image.resize((224, 224))  # 모델 입력 사이즈 맞추기
    image = np.array(image) / 255.0  # 정규화
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    
    # 예측 (재구성 손실 계산)
    loss = model.predict(image)
    reconstruction_loss = np.mean(np.abs(image - loss))  # 재구성 오차 계산
    return reconstruction_loss

# Streamlit 앱
st.set_page_config(page_title="균열 탐지 대시보드", layout="wide")
page = st.sidebar.selectbox("페이지 선택", ["1. 이미지 업로드", "2. 모델 성능", "3. 예측 결과"])

if page == "1. 이미지 업로드":
    st.header("📷 이미지 업로드")
    method = st.radio("이미지 업로드 방법 선택", ["파일 업로드", "카메라 촬영"])
    
    if method == "파일 업로드":
        uploaded_files = st.file_uploader("이미지를 업로드하세요", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    else:
        uploaded_files = st.camera_input("사진을 촬영하세요")
    
    if uploaded_files:
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        
        st.success(f"{len(uploaded_files)}개의 이미지가 업로드되었습니다.")
        if st.button("예측 시작"):
            model = load_trained_model()  # 모델 불러오기
            st.session_state['uploaded_images'] = uploaded_files
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                reconstruction_loss = predict_image(model, image)
                # 예측 결과 출력
                if reconstruction_loss < 0.2:
                    level = "🟢 안전"
                elif reconstruction_loss < 0.5:
                    level = "🟡 주의"
                else:
                    level = "🔴 위험"
                st.success(f"이미지 예측 결과: {level} (Loss: {reconstruction_loss:.3f})")
            st.success("이미지가 저장되었습니다. 상단에서 '3. 예측 결과' 페이지로 이동하세요.")
        
elif page == "2. 모델 성능":
    st.header("📊 모델 성능 평가")
    tp = st.number_input("TP (True Positive)", min_value=0, value=50)
    tn = st.number_input("TN (True Negative)", min_value=0, value=40)
    fp = st.number_input("FP (False Positive)", min_value=0, value=5)
    fn = st.number_input("FN (False Negative)", min_value=0, value=5)
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt="d", cmap="Blues", xticklabels=["정상", "균열"], yticklabels=["정상", "균열"])
    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    st.pyplot(fig)

    st.markdown(f"**Accuracy**: {accuracy:.2f}")
    st.markdown(f"**Precision**: {precision:.2f}")
    st.markdown(f"**Recall**: {recall:.2f}")
    st.markdown(f"**F1 Score**: {f1:.2f}")

elif page == "3. 예측 결과":
    st.header("🧠 예측 결과 및 위험도 평가")
    uploaded_images = st.session_state.get('uploaded_images', None)

    if not uploaded_images:
        st.warning("1번 페이지에서 이미지를 업로드해주세요.")
    else:
        for idx, file in enumerate(uploaded_images):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(file, caption=f"이미지 {idx+1}", width=250)

            with col2:
                # 실제 예측된 위험도 표시
                dummy_loss = np.random.rand()  # 이 부분을 실제 예측 값으로 변경
                if dummy_loss < 0.2:
                    level = "🟢 안전"
                elif dummy_loss < 0.5:
                    level = "🟡 주의"
                else:
                    level = "🔴 위험"
                st.markdown(f"### 위험도: {level} (Loss: {dummy_loss:.3f})")
