Python 3.13.2 (tags/v3.13.2:4f8bb39, Feb  4 2025, 15:23:48) [MSC v.1942 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> from tensorflow.keras.models import load_model
... import numpy as np
... from PIL import Image
... 
... # 학습된 모델을 불러오는 함수
... def load_trained_model(model_path='model/crack_detector.h5'):
...     """
...     모델을 불러오는 함수
...     model_path: 학습된 모델 파일 경로 (디폴트는 'model/crack_detector.h5')
...     """
...     model = load_model(model_path)
...     return model
... 
... # 이미지 예측을 위한 함수
... def predict_image(model, image):
...     """
...     이미지를 모델을 통해 예측하는 함수
...     model: 불러온 모델
...     image: 예측할 이미지
...     """
...     # 이미지 전처리
...     image = image.resize((224, 224)) # 예시: 모델 입력 사이즈에 맞게 리사이징
...     image = np.array(image) / 255.0 # 정규화
...     image = np.expand_dims(image, axis=0) # 배치 차원 추가
... 
...     # 예측 (오토인코더의 경우 reconstruction loss를 사용할 수 있음)
...     loss = model.predict(image)
...     return np.mean(np.abs(image - loss)) # 예시로 reconstruction loss 반환
... 이전 메일
