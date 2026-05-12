import streamlit as st
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cargar modelos y herramientas
@st.cache_resource
def cargar_modelos():
    with open('../models/modelo_xgboost.pkl', 'rb') as f:
        xgb = pickle.load(f)
    with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('../models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('../models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    modelo_nn = load_model('../models/modelo_red_neuronal.keras')
    return xgb, tfidf, le, tokenizer, modelo_nn

xgb, tfidf, le, tokenizer, modelo_nn = cargar_modelos()

# Función de limpieza
stop_words = set(stopwords.words('spanish'))
negaciones = {'no', 'nada', 'nunca', 'jamás', 'ningún',
              'ninguna', 'tampoco', 'sin', 'ni', 'pero'}
stop_words_filtradas = stop_words - negaciones

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    palabras = texto.split()
    palabras = [p for p in palabras if p not in stop_words_filtradas]
    return ' '.join(palabras)

# Interfaz
st.title('🛍️ Análisis de Sentimientos en Reseñas')
st.subheader('Amazon Reviews en Español')

texto = st.text_area('Escribe tu reseña aquí:', height=150)

if st.button('Analizar'):
    if texto.strip() == '':
        st.warning('Por favor escribe una reseña')
    else:
        texto_limpio = limpiar_texto(texto)

        # XGBoost
        texto_tfidf = tfidf.transform([texto_limpio])
        pred_xgb = xgb.predict(texto_tfidf)[0]
        sentimiento_xgb = le.inverse_transform([pred_xgb])[0]

        # Red Neuronal
        seq = tokenizer.texts_to_sequences([texto_limpio])
        pad = pad_sequences(seq, maxlen=100, truncating='post', padding='post')
        pred_nn = modelo_nn.predict(pad).argmax(axis=1)[0]
        sentimiento_nn = le.inverse_transform([pred_nn])[0]

        # Emojis
        emojis = {'positivo': '😊', 'neutro': '😐', 'negativo': '😞'}

        # Mostrar resultados
        col1, col2 = st.columns(2)
        with col1:
            st.metric('XGBoost', f"{emojis[sentimiento_xgb]} {sentimiento_xgb.upper()}")
        with col2:
            st.metric('Red Neuronal', f"{emojis[sentimiento_nn]} {sentimiento_nn.upper()}")