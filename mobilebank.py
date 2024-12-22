import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import librosa
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2Tokenizer
#pip install -U scikit-learn
#türkçe ses eğitim
tokenizer=Wav2Vec2Tokenizer.from_pretrained('m3hrdadfi/wav2vec2-large-xlsr-turkish')
model=Wav2Vec2ForCTC.from_pretrained('m3hrdadfi/wav2vec2-large-xlsr-turkish')


audio_value = st.audio_input("Record a voice message")

if audio_value:
    #st.audio(audio_value)

    x, sr = librosa.load(audio_value, sr=16000)

    input_values = tokenizer(x, return_tensors="pt").input_values
    logits = model(input_values).logits
    pretrained = torch.argmax(logits, dim=-1)
    sonuc = tokenizer.decode(pretrained[0])
    mesaj=sonuc
    df=pd.read_csv('bankv2.csv')
    df=df[['sorgu','label']]
    cv=CountVectorizer(max_features=250)
    rf=RandomForestClassifier()
    x=cv.fit_transform(df['sorgu']).toarray()
    y=df['label']
    model=rf.fit(x,y)
    mesajvektor=cv.transform([mesaj]).toarray()
    sonuc=model.predict(mesajvektor)
    st.write(sonuc[0])