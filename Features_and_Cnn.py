import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import models
from keras import layers
import warnings
warnings.filterwarnings('ignore')

#Cria o cabeçalho da lista CSV.
cab = 'nome_arquivo zero_cr chroma_cqt chroma_cens tonnetz chroma_stf rmse spec_centroid spec_bandwidth spec_contrast spec_rolloff'
for i in range(1, 21):
    cab += f' mfcc{i}'
cab += ' label'
cab = cab.split()

#Cria um CSV, localiza todas pastas e arquivos no diretório, extrai as features das músicas,
#concatena tudo dentro do CSV usando .append dentro do respetivo cabeçalho.
file = open('D:/Final/Dados em CSV/data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(cab)

generos = 'blues classical country disco hiphop jazz metal pop reggae rock'.split() #Divide.
for g in generos: #Roda o for em todas pastas.
  
    for nome_arquivo in os.listdir(f'D:/Music Database/GTZAN Music Database/{g}'): #Roda arquivo por arquivo.
     
        musica = f'D:/Music Database/GTZAN Music Database/{g}/{nome_arquivo}'#Diretório da música.
     
        x, sr = librosa.load(musica, mono=True, duration=30) #Ler a música.
      
        #Extrai todas features abaixo, usando librosa.
        zcr = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
      
        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7*12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1
      
        chroma_cqt = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)

        chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)

        tonnetz = librosa.feature.tonnetz(librosa.effects.harmonic(x), sr=sr)

        del cqt
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
        del x
       
        chroma_stft = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)

        rmse = librosa.feature.rmse(S=stft)

        spec_cent = librosa.feature.spectral_centroid(S=stft)

        spec_bw = librosa.feature.spectral_bandwidth(S=stft)

        spec_contrast = librosa.feature.spectral_contrast(S=stft, n_bands=6)

        spec_rolloff = librosa.feature.spectral_rolloff(S=stft)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        del stft

        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)

        to_append = f'{nome_arquivo} {np.mean(zcr)} {np.mean(chroma_cqt)} {np.mean(chroma_cens)} {np.mean(tonnetz)} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(spec_contrast)} {np.mean(spec_rolloff)}'      
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('D:/Final/Dados em CSV/data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())


#Lê o arquivo CSV usando o pandas.
todas_features = pd.read_csv('D:/Final/Dados em CSV/data.csv')
todas_features.head()

#Exclui a coluna com os nomes.
todas_features = todas_features.drop(['nome_arquivo'],axis=1)

#Cria uma ligação entre os generos e um valor int, cada int representa um genero.
genre_list = todas_features.iloc[:, -1] #Cria um int para cada label
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list) #Separa por labens distintas

# 0 - Blues  1 - Classical  2 - Country  3 - Disco  4 - Hiphop  5 - Jazz  6 - Metal  7 - Pop  8 - Reggae # 9 - Rock

#genre_list.head()
#genre_list[4]      #De 1000 músicas a 0004 é blues.
#y[5]               #Valor 0 = Blues.

#X é calculado removendo o mean e divindo pela variancia.
scaler = StandardScaler()
X = scaler.fit_transform(np.array(todas_features.iloc[:, :-1], dtype = float))    

#Dividi os dados entre treino e teste em 800/200.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2, random_state=101)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#len(y_train)
#len(y_test)
#X_train


#Cria o modelo da CNN, com 256 neurônios.
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


#Definições de como o modelo deverá ser treinado.
model.compile(optimizer='adam', #Algoritimo Adam para otimizar a rede neural.
              loss='sparse_categorical_crossentropy', #Função que utiliza as baixas.
              metrics=['accuracy']) #Precisão escolhida como métrica

#Treinando o modelo.
history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=64)


#Calculando a precisão e perca que o modelo conseguiu categorizando da musica baseado nas features extraidas.
test_loss, test_acc = model.evaluate(X_test,y_test)
print('Precisão: ', test_acc)
print('Perdido: ', test_loss)

#Separando 200 amostras para o teste de validação.
x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]

#Cria o modelo da CNN, com 512 neurônios.
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#Definições de como o modelo deverá ser treinado.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Treinando o modelo.
model.fit(partial_x_train,
          partial_y_train,
          epochs=30,
          batch_size=256,
          validation_data=(x_val, y_val))

resultado = model.evaluate(X_test, y_test)
resultado