import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.style as ms 
import librosa
import librosa.display
import IPython.display as aqui
import os
import matplotlib
import pylab

# Quando não coloca onde salvar
# C:\Users\Heaven\AppData\Local\Programs\Microsoft VS Code

#Ler o audio
y, sr = librosa.load('D:/Music Database/GTZAN Music Database/pop/pop.00020.wav', sr=44100) #SR = Sample Rate.
aqui.Audio(y, rate=sr) #Renderiza um audio player.

#Waveform do audio
plt.figure(figsize=(14, 5)) #Parâmetros da figura.
librosa.display.waveplot(y, sr=sr) #Renderiza a figura do wav.

#Mel-Spectrogram 
y, sr = librosa.load('D:/Music Database/GTZAN Music Database/pop/pop.00020.wav', sr=44100) #SR = Sample Rate.
X = librosa.stft(y) #Transformada de fourier de curto tempo.
Xdb = librosa.amplitude_to_db(abs(X)) #Converte para decibéis.
plt.figure(figsize=(14, 5)) #Parâmetros da figura.
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') #Renderiza a figura.
plt.colorbar() #Barra de cores do lado.

# Criando um .wav de 5 segundos com som de TUMMMMMMM
sr = 22050 #Sample rate.
T = 5.0    #Segundos.
t = np.linspace(0, T, int(T*sr), endpoint=False) #time variable.
x = 0.5*np.sin(2*np.pi*220*t) #pure sine wave at 220 Hz.
aqui.Audio(x, rate=sr) #load a NumPy array.
librosa.output.write_wav('D:/Music Database/GTZAN Music Database/pop/', x, sr)

#Mel-scaled power spectrogram
S = librosa.feature.melspectrogram(y, sr=44100) #Obtem um Mel-spectrogram.
Sdb = librosa.power_to_db(S, ref=np.max) #Converte para decibéis.
plt.figure(figsize=(14, 5))
librosa.display.specshow(Sdb, sr=44100, x_axis='time', y_axis='hz')
plt.colorbar()

#Chromagram // Intensidade de cada uma das 12 notes
y, sr = librosa.load('D:/Music Database/GTZAN Music Database/pop/pop.00020.wav', sr=44100) 
chromagram = librosa.feature.chroma_stft(y=y, sr=44100)
plt.figure(figsize=(14, 5))
librosa.display.specshow(chromagram, sr=44100, x_axis='time', y_axis='chroma')

#Estimativa do tempo Estático
y, sr = librosa.load('D:/Music Database/GTZAN Music Database/pop/pop.00020.wav', sr=44100) 
onset_env = librosa.onset.onset_strength(y, sr=44100)
tempo = librosa.beat.tempo(onset_env, sr=44100)
tempo

#Estimativa de tempo junto com grafico de autocorrelação do onset
y, sr = librosa.load('D:/Music Database/GTZAN Music Database/pop/pop.00020.wav', sr=44100) 
onset_env = librosa.onset.onset_strength(y, sr=44100)
tempo = np.asscalar(tempo) # Convert to scalar
# Compute 2-second windowed autocorrelation
hop_length = 512
ac = librosa.autocorrelate(onset_env, 2 * sr // hop_length)
freqs = librosa.tempo_frequencies(len(ac), sr=sr,hop_length=hop_length)
# Plot on a BPM axis.  We skip the first (0-lag) bin.
plt.figure(figsize=(14,5))
plt.semilogx(freqs[1:], librosa.util.normalize(ac)[1:],label='Onset autocorrelation', basex=2)
plt.axvline(tempo, 0, 1, color='r', alpha=0.75, linestyle='--',label='Tempo: {:.2f} BPM'.format(tempo))
plt.xlabel('Tempo (BPM)')
plt.grid()
plt.title('Static tempo estimation')
plt.legend(frameon=True)
plt.axis('tight')

#Estimativa do tempo Dinâmico
y, sr = librosa.load('D:/Music Database/GTZAN Music Database/pop/pop.00020.wav', sr=44100) 
onset_env = librosa.onset.onset_strength(y, sr=44100)
dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=44100, aggregate=None)
dtempo

#Estimativa de tempo dinâmico com tempogram
y, sr = librosa.load('D:/Music Database/GTZAN Music Database/pop/pop.00020.wav', sr=44100) 
onset_env = librosa.onset.onset_strength(y, sr=44100)
plt.figure(figsize=(14,5))
tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=44100,hop_length=512)
librosa.display.specshow(tg, x_axis='time', y_axis='tempo')
plt.plot(librosa.frames_to_time(np.arange(len(dtempo))), dtempo,color='w', linewidth=1.5, label='Tempo estimate')
plt.title('Dynamic tempo estimation')

#Beats
y, sr = librosa.load('D:/Music Database/GTZAN Music Database/pop/pop.00020.wav', sr=44100) 
y_harmonic, y_percussive = librosa.effects.hpss(y) #Separa harmonico do percursivo
tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=44100)
beats[:20]

#Salva alguma imagem
Salvar_imagem = 'D:/Music Database/GTZAN Music Database/pop/pop.00020.jpg'
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=44100) #Escolher o espectrograma.
plt.margins(0)  # as suggested by Eran W
pylab.axis('off') # no axis
pylab.savefig(Salvar_imagem, transparent = True, bbox_inches = 'tight', pad_inches = 0)
pylab.close()