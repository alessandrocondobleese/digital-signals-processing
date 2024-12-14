import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pydub import AudioSegment
import pygame
import os
from moviepy import *


def fft(x):

    N = len(x)

    if N == 1:
        return x

    else:
        X_even = fft(x[0::2])
        X_odd = fft(x[1::2])

        X = np.zeros(N, dtype=complex)

        for m in range(N):
            m_alias = m % (N // 2)
            X[m] = X_even[m_alias] + np.exp(-2j * np.pi * m / N) * X_odd[m_alias]

        return X


def animate_frequencies(
    song_filename, window_size=2048, overlap=1024, sample_rate=44100
):
    song = AudioSegment.from_mp3(song_filename)
    song.export("temp.wav", format="wav")

    song_samples = np.array(song.get_array_of_samples())

    pygame.mixer.init(frequency=sample_rate)
    pygame.mixer.music.load("temp.wav")

    figure, figure_axis = plt.subplots(figsize=(12, 6))

    figure.patch.set_facecolor("#1c1c1c")

    figure_axis.set_facecolor("#2c2c2c")
    figure_axis.grid(color="#444444", linestyle="--", linewidth=0.5, alpha=0.5)
    figure_axis.set_xlim(0, sample_rate // 2)
    figure_axis.set_ylim(0, 1)
    figure_axis.set_title("Espectro de Frecuencias", fontsize=16, color="white")
    figure_axis.set_xlabel("Frecuencia (Hz)", fontsize=12, color="white")
    figure_axis.set_ylabel("Magnitud", fontsize=12, color="white")
    figure_axis.tick_params(colors="white", labelsize=10)

    bars_number = window_size // 2
    bars_frequencies = np.linspace(0, sample_rate // 2, bars_number) 
    bars = figure_axis.bar(bars_frequencies, np.zeros(bars_number), color="#ff5733",  width=10, lw=10, alpha=0.8)

    #pygame.mixer.music.play()

    stop_animation = {"stop": False}

    def handle_close(event):
        stop_animation["stop"] = True
        #pygame.mixer.music.stop()

    figure.canvas.mpl_connect("close_event", handle_close)

    def update(frame):
        if stop_animation["stop"]:
            return bars  # Detener la animación si se cerró la ventana

        start = frame * (window_size - overlap)
        if start + window_size >= len(song_samples):
            return bars  # Detener la animación si llegamos al final

        window = song_samples[start : start + window_size]

        spectrum = fft(window)
        freqs = np.fft.fftfreq(len(window), d=1 / sample_rate)

        pos_freqs = freqs[: len(freqs) // 2]
        pos_spectrum = np.abs(spectrum)[: len(spectrum) // 2]

        pos_spectrum /= np.max(pos_spectrum) if np.max(pos_spectrum) > 0 else 1

        for bar, height in zip(bars, pos_spectrum):
            bar.set_height(height)

        max_spectrum = np.max(pos_spectrum)
        if max_spectrum > 0:
            pos_spectrum = (pos_spectrum / max_spectrum) * 100  # Normalización a un rango de 0 a 100

        return (bars,)


    # Configurar la animación
    frames = (len(song_samples) - window_size) // (window_size - overlap)
    _animation = FuncAnimation(
        figure,
        update,
        frames=60 * 15,
        interval=1000 * (window_size - overlap) / sample_rate,
        blit=False,
    )

    _animation.save("temp_video.mp4", writer="ffmpeg", fps=60)

    #pygame.mixer.quit()

    video = VideoFileClip("temp_video.mp4")
    audio = AudioFileClip("temp.wav")

    audio = CompositeAudioClip([audio.subclipped(0, 15)])

    video.audio = audio
    video.write_videofile("song-video.mp4", codec="libx264", audio_codec="aac")

    os.remove("temp.wav")


if __name__ == "__main__":
    # Ruta del archivo MP3
    mp3_file = "practicas/resources/audio/song.mp3"  # Cambia esta ruta a la ubicación de tu archivo MP3

    # Ejecutar la animación
    animate_frequencies(mp3_file)
