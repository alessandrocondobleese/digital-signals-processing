import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pydub import AudioSegment
from scipy.fft import fft
import pygame
import time


def animate_frequencies(mp3_file, window_size=2048, overlap=1024, sample_rate=44100):
    # Cargar el archivo MP3
    song = AudioSegment.from_mp3(mp3_file)
    samples = np.array(song.get_array_of_samples())

    # Configurar pygame para reproducir el audio
    pygame.mixer.init(frequency=sample_rate)
    song.export("temp.wav", format="wav")  # Convertir el MP3 a WAV para pygame
    pygame.mixer.music.load("temp.wav")

    # Configurar el gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    (line,) = ax.plot([], [], color="#ff5733", lw=2.5, alpha=0.8)
    ax.set_facecolor("#2c2c2c")  # Fondo oscuro
    fig.patch.set_facecolor("#1c1c1c")  # Fondo de toda la figura
    ax.grid(color="#444444", linestyle="--", linewidth=0.5, alpha=0.5)

    # Configuración inicial del gráfico
    ax.set_xlim(0, sample_rate // 2)
    ax.set_ylim(0, 1)
    ax.set_title("Now Playing: Espectro de Frecuencias 🎵", fontsize=16, color="white")
    ax.set_xlabel("Frecuencia (Hz)", fontsize=12, color="white")
    ax.set_ylabel("Magnitud", fontsize=12, color="white")
    ax.tick_params(colors="white", labelsize=10)

    # Iniciar la música
    pygame.mixer.music.play()

    # Variable para detener la animación si se cierra la ventana
    stop_animation = {"stop": False}

    # Manejo del evento de cierre de ventana
    def handle_close(event):
        stop_animation["stop"] = True
        pygame.mixer.music.stop()  # Detener la música si se cierra la ventana

    # Conectar el evento de cierre de ventana
    fig.canvas.mpl_connect("close_event", handle_close)

    # Función para actualizar el gráfico
    def update(frame):
        if stop_animation["stop"]:
            return (line,)  # Detener la animación si se cerró la ventana

        start = frame * (window_size - overlap)
        if start + window_size >= len(samples):
            return (line,)  # Detener la animación si llegamos al final

        # Tomar una ventana de datos
        window = samples[start : start + window_size]

        # Calcular la FFT
        spectrum = fft(window)
        freqs = np.fft.fftfreq(len(window), d=1 / sample_rate)

        # Filtrar las frecuencias positivas
        pos_freqs = freqs[: len(freqs) // 2]
        pos_spectrum = np.abs(spectrum)[: len(spectrum) // 2]

        # Normalizar el espectro
        pos_spectrum /= np.max(pos_spectrum) if np.max(pos_spectrum) > 0 else 1

        # Actualizar la línea del gráfico
        line.set_data(pos_freqs, pos_spectrum)
        ax.set_ylim(0, np.max(pos_spectrum) * 1.2)
        return (line,)

    # Configurar la animación
    frames = (len(samples) - window_size) // (window_size - overlap)
    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 * (window_size - overlap) / sample_rate,
        blit=True,
    )

    # Mostrar el gráfico
    plt.show()

    # Limpiar archivos temporales
    pygame.mixer.quit()
    import os

    os.remove("temp.wav")


if __name__ == "__main__":
    # Ruta del archivo MP3
    mp3_file = "practicas/song.mp3"  # Cambia esta ruta a la ubicación de tu archivo MP3

    # Ejecutar la animación
    animate_frequencies(mp3_file)
