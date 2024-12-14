import marimo

__generated_with = "0.9.32"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import numpy as np
    import pandas as pd
    import altair as alt
    return alt, np, pd


@app.cell
def __():
    import scipy.signal as signal
    return (signal,)


@app.cell
def __(alt, pd):
    def get_signal_chart(signal, time):
        signal_dataframe = pd.DataFrame({"time": time, "amplitude": signal})
        signal_chart = (
            alt.Chart(signal_dataframe)
            .mark_line(size=3)
            .encode(
                x=alt.X("time:O", title="Time", axis=alt.Axis(labels=False)),
                y=alt.Y("amplitude:Q", title="Amplitude"),
            )
        )

        return signal_chart
    return (get_signal_chart,)


@app.cell
def __(alt, dft, np, pd):
    def get_signal_frequencies_chart(signal, time):
        signal_fourier_frequencies_spectrum = dft(signal)
        signal_fourier_frequencies_spectrum = np.abs(
            signal_fourier_frequencies_spectrum
        )

        sample_frequencies = np.fft.fftfreq(time.size, d=(time[1] - time[0]))

        positive_indices = sample_frequencies >= 0
        sample_frequencies = sample_frequencies[positive_indices]
        signal_fourier_frequencies_spectrum = signal_fourier_frequencies_spectrum[
            positive_indices
        ]

        signal_frequencies_dataframe = pd.DataFrame(
            {
                "frequency": sample_frequencies,
                "magnitude": signal_fourier_frequencies_spectrum,
            }
        )

        signal_frequencies_chart = (
            alt.Chart(signal_frequencies_dataframe)
            .mark_bar(color="#ff7f0e")
            .encode(
                x=alt.X("frequency:Q", title="Frecuencia"),
                y=alt.Y(
                    "magnitude:Q",
                    title="Magnitud del espectro de Fourier",
                    scale=alt.Scale(
                        domain=[0, max(signal_fourier_frequencies_spectrum) * 1.2]
                    ),
                ),
            )
        )

        return signal_frequencies_chart
    return (get_signal_frequencies_chart,)


@app.cell
def __(mo, np):
    def dft(x):
        N = len(x)

        X = np.zeros(N, dtype=complex)

        for m in range(N):
            for n in range(N):
                X[m] = X[m] + x[n] * (
                    np.cos(2 * np.pi * m / N * n)
                    - 1j * np.sin(2 * np.pi * m / N * n)
                )

        return X

    mo.show_code()
    return (dft,)


@app.cell
def __(np, signal):
    impulse_time = np.linspace(0, 1, 50)
    impulse = signal.unit_impulse(impulse_time.size)
    return impulse, impulse_time


@app.cell
def __(np, signal):
    sawtooth_time = np.linspace(0, 1, 500)
    sawtooth = signal.sawtooth(2 * np.pi * 5 * sawtooth_time)
    return sawtooth, sawtooth_time


@app.cell
def __(np, signal):
    square_time = np.linspace(0, 1, 500)
    square = signal.square(2 * np.pi * 5 * square_time)
    return square, square_time


@app.cell
def __(np, signal):
    chirp_time = np.linspace(0, 1, 500)
    chirp = signal.chirp(chirp_time, f0=6, f1=1, t1=1, method="hyperbolic")
    return chirp, chirp_time


@app.cell
def __(np, signal):
    gausspulse_time = np.linspace(0, 1, 500, endpoint=False)
    gausspulse = signal.gausspulse(gausspulse_time, fc=5)
    return gausspulse, gausspulse_time


@app.cell
def __(mo):
    sin_amplitude_range_slider = mo.ui.slider(
        label="Amplitud",
        start=0,
        value=1,
        stop=10,
        full_width=True,
        show_value=True,
    )
    return (sin_amplitude_range_slider,)


@app.cell
def __(mo):
    sin_frequency_range_slider = mo.ui.slider(
        label="Frecuencia", start=1, stop=10, full_width=True, show_value=True
    )
    return (sin_frequency_range_slider,)


@app.cell
def __(mo):
    sin_phase_range_slider = mo.ui.slider(
        label="Fase", start=1, stop=10, full_width=True, show_value=True
    )
    return (sin_phase_range_slider,)


@app.cell
def __(mo):
    sin_sampling_frequency_number = mo.ui.number(
        label="Frecuencia de muestreo",
        start=2,
        full_width=True,
    )
    return (sin_sampling_frequency_number,)


@app.cell
def __(
    mo,
    np,
    sin_amplitude_range_slider,
    sin_frequency_range_slider,
    sin_phase_range_slider,
    sin_sampling_frequency_number,
):
    sin_sinusoide_time = np.linspace(
        0, 1, int(sin_sampling_frequency_number.value * 1), endpoint=False
    )
    sin_sinusoide = sin_amplitude_range_slider.value * np.sin(
        2 * np.pi * sin_frequency_range_slider.value * sin_sinusoide_time
        + np.pi / sin_phase_range_slider.value
    )

    mo.show_code()
    return sin_sinusoide, sin_sinusoide_time


@app.cell
def __(mo):
    cosine_amplitude_range_slider = mo.ui.slider(
        label="Amplitud",
        start=0,
        value=1,
        stop=10,
        full_width=True,
        show_value=True,
    )

    cosine_frequency_range_slider = mo.ui.slider(
        label="Frecuencia", start=1, stop=10, full_width=True, show_value=True
    )

    cosine_phase_range_slider = mo.ui.slider(
        label="Fase", start=1, stop=10, full_width=True, show_value=True
    )
    return (
        cosine_amplitude_range_slider,
        cosine_frequency_range_slider,
        cosine_phase_range_slider,
    )


@app.cell
def __(
    cosine_amplitude_range_slider,
    cosine_frequency_range_slider,
    cosine_phase_range_slider,
    mo,
    np,
    sin_sampling_frequency_number,
):
    cosine_sinusoide_time = np.linspace(
        0, 1, int(sin_sampling_frequency_number.value * 1), endpoint=False
    )
    cosine_sinusoide = cosine_amplitude_range_slider.value * np.cos(
        2 * np.pi * cosine_frequency_range_slider.value * cosine_sinusoide_time
        + np.pi / cosine_phase_range_slider.value
    )

    mo.show_code()
    return cosine_sinusoide, cosine_sinusoide_time


@app.cell
def __(
    cosine_amplitude_range_slider,
    cosine_frequency_range_slider,
    cosine_phase_range_slider,
    cosine_sinusoide,
    get_signal_chart,
    get_signal_frequencies_chart,
    mo,
    sin_amplitude_range_slider,
    sin_frequency_range_slider,
    sin_phase_range_slider,
    sin_sampling_frequency_number,
    sin_sinusoide,
    sin_sinusoide_time,
):
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md(
                        f"""
                        ## Componente seno
                        {sin_amplitude_range_slider}
                        {sin_frequency_range_slider}
                        {sin_phase_range_slider}
                        {sin_sampling_frequency_number}
                        """
                    ).callout(),
                    mo.md(
                        f"""
                        ## Componente coseno
                        {cosine_amplitude_range_slider}
                        {cosine_frequency_range_slider}
                        {cosine_phase_range_slider}
                        {sin_sampling_frequency_number}
                        """
                    ).callout(),
                ],
                widths="equal",
            ),
            mo.ui.altair_chart(
                get_signal_chart(
                    sin_sinusoide + cosine_sinusoide, sin_sinusoide_time
                )
            ),
            mo.ui.altair_chart(
                get_signal_frequencies_chart(
                    sin_sinusoide + cosine_sinusoide, sin_sinusoide_time
                )
            ),
        ]
    )
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
