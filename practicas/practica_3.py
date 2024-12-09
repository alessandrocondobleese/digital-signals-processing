import marimo

__generated_with = "0.9.21"
app = marimo.App(width="medium")


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
            .mark_line()
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

        positive_frequencies = signal_fourier_frequencies_spectrum > 0
        signal_fourier_frequencies_spectrum = signal_fourier_frequencies_spectrum[
            positive_frequencies
        ]

        sample_frequencies = np.fft.fftfreq(time.size)
        sample_frequencies = sample_frequencies[positive_frequencies]

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
def __(np):
    def dft(x):
        """The Discrete Fourier Transform

        Parameters
        ----------
        x : np.ndarray
            The input signal

        Returns
        -------
        X : np.ndarray, same shape as x
            The DFT sequence: X[m] corresponds to analysis frequency index m
        """

        # Get the number of samples = number of frequencies
        N = len(x)

        # Allocate the output array
        X = np.zeros(N, dtype=complex)

        # For each analysis frequency
        for m in range(N):
            # For each sample
            for n in range(N):
                # Compare to cos and -sin at this frequency
                X[m] = X[m] + x[n] * (
                    np.cos(2 * np.pi * m / N * n)
                    - 1j * np.sin(2 * np.pi * m / N * n)
                )
        # Return the DFT array
        return X
    return (dft,)


@app.cell
def __(np, signal):
    impulse_time = np.linspace(0, 1, 500)
    impulse = signal.unit_impulse(impulse_time.size)
    return impulse, impulse_time


@app.cell
def __(
    get_signal_chart,
    get_signal_frequencies_chart,
    impulse,
    impulse_time,
    mo,
):
    mo.ui.altair_chart(
        get_signal_chart(impulse, impulse_time)
        & get_signal_frequencies_chart(impulse, impulse_time)
    )
    return


@app.cell
def __(np, signal):
    sawtooth_time = np.linspace(0, 1, 500)
    sawtooth = signal.sawtooth(2 * np.pi * 5 * sawtooth_time)
    return sawtooth, sawtooth_time


@app.cell
def __(
    get_signal_chart,
    get_signal_frequencies_chart,
    mo,
    sawtooth,
    sawtooth_time,
):
    mo.ui.altair_chart(
        get_signal_chart(sawtooth, sawtooth_time)
        & get_signal_frequencies_chart(sawtooth, sawtooth_time)
    )
    return


@app.cell
def __(np, signal):
    square_time = np.linspace(0, 1, 500)
    square = signal.square(2 * np.pi * 5 * square_time)
    return square, square_time


@app.cell
def __(
    get_signal_chart,
    get_signal_frequencies_chart,
    mo,
    square,
    square_time,
):
    mo.ui.altair_chart(
        get_signal_chart(square, square_time)
        & get_signal_frequencies_chart(square, square_time)
    )
    return


@app.cell
def __(np, signal):
    chirp_time = np.linspace(0, 1, 500)
    chirp = signal.chirp(chirp_time, f0=6, f1=1, t1=1, method="hyperbolic")
    return chirp, chirp_time


@app.cell
def __(
    chirp,
    chirp_time,
    get_signal_chart,
    get_signal_frequencies_chart,
    mo,
):
    mo.ui.altair_chart(
        get_signal_chart(chirp, chirp_time)
        & get_signal_frequencies_chart(chirp, chirp_time)
    )
    return


@app.cell
def __(np, signal):
    gausspulse_time = np.linspace(0, 1, 500, endpoint=False)
    gausspulse = signal.gausspulse(gausspulse_time, fc=5)
    return gausspulse, gausspulse_time


@app.cell
def __(
    gausspulse,
    gausspulse_time,
    get_signal_chart,
    get_signal_frequencies_chart,
    mo,
):
    mo.ui.altair_chart(
        get_signal_chart(gausspulse, gausspulse_time)
        & get_signal_frequencies_chart(gausspulse, gausspulse_time)
    )
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
