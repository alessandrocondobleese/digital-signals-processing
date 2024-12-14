import marimo

__generated_with = "0.9.32"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import scipy.signal as signal
    import numpy as np
    import altair as alt
    import pandas as pd
    import cv2
    return alt, cv2, np, pd, signal


@app.cell
def __(np, pd, signal):
    _b = [1 / 2]
    _a = [1, -1 / 2]

    _x = np.array([1, 0, 0, 0, 0, 0, 0])
    _y = signal.lfilter(_b, _a, _x)

    n = np.arange(len(_x))
    data = pd.DataFrame({"n": n, "x": _x, "y": _y})
    return data, n


@app.cell(hide_code=True)
def __(alt, data, n):
    line_x = (
        alt.Chart(data)
        .mark_line(point=True, color="blue", interpolate="step-after")
        .encode(
            x=alt.X("n:Q", title="n"),
            y=alt.Y("x:Q", title="Value"),
            tooltip=["n", "x"],
        )
    )

    line_y = (
        alt.Chart(data)
        .mark_line(point=True, color="red", interpolate="step-after")
        .encode(
            x=alt.X("n:Q", title="n"),
            y=alt.Y("y:Q", title="Value"),
            tooltip=["n", "y"],
        )
    )

    vs_chart = (
        alt.layer(line_x, line_y)
        .properties(title="x[n] contra y[n]")
        .configure_axis(grid=True)
        .configure_view(strokeWidth=0)
        .configure_axisX(tickCount=len(n))
        .configure_axisY(tickCount=2)
    )

    # mo.ui.altair_chart(vs_chart)
    return line_x, line_y, vs_chart


@app.cell
def __(mo):
    sampling_frequency_number = mo.ui.number(
        start=1, value=44100, full_width=True, label="Frecuencia de Muestreo"
    )
    sampling_frequency_number
    return (sampling_frequency_number,)


@app.cell
def __(sampling_frequency_number):
    sampling_frequency = sampling_frequency_number.value
    return (sampling_frequency,)


@app.cell
def __(mo):
    cut_frequency_number = mo.ui.number(
        start=1, value=500, full_width=True, label="Frecuencia de Corte (Hz)"
    )
    cut_frequency_number
    return (cut_frequency_number,)


@app.cell
def __(cut_frequency_number):
    cut_frequency = cut_frequency_number.value
    return (cut_frequency,)


@app.cell
def __(mo):
    stop_frequency_number = mo.ui.number(
        start=500, value=1000, full_width=True, label="Frecuencia de Rechazo (Hz)"
    )
    stop_frequency_number
    return (stop_frequency_number,)


@app.cell
def __(stop_frequency_number):
    stop_frequency = stop_frequency_number.value
    return (stop_frequency,)


@app.cell
def __(mo):
    passband_ripple_number = mo.ui.number(
        start=0,
        value=10,
        full_width=True,
        label="Ondulación de pasa de banda (dB)",
    )
    passband_ripple_number
    return (passband_ripple_number,)


@app.cell
def __(passband_ripple_number):
    passband_ripple = passband_ripple_number.value
    return (passband_ripple,)


@app.cell
def __(mo):
    stopband_atten_number = mo.ui.number(
        start=0,
        value=60,
        full_width=True,
        label="Atenuación de banda de rechazo (dB)",
    )
    stopband_atten_number
    return (stopband_atten_number,)


@app.cell
def __(stopband_atten_number):
    stopband_atten = stopband_atten_number.value
    return (stopband_atten,)


@app.cell
def __(mo):
    btype_dropdown = mo.ui.dropdown(
        ["lowpass", "highpass", "bandpass", "bandstop"],
        value="lowpass",
        full_width=True,
        label="Tipo de filtro",
    )
    btype_dropdown
    return (btype_dropdown,)


@app.cell
def __(btype_dropdown):
    btype = btype_dropdown.value
    return (btype,)


@app.cell
def __(
    btype,
    cut_frequency,
    passband_ripple,
    sampling_frequency,
    signal,
    stop_frequency,
    stopband_atten,
):
    butterworth_order, butterworth_cut = signal.buttord(
        cut_frequency,
        stop_frequency,
        passband_ripple,
        stopband_atten,
        fs=sampling_frequency,
    )

    cheb_order, cheb_cut = signal.cheb1ord(
        cut_frequency,
        stop_frequency,
        passband_ripple,
        stopband_atten,
        fs=sampling_frequency,
    )

    cheb_order2, cheb_cut2 = signal.cheb2ord(
        cut_frequency,
        stop_frequency,
        passband_ripple,
        stopband_atten,
        fs=sampling_frequency,
    )

    ellipord_order, ellipord_cut = signal.ellipord(
        cut_frequency,
        stop_frequency,
        passband_ripple,
        stopband_atten,
        fs=sampling_frequency,
    )


    FILTER = {
        "Butterworth": signal.butter(
            N=butterworth_order,
            Wn=butterworth_cut,
            fs=sampling_frequency,
            btype=btype,
        ),
        "Chebyshev I": signal.cheby1(
            N=cheb_order,
            Wn=cheb_cut,
            rp=passband_ripple,
            fs=sampling_frequency,
            btype=btype,
        ),
        "Chebyshev II": signal.cheby2(
            N=cheb_order2,
            Wn=cheb_cut2,
            rs=stopband_atten,
            fs=sampling_frequency,
            btype=btype,
        ),
        "Elliptic": signal.ellip(
            N=ellipord_order,
            Wn=ellipord_cut,
            rp=passband_ripple,
            rs=stopband_atten,
            fs=sampling_frequency,
            btype=btype,
        ),
    }
    return (
        FILTER,
        butterworth_cut,
        butterworth_order,
        cheb_cut,
        cheb_cut2,
        cheb_order,
        cheb_order2,
        ellipord_cut,
        ellipord_order,
    )


@app.cell
def __(FILTER, np, pd, sampling_frequency, signal):
    impulse_responses = []
    frequencies = []

    for name in FILTER.keys():
        b, a = FILTER[name]

        x = np.zeros(1000)
        x[249] = 1

        impulse_response = signal.filtfilt(b, a, x)
        w, h = signal.freqz(b, a)

        w = w * sampling_frequency / (2 * np.pi)

        impulse_responses.append(
            pd.DataFrame(
                {
                    "x": impulse_response,
                    "n": np.arange(len(impulse_response)),
                    "type": name,
                }
            )
        )

        frequencies.append(
            pd.DataFrame(
                {"frequency": w, "magnitude": 20 * np.log10(abs(h)), "type": name}
            )
        )
    return (
        a,
        b,
        frequencies,
        h,
        impulse_response,
        impulse_responses,
        name,
        w,
        x,
    )


@app.cell
def __(frequencies, impulse_responses, pd):
    frequencies_data = pd.concat(frequencies)
    impulse_df = pd.concat(impulse_responses)
    return frequencies_data, impulse_df


@app.cell
def __(alt, frequencies_data, impulse_df, mo):
    mo.vstack(
        [
            mo.ui.altair_chart(
                alt.Chart(frequencies_data)
                .mark_line()
                .encode(
                    x="frequency",
                    y="magnitude",
                    color=alt.Color("type:N"),
                )
            ),
            mo.ui.altair_chart(
                alt.Chart(impulse_df)
                .mark_line(interpolate="step-after")
                .encode(
                    x="n",
                    y="x",
                    color=alt.Color("type:N"),
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
