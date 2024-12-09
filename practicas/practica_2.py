import marimo

__generated_with = "0.9.21"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    sampling_frequency_number = mo.ui.number(
        start=10,
        step=1,
    )
    return (sampling_frequency_number,)


@app.cell
def __(sampling_frequency_number):
    sampling_frequency = sampling_frequency_number.value
    return (sampling_frequency,)


@app.cell
def __(numpy, sampling_frequency):
    discrete_time = numpy.arange(0, 10, 1 / sampling_frequency)
    discrete_x = numpy.sin(2 * numpy.pi * discrete_time)
    return discrete_time, discrete_x


@app.cell
def __(sampling_frequency_number):
    sampling_frequency_number
    return


@app.cell
def __(numpy):
    def delay(x, K):
        return numpy.pad(x, (K, 0), mode="constant")
    return (delay,)


@app.cell
def __():
    def gain(x, alpha):
        return alpha * x
    return (gain,)


@app.cell
def __(numpy):
    def mix(x, h):
        max_len = max(len(x), len(h))
        x = numpy.pad(x, (0, max_len - len(x)), mode="constant")
        h = numpy.pad(h, (0, max_len - len(h)), mode="constant")
        return x + h
    return (mix,)


@app.cell
def __(altair, delay, discrete_x, gain, mix, mo, pandas):
    _dataframe_x = pandas.DataFrame(
        {
            "Time": range(len(discrete_x)),
            "Amplitude": discrete_x,
            "Type": "Original",
        }
    )

    _dataframe_delayed = pandas.DataFrame(
        {
            "Time": range(len(delay(discrete_x, 50))),
            "Amplitude": delay(discrete_x, 50),
            "Type": "Delay (50)",
        }
    )

    _dataframe_gained = pandas.DataFrame(
        {
            "Time": range(len(gain(discrete_x, 2 / 3))),
            "Amplitude": gain(discrete_x, 2 / 3),
            "Type": "Gain (15)",
        }
    )

    _dataframe_mixed = pandas.DataFrame(
        {
            "Time": range(len(mix(discrete_x, delay(discrete_x, 30)))),
            "Amplitude": mix(discrete_x, delay(discrete_x, 30)),
            "Type": "Mix",
        }
    )

    _df_all = pandas.concat(
        [_dataframe_x, _dataframe_delayed, _dataframe_gained, _dataframe_mixed]
    )

    _chart = (
        altair.Chart(_df_all)
        .mark_line(point=True)
        .encode(x="Time", y="Amplitude", color="Type")
    )

    mo.ui.altair_chart(_chart)
    return


@app.cell
def __(numpy):
    def convolution(x, h):
        n = len(x) + len(h) - 1

        y = numpy.zeros(n)

        for i in range(len(x)):
            for j in range(len(h)):
                y[i + j] += x[i] * h[j]

        return y
    return (convolution,)


@app.cell
def __(numpy):
    delta = numpy.zeros(51)
    delta[-1] = 1
    return (delta,)


@app.cell
def __():
    G = [2 / 3]
    return (G,)


@app.cell
def __(G, altair, convolution, delta, discrete_x, mo, pandas):
    _dataframe_x = pandas.DataFrame(
        {
            "Time": range(len(discrete_x)),
            "Amplitude": discrete_x,
            "Type": "Original",
        }
    )

    _dataframe_delayed = pandas.DataFrame(
        {
            "Time": range(len(convolution(discrete_x, delta))),
            "Amplitude": convolution(discrete_x, delta),
            "Type": "Delay",
        }
    )

    _dataframe_gained = pandas.DataFrame(
        {
            "Time": range(len(convolution(discrete_x, G))),
            "Amplitude": convolution(discrete_x, G),
            "Type": "Gain",
        }
    )


    _df_all = pandas.concat([_dataframe_x, _dataframe_delayed, _dataframe_gained])

    _chart = (
        altair.Chart(_df_all)
        .mark_line(point=True, size=2)
        .encode(
            x="Time",
            y="Amplitude",
            color="Type",
        )
    )

    mo.ui.altair_chart(_chart)
    return


@app.cell
def __(mo):
    import altair as alt
    import pandas as pd
    import numpy as np

    # Crear señales de ejemplo
    time = np.arange(0, 100, 1)  # Tiempo de 0 a 100 con paso 1

    # Señal escalón
    original_signal_step = np.where(time > 50, 1, 0)  # Escalón en 50

    # Señal en diente de sierra
    original_signal_saw = (
        np.mod(time, 20) / 20
    )  # Diente de sierra con periodo de 20

    # Señal resultante (combinación de escalón y diente de sierra)
    combined_signal = (
        original_signal_step + original_signal_saw
    )  # Suma de ambas señales

    # Crear DataFrames
    _dataframe_step = pd.DataFrame(
        {
            "Time": time,
            "Amplitude": original_signal_step,
            "Type": "Step",
        }
    )

    _dataframe_saw = pd.DataFrame(
        {
            "Time": time,
            "Amplitude": original_signal_saw,
            "Type": "Saw",
        }
    )

    _dataframe_combined = pd.DataFrame(
        {
            "Time": time,
            "Amplitude": combined_signal,
            "Type": "Convolution",
        }
    )

    # Combinar los DataFrames
    _df_all = pd.concat([_dataframe_step, _dataframe_saw, _dataframe_combined])


    # Crear la gráfica
    _chart = (
        alt.Chart(_df_all)
        .mark_line(point=True, size=2)  # Líneas y puntos visibles
        .encode(
            x="Time",
            y="Amplitude",
            color="Type",
        )
    )

    # Mostrar la gráfica
    mo.ui.altair_chart(_chart)
    return (
        alt,
        combined_signal,
        np,
        original_signal_saw,
        original_signal_step,
        pd,
        time,
    )


app._unparsable_cell(
    r"""
    |simport marimo as mo
    """,
    name="__"
)


@app.cell
def __():
    import numpy
    import altair
    import pandas
    return altair, numpy, pandas


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
