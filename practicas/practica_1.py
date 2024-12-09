import marimo

__generated_with = "0.9.21"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    frequency_number = mo.ui.slider(
        start=5, step=1, stop=25, label="Frecuencía", full_width=True
    )
    return (frequency_number,)


@app.cell
def __(frequency_number):
    frequency = frequency_number.value
    return (frequency,)


@app.cell
def __(mo, numpy):
    time = numpy.linspace(0, 1, 1000)

    mo.show_code()
    return (time,)


@app.cell
def __(frequency, mo):
    signal_frequency = frequency
    signal_amplitude = 1

    mo.show_code()
    return signal_amplitude, signal_frequency


@app.cell
def __(mo, numpy, signal_amplitude, signal_frequency, time):
    x = signal_amplitude * numpy.sin(2 * numpy.pi * signal_frequency * time)

    mo.show_code()
    return (x,)


@app.cell
def __(mo):
    sample_frequency_number = mo.ui.slider(
        start=1,
        step=1,
        stop=100,
        label="Frecuencía de Muestreo",
        full_width=True,
    )
    return (sample_frequency_number,)


@app.cell
def __(sample_frequency_number):
    sample_frequency = sample_frequency_number.value
    return (sample_frequency,)


@app.cell
def __(mo, numpy, sample_frequency):
    discrete_time = numpy.arange(0, 1, 1 / sample_frequency)

    mo.show_code()
    return (discrete_time,)


@app.cell
def __(discrete_time, mo, numpy, signal_amplitude, signal_frequency):
    x_sampled = signal_amplitude * numpy.sin(
        2 * numpy.pi * signal_frequency * discrete_time
    )

    mo.show_code()
    return (x_sampled,)


@app.cell
def __(
    altair,
    discrete_time,
    pandas,
    sample_frequency,
    time,
    x,
    x_sampled,
):
    dataframe_continuos = pandas.DataFrame(
        {
            "Tiempo": time,
            "Amplitud": x,
            "Tipo": ["Continua"] * len(time),
        }
    )

    dataframe_sampled = pandas.DataFrame(
        {
            "Tiempo": discrete_time,
            "Amplitud": x_sampled,
            "Tipo": ["Muestreada"] * len(discrete_time),
        }
    )

    dataframe_signals = pandas.concat([dataframe_continuos, dataframe_sampled])

    # Gráfico de líneas con colores mejorados y formato atractivo
    chart = (
        altair.Chart(dataframe_signals)
        .mark_line(size=3)  # Hacer la línea más gruesa
        .encode(
            x="Tiempo",
            y="Amplitud",
            color=altair.Color(
                "Tipo",
                scale=altair.Scale(
                    domain=["Continua", "Muestreada"], range=["#1f77b4", "#ff7f0e"]
                ),
            ),  # Colores agradables
            strokeDash="Tipo",  # Líneas discontinuas para diferenciar
            tooltip=[
                "Tiempo",
                "Amplitud",
                "Tipo",
            ],  # Tooltip para información adicional
        )
        .properties(
            title=f"Teorema de Nyquist-Shannon: Muestreo con {sample_frequency} Hz",
        )
    )

    # Agregar los puntos muestreados
    chart_samples = (
        altair.Chart(dataframe_sampled)
        .mark_point(
            filled=True, size=80, color="#ff6347"
        )  # Usar un color más atractivo para los puntos
        .encode(x="Tiempo", y="Amplitud")
    )

    # Combinar el gráfico de la señal continua y las muestras
    chart = chart + chart_samples
    return (
        chart,
        chart_samples,
        dataframe_continuos,
        dataframe_sampled,
        dataframe_signals,
    )


@app.cell
def __(
    chart,
    frequency,
    frequency_number,
    mo,
    sample_frequency,
    sample_frequency_number,
    signal_frequency,
):
    mo.vstack(
        [
            mo.vstack(
                [
                    frequency_number,
                    sample_frequency_number,
                ]
            ),
            mo.hstack(
                [
                    mo.ui.altair_chart(chart),
                    mo.vstack(
                        [
                            mo.stat(
                                frequency,
                                label="Frecuencía",
                                bordered=True,
                            ),
                            mo.stat(
                                sample_frequency,
                                label="Frecuencía de Muestreo",
                                bordered=True,
                            ),
                            mo.stat(
                                "Sí"
                                if sample_frequency <= 2 * signal_frequency
                                else "No",
                                label="Efecto Alias",
                                bordered=True,
                            ),
                        ]
                    ),
                ]
            ),
        ],
        gap=2,
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import numpy
    import pandas
    import altair
    return altair, numpy, pandas


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
