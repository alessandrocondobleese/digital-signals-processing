import marimo

__generated_with = "0.9.32"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import pandas as pd
    import altair as alt
    return alt, pd


@app.cell
def __():
    import numpy as np
    return (np,)


@app.cell
def __(mo):
    samples_number = mo.ui.number(start=8)
    samples_number
    return (samples_number,)


@app.cell
def __(samples_number):
    samples = samples_number.value
    return (samples,)


@app.cell
def __(alt, mo, np, pd, samples):
    data = pd.DataFrame(
        {
            "N": range(1, samples),  # Valores de N del 1 al 100
            "N_squared": [n**2 for n in range(1, samples)],
            "N_log_N": [n * np.log(n) for n in range(1, samples)]
        }
    )


    chart = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("N", title="N"),
            y=alt.Y("value:Q", title="Value"),
            color=alt.Color("variable:N", legend=alt.Legend(title="Curves")),
        )
        .transform_fold(["N", "N_squared", "N_log_N"], as_=["variable", "value"])
    )

    mo.ui.altair_chart(chart)
    return chart, data


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
