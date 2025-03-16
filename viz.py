import numpy as np
import matplotlib.pyplot as plt
from numpy._core.numerictypes import int64
from pandas import _is_numpy_dev
from pandas._libs.internals import BlockManager
import polars as pl
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


data = pl.read_csv("car_price_dataset.csv")
data.describe().write_json("data.json")


def mean_prices(data, x):
    print(data.group_by(x).agg(pl.col("Price").mean()).sort("Price"))


def mean_mileage(data, x):
    print(data.group_by(x).agg(pl.col("Mileage").mean()).sort("Mileage"))


def variance_prices(data, x):
    print(data.group_by(x).agg(pl.col("Price").std()).sort("Price"))


def avg_price_increase(data: pl.DataFrame):
    data = data.group_by("Year").agg(pl.col("Price").mean()).sort("Year")
    arr = []
    for i in range(1, len(data)):
        diff = data["Price"][i] - data["Price"][i - 1]
        arr.append(diff)


def plot_heatmap(data: pl.DataFrame, name):
    data = data.select(pl.col(pl.Int64, pl.Float64))
    labels = data.columns
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.heatmap(data.corr(), annot=True, yticklabels=labels, xticklabels=labels, ax=ax)
    fig.savefig(f"{name}.png")


def plot_covar(data: pl.DataFrame):
    data = data.select(pl.col(pl.Int64, pl.Float64))
    labels = data.columns
    cov_matrix = data.to_pandas().cov()
    log_cov = np.log10(np.abs(cov_matrix) + 1) * np.sign(cov_matrix)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.heatmap(
        log_cov,
        annot=True,
        yticklabels=labels,
        xticklabels=labels,
        cmap="magma",
        ax=ax,
    )
    fig.savefig("covar.png")


def plot_boxplot(data: pl.DataFrame, x, y, labelsize=10):
    sorted_order = (
        data.group_by(x)
        .agg(pl.col(y).mean())
        .sort(y)  # Sort brands by mean price
        .get_column(x)
        .to_list()
    )
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.violinplot(x=x, y=y, data=data, order=sorted_order, hue=x, ax=ax)
    ax.tick_params(labelsize=labelsize)
    fig.savefig(f"{x} boxplot.png")


def plot_bar(data, x, y, name):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.barplot(data, x=x, y=y, hue=x, ax=ax)
    if x == "Year":
        ax.set_xticks([f"{2000+i}" for i in range(00, 28, 4)])
    fig.savefig(f"{name}.png")


def plot_scatterplot(data: pl.DataFrame, x, y, hue, name):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.scatterplot(x=x, y=y, hue=hue, data=data.sort(hue), s=10, ax=ax)
    fig.savefig(f"{name}.png")


def plot_lines(data: pl.DataFrame, x, y, hue):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for i in data[hue].unique().sort():
        df = data.filter(pl.col(hue) == i)
        xline = df[x].to_numpy().astype(float)
        yline = df[y].to_numpy().astype(float)
        a, b = np.polyfit(xline, yline, 1)
        print(a, b, i)
        ax.plot(xline, xline * a + b)
    ax.legend(data[hue].unique().sort())
    fig.savefig("lines.png")


def plot_distplot(data, x, hue, name):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.histplot(data, x=x, hue=hue, stat="density")
    fig.savefig(f"{name}.png")


def plot_kdeplot(data, x, hue, name):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.kdeplot(data, x=x, hue=hue, fill=True)
    fig.savefig(f"{name}.png")


def plot_stacked(data: pl.DataFrame, x, y, z, name):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.lineplot(data, x=x, y=y, hue=z)
    fig.savefig(f"{name}")


plot_stacked(data, "Mileage", "Price", "Fuel_Type", "line.png")
plot_heatmap(data, "heatmap")
plot_covar(data)
plot_bar(
    data.group_by("Fuel_Type").agg(pl.col("Price").mean()).sort("Fuel_Type"),
    "Fuel_Type",
    "Price",
    "price_hist",
)
plot_bar(
    data.group_by("Year").agg(pl.col("Price").mean()).sort("Year"),
    "Year",
    "Price",
    "year_bar",
)
plot_distplot(data, "Price", "Fuel_Type", "displot")
plot_kdeplot(data, "Price", "Fuel_Type", "kdeplot")
plot_scatterplot(data, "Mileage", "Price", "Fuel_Type", "Fuel Type")
plot_lines(data, "Mileage", "Price", "Fuel_Type")
plot_scatterplot(data, "Mileage", "Price", "Year", "Year")
plot_scatterplot(data, "Mileage", "Price", "Brand", "Brand")
plot_boxplot(data, "Brand", "Price", "small")
plot_boxplot(data, "Fuel_Type", "Price")
plot_boxplot(data, "Owner_Count", "Price")
plot_boxplot(data, "Engine_Size", "Price", 6)
plot_boxplot(data, "Doors", "Price")
