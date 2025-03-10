import numpy as np
import matplotlib.pyplot as plt
from numpy._core.numerictypes import int64
from pandas import _is_numpy_dev
from pandas._libs.internals import BlockManager
import polars as pl
import seaborn as sns
import pandas as pd


def is_num(data):
    return data.dtype.is_numeric()


data = pl.read_csv("car_price_dataset.csv")
data[:250].write_csv("car_price_dataset_reduced.csv")

for i in data:
    if i.dtype.is_numeric():
        mean = i.mean()
        var = i.var()
        covar = data.select(pl.cov(i, i)).item()
        # print(i.name, ": ", mean, var, covar)
print(data["Fuel_Type"].value_counts(), data.columns)


def plot_mean_prices(data, x, name):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    print(data.group_by(x).agg(pl.col("Price").mean()))


def plot_variance_prices(data, x, name):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    print(data.group_by(x).agg(pl.col("Price").std()))


plot_mean_prices(data, "Fuel_Type", "pic")
plot_mean_prices(data.sort("Year"), "Year", "pic")
plot_mean_prices(data, "Brand", "pic")
plot_variance_prices(data, "Fuel_Type", "pic")
plot_variance_prices(data.sort("Year"), "Year", "pic")
plot_variance_prices(data, "Brand", "pic")


def avg_price_increase(data: pl.DataFrame):
    data = data.group_by("Year").agg(pl.col("Price").mean()).sort("Year")
    arr = []
    for i in range(1, len(data)):
        diff = data["Price"][i] - data["Price"][i - 1]
        arr.append(diff)
    print(data)
    print(np.array(arr).mean())


avg_price_increase(data)


def plot_heatmap(data, name):
    data = data.select(pl.col(pl.Int64))
    labels = data.columns
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.heatmap(data.corr(), annot=True, yticklabels=labels, xticklabels=labels, ax=ax)
    fig.savefig(f"{name}.png")


def plot_boxplot(data, x, y):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.boxplot(x=x, y=y, data=data, hue=x, ax=ax)
    fig.savefig(f"boxplot {x}.png")


def plot_bar(data, x, y, name):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.barplot(data, x=x, y=y, hue=x, ax=ax)
    if x == "Year":
        ax.set_xticks([f"{2000+i}" for i in range(00, 28, 4)])
    fig.savefig(f"{name}.png")


def plot_scatterplot(data, x, y, hue, name):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.scatterplot(x=x, y=y, hue=hue, data=data, s=10, ax=ax)
    fig.savefig(f"{name}.png")


def plot_distplot(data, x, hue, name):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.histplot(data, x=x, hue=hue, stat="density")
    fig.savefig(f"{name}.png")


def plot_kdeplot(data, x, hue, name):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.kdeplot(data, x=x, hue=hue, fill=True)
    fig.savefig(f"{name}.png")


def plot_stacked(data: pl.DataFrame, x, y, z, name):
    data["Price"] = data.select(pl.col("Price"))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.lineplot(data, x=x, y=y, hue=z)
    fig.savefig(f"{name}")


plot_stacked(data, "Mileage", "Price", "Fuel_Type", "line.png")
plot_heatmap(data, "heatmap")
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
plot_scatterplot(data, "Mileage", "Price", "Year", "Year")
plot_scatterplot(data, "Mileage", "Price", "Brand", "Brand")
plot_boxplot(data, "Fuel_Type", "Price")
plot_boxplot(data, "Owner_Count", "Price")
plot_boxplot(data, "Brand", "Price")
plot_boxplot(data, "Doors", "Price")
