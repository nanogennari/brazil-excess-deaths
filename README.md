# An analysis of excess deaths in Brazil during the COVID-19 pandemic

![Brazil deaths](https://github.com/nanogennari/brazil-excess-deaths/blob/main/brazil_deaths.png?raw=true)

## Introduction

During a pandemic is hard to get the dimension of the impact in a population, data can be scarce and official statistics can be lower than the real values due to under reporting. In this case a reliable indicator of the impact of the disease in a population can be the excess deaths in that population.

## Companion Article

This analysis has a companion article publised on Medium: https://nanogennari.medium.com/an-analysis-of-excess-deaths-in-brazil-during-the-covid-19-pandemic-7899c1650e88

## Used data

* capyvara's scrape of the brasilian civil registry: https://github.com/capyvara/brazil-civil-registry-data
* Qour Wold on Data covid dataset: https://ourworldindata.org/covid-deaths#what-is-the-cumulative-number-of-confirmed-deaths

## Used libraries

* Pandas
* Numpy
* Matplotlib
* Seaborn

## Files

* `excess_deaths.ipynb`: contains the executed analysis.
* `excess_deaths.py`: contains auxiliary classes and functions.
* `civil_registry_deaths.csv`: capyvara's data used to calculate excess deaths.
* `owid-covid-data.csv`: Our World on Data covid dataset.