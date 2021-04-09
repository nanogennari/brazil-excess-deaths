# An analysis of excess deaths in Brazil during the COVID-19 pandemic

![Brazil deaths](https://github.com/nanogennari/brazil-excess-deaths/blob/main/brazil_deaths.png?raw=true)

## Introduction

During a pandemic is hard to get the dimension of the impact in a population, data can be scarce and official statistics can be lower than the real values due to under reporting. In this case a reliable indicator of the impact of the disease in a population can be the excess deaths in that population.

## Companion Article

This analysis has a companion article publised on Medium: https://nanogennari.medium.com/an-analysis-of-excess-deaths-in-brazil-during-the-covid-19-pandemic-7899c1650e88

## Used data

* capyvara's scrape of the brazilian civil registry: https://github.com/capyvara/brazil-civil-registry-data
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

## Results

A correlation between the number of excess deaths and the confirmed COVID deaths was observed in the considered period (March/2020-March/2021).

![Excess vs confirmed deaths](https://github.com/nanogennari/brazil-excess-deaths/blob/main/excess_deaths_relation.png?raw=true)

The states with the biggest increase in excess deaths where:

+----------------+---------+-----------------------+
|     State      | P-score | Deaths above the mean |
+----------------+---------+-----------------------+
| Amazonas       | 236.71% |                18,709 |
| Maranhão       | 166.07% |                20,388 |
| Pará           | 129.57% |                21,635 |
| Rio de Janeiro | 86.49%  |                89,240 |
| Ceará          | 79.37%  |                28.095 |
+----------------+---------+-----------------------+

The cities with biggest increase in excess deaths where (restrictions where applied, see the notebook):

+-------------------------+---------+--------+
|          City           | p-score | Deaths |
+-------------------------+---------+--------+
| Caxias-MA               |  268.34 |    788 |
| Manaus-AM               |  221.13 |  14964 |
| Benevides-PA            |  217.70 |    264 |
| Viçosa do Ceará-CE      |  205.36 |    268 |
| Itacoatiara-AM          |  205.34 |    369 |
| Brejo Santo-CE          |  190.51 |    246 |
| Colinas do Tocantins-TO |  185.41 |    200 |
| Maricá-RJ               |  185.18 |   1237 |
| Acopiara-CE             |  181.40 |    190 |
| Arraial do Cabo-RJ      |  179.69 |    210 |
+-------------------------+---------+--------+

## Acknowledgments

First and foremost, praises and thanks to capyvara for making the data necessary for this analysis available and easy to work with. Also thanks to the Our World in Data for making awesome data easy to get and to work on.

This analysis was made as a part of the Udacity's Data Scientist Nanodegree Program.
