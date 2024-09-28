# Energy-Price-Interval-Forecasting
Final project for the Financial Engineering (FE) Course at Politecnico di Milano, Quantitative Finance.
Developed by Sara Cupini, Francesco Panichi, Davide Pagani. On Python in an Object-Oriented fashion. Final report is present.

DISCLAIMER: the whole project builds on the foundation code developed by professor Alessandro Brusaferri (researcher at CNR), 
who kindly offered it to us during the laboratory lessons of the FE course. 
This repository therefore does not contain the whole project, to avoid unjust publication of the professor's contents.
Rather, it collects the scripts which were written directly by us for the project, which leverage on the (here unpublished) codebase.

PRESENTING THE THREE SCRIPTS:

__ data_analysis.py: we perfomed data cleaning, validation, aggregation and visualization. The dataset consists of hourly time series
of energy prices, solar and wind production, energy consumption, plus some calendar variables.

__ SARIMAX.py: defines a class through which forecasts of the energy price can be made using a SARIMAX model, that is an auto-regressive
and moving-average with external variables, which can be added as dummies using functions of the class.

__ recalibration_functions.py: defines a class through which an Ensemble model can be created, mixing SARIMAX and other models' point
predictions. Then creates the interval predictions using Conformal Prediction technique. Then computes metrics of evaluation, both for
point and interval forecasting.
