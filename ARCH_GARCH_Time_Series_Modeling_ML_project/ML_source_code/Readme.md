# ARCH and GARCH Model

## ARCH Model

ARCH — Autoregressive Conditional Heteroskedasticity
The ARCH process introduced by Engle (1982) explicitly recognizes the difference between the unconditional and the conditional variance allowing the latter to change over time as a function of past errors.
Autoregressive: The current value can be expressed as a function of the previous values i.e. they are correlated.
Conditional: This informs that the variance is based on past errors.
Heteroskedasticity: This implies the series displays unusual variance (varying variance).

## Correlation vs AutoCorrelation

- Correlation is a bivariate analysis that measures the strength of association between two variables and the direction of the relationship. In terms of the strength of relationship, the value of the correlation coefficient varies between +1 and -1.
- A value of ± 1 indicates a perfect degree of association between the two variables. As the correlation coefficient value goes towards 0, the relationship between the two variables will be weaker.
- Auto-correlation refers to the case when your errors are correlated with each other. In layman terms, if the current observation of your dependent variable is correlated with your past observations, you end up in the trap of auto-correlation. 

## GARCH Model

GARCH is a better fit for modeling time series data when the data exhibits heteroskedacisticity but also volatility clustering.
As the name suggests, the GARCH is just the generalized version of the ARCH model. This generalization is expressed in including past variances as well as past squared residuals to estimate current (and subsequent) variances. 
The generalization comes from the fact that including a single past variance would (in theory) contain in itself the explanatory power of all other previous squared error terms. 
It serves as a sort of ARMA equivalent to the ARCH, where we’re including both past values and past errors (albeit squared). 


## Time Series Basics

-   Chronological Data
- Cannot be shuffled
- Each row indicate specific time record
- Train – Test split happens chronologically
- Data is analyzed univariately (for given use case)
- Nature of the data represents if it can be predicted or not

## Code Description


    File Name : Engine.py
    File Description : Main class for starting different parts and processes of the lifecycle


    File Name : Arch_Model.py
    File Description : Code to train and visualize the ARCH model



## Steps to Run

There are two ways to execute the end to end flow.

- Modular Code
- IPython

### Modular code

- Create virtualenv
- Install requirements `pip install -r requirements.txt`
- Run Code `python Engine.py`
- Check output for all the visualization

### IPython Google Colab

Follow the instructions in the notebook `ARCH_and_GARCH.ipynb`

