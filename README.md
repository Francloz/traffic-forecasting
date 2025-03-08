# Traffic Forecasting of PeMS-BAY and METR-LA

## Data collection
1. Source PeMS-BAY and METR-LA

## Preprocessing
1. Normalization
2. Differentiation
3. Clustering / Segmentation (for group-wise processing and modeling)
4. Outliers and Data Imputation

## Tools and methods
Mainly: Propet, Kats, Scipy, Scikit-learn, PyTorch and pgmpy.

## Modeling
We will use the following options:
1. Graph NN with LSTMs
2. Traditional models: ARIMA, VARIMA, ETS, etc.
3. Usual deep FFNN

## Evaluation
1. Preprocessing ablation study. Removing parts of the preprocessing to see the effect on the modeling accuracy
2. Accuracy at 5, 15 and 30 minutes into the future.


## Modeling Details
### GNN + LSTMs
Studying the physical relationship between sensors, i.e. which road's sensor feeds into another sensor's road, is 
impractical with large numbers of sensors. Because of that we will attempt to model the causal graph using bayesian 
network principles.

Once the graph is built, a Bayesian Optimizer will be used to find hyperparameter regions that show promise for few epochs. 
After this, a targeted Bayesian Optimizer with an increased number of epochs. Lastly, a select few models found with the
optimizer will be trained until they start to overfit.



## References  
<a id="1">[1]</a>  
Kwak, S. (2020).  
*PEMS-BAY and METR-LA in CSV*.  
Zenodo. [https://doi.org/10.5281/zenodo.5724362](https://doi.org/10.5281/zenodo.5724362). 