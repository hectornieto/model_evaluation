# model_evaluation

## Synopsis

This project contains *Python* code for evaluating geophysical models. 

The project consists of: 

1. A module for conventional model evaluation based on two collocated systems (a reference considered as "ground truth" and an estimate)

2. A module for Triple Collocation Analysis of three collocated systems in which all three are considered that contain errors.

## Installation

Download the project to your local system, enter the download directory and then type

`python setup.py install` 

The following Python libraries will be required:

- Numpy
- Scipy


## Code Example
### Double collocation

Mean of the observed and predicted (and hence mean error bias), Mean Absolute Error and Root Mean Square Error can be computed as.
```python
import model_evaluation.double_collocation as dc
mean_obs, mean_pre, mae, rmse = dc.error_metrics(obs, pre)
```

RMSE Wilmott's decomposition between its systematic and unsystematic (noise error) compontes are obtained as.
```python
rmse_s, rmse_u = dc.rmse_wilmott_decomposition(obs, pre)
```

Correlation coefficient and its significance, linear regression coefficients (slope and intercept) and Willmott's Index of Agreement between the observed and the predicted can be computed as.
```python
cor, p_value, slope, intercept, d = dc.agreement_metrics(obs, pre)
```

### Triple collocation
Noise standard error, correlation coefficient to the true value, signal to noise ratio in decibels and senstivity of the measurement system to changes in the target variable are computed as.
```python
import model_evaluation.tripe_collocation as tc

# First compute the covariance matrix for the triple collocated systems
q_hat = tc.covariance_matrix(x, y, z)

# etc method does not require variable rescaling
stderr, rho, snr_db, sensitivity = tc.etc(q_hat)

```
A vectorized version for computing the covariance matrix is available is a spatially distributed TC analysis is desired
```python
# First compute the covariance matrix for the triple collocated systems
# Each system represent an array of shape (N, f) with N samples and f independent elemenents (e.g. spatial pixels)
q_hat = tc.covariance_matrix_vec(x, y, z)

# etc method does not require variable rescaling
stderr, rho, snr_db, sensitivity = tc.etc(q_hat)

```

## Main Scientific References
- Willmott, C. J. (1982). Some Comments on the Evaluation of Model Performance, Bulletin of the American Meteorological Society, 63(11), 1309-1313. https://doi.org/10.1175/1520-0477(1982)063<1309:SCOTEO>2.0.CO;2.
- McColl, K.A., J. Vogelzang, A.G. Konings, D. Entekhabi, M. Piles, A. Stoffelen (2014). Extended Triple Collocation: Estimating errors and correlation coefficients with respect to an unknown target. Geophysical Research Letters 41:6229-6236. https://doi.org/10.1002/2014GL061322
- Gruber, A., Su, C.-H., Zwieback, S., Crow, W., Dorigo, W., Wagner, W., 2016. Recent advances in (soil moisture) triple collocation analysis. International Journal of Applied Earth Observation and Geoinformation 45, 200–211. https://doi.org/10.1016/j.jag.2015.09.002
- Yilmaz, M.T., Crow, W.T., 2014. Evaluation of Assumptions in Soil Moisture Triple Collocation Analysis. Journal of Hydrometeorology 15, 1293–1302. https://doi.org/10.1175/JHM-D-13-0158.1
- Yilmaz, M.T., Crow, W.T., 2013. The Optimality of Potential Rescaling Approaches in Land Data Assimilation. Journal of Hydrometeorology 14, 650–660. https://doi.org/10.1175/JHM-D-12-052.1
- González-Gambau, V., Turiel, A., González-Haro, C., Martínez, J., Olmedo, E., Oliva, R., Martín-Neira, M., 2020. Triple Collocation Analysis for Two Error-Correlated Datasets: Application to L-Band Brightness Temperatures over Land. Remote Sensing 12, 3381. https://doi.org/10.3390/rs12203381

## Tests
The folder *./Test* contains test scripts to evaluate the validity of the modules.

## Contributors
- **Hector Nieto** <hector.nieto@complutig.com> <hector.nieto.solana@gmail.com> main developer

## License
Copyright 2021 Hector Nieto and contributors.
    
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
