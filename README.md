# Linear-Regression
## TO-DO:
Implement least squares linear regression to predict density of wine based on its acidity.

## Datasets:
- [linearX.csv](https://github.com/aarunishsinha/Linear-Regression/blob/main/data/linearX.csv) - acidity of the wine
- [linearY.csv](https://github.com/aarunishsinha/Linear-Regression/blob/main/data/linearY.csv) - density of the wine

## Inference:
- Implemented batch gradient descent for optimising the least squares loss function with learning rate = 0.01 and stopping criteria as:
```
(Old_cost - New_cost) < epsilon
```
`epsilon` is a small positive quantity
Final Set of parameters obtained:
```
θ_1 = 0.0007777711241840175
θ_0 = 0.9965210161010347
```
![alt text](https://github.com/aarunishsinha/Linear-Regression/blob/main/out/q1_b.png "Hypothesis Function")

![alt text](https://github.com/aarunishsinha/Linear-Regression/blob/main/out/q1_d.png "Contours of the Error Function")

### Contours for different learning rates:
- η = 0.001
![alt text](https://github.com/aarunishsinha/Linear-Regression/blob/main/out/1q1_e.png "Contours of the Error Function")
- η = 0.025
![alt text](https://github.com/aarunishsinha/Linear-Regression/blob/main/out/2q1_e.png "Contours of the Error Function")
- η = 0.1
![alt text](https://github.com/aarunishsinha/Linear-Regression/blob/main/out/3q1_e.png "Contours of the Error Function")

From the contours above one can conclude that, as the learning rate increases the number of iterations in gradient descent decreases such that after a certain value the value of descent step gets so large that the it never converges.
