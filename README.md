# MES
Implementation of Max-value Entropy Search (MES). (Using only numpy)  
paper: [Max-value Entropy Search for Efficient Bayesian Optimization](https://arxiv.org/abs/1703.01968) (ICML 2017)

## Overview
In MES, the mutual information between the function value at the observation point and the optimal value is used to the acquisition function.
Since this mutual information cannot be computed analytically, they approximate the predictive distribution conditioned on the optimal value by the truncated normal distribution.

### Sampling optimal value from GP
They propose two sampling method.

**Gumbel sampling**  
Optimal value is sampled from Gumbel distribution that approximate true distribution.  

**Optimize Sampled function from GP posterior**  
First, We construct Bayesian linear regression that approximate GP posterior using Random Fourier Features.
Then, we ca sample functions by sampling weights of BLR.
We can get optimal value to maximizing sampled function.

## Plot
Simple regret and Inference regret of 20 seeds. Benchmark function is the sample path of the GP.  
Comparsion methods is follows.
- Random search
- Probability of improvement (PI)
- Expected improvement (EI)
In this experiment, since the input space is discretized, I sampled the optimal value from the predictive distributions of finite input points.

MES performed as well or better than the existing methods.  
The results show that MES has the fastest convergence to the optimal value.
|Simple regret| Inference regret|
|---|---|
|<img src=https://github.com/SK-tklab/MES/blob/main/image/MES_sr.png width="400px">  |<img src=https://github.com/SK-tklab/MES/blob/main/image/MES_ir.png width="400px">  |
