# Honours
Honours_Project

Here are the explanations for each of the python code file:

1. bm.py and bm1.py are the simulations of 1-dimensional Brownian motion.
2. bm2.py is the simulation of a 2-dimensional Brownian motion.

Forward Problem:

3. cirem.py and EMCIR.py are different version of Euler-Maruyama method that simulates CIR model.
4. EMVasicek.py is the simulation of Euler-Maruyama method simulates Vasicek model.
5. sde.py, NNCIR.py and NNVasicek.py are the simulations of Neural Network method of both CIR and Vasicek model.

Inverse Problem:

6. ls.py and nn.py are two versions of the least square method of parameter estimation for CIR model.
7. NNPE.py is the Neural Network method of parameter estimation for CIR model.
8. RNN.py and LSTM.py are attempts of RNN and LSTM neural network, but failed to estimate the parameters.
9. plot.py is the function that plot the real data.
10. t_analysis.py is the non-linear equation method for parameters estimation.

Plots:

11. The file names ended with png are the plots of results.
12. The png file contained in CIR, Vasicek, NNCIR, CIR, Plot folders are also the plots.

Others:

13. torchsde-master folder is the torchsde package required to solve the SDEs with Neural Network.
14. data.csv is the real interest rate dataset.