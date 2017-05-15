# poissonmf_cs
Poisson Factorization with joint content and social trust factors (paper under review)

This code is provided as it is.

To compile to code just run make and it generates three binary files:
1) poisson_scr_cpp: joint model without any weight factor between content and social part of the model
2) poisson_weighted_learn: joint model with the weight variable for content and social part of the model (it can be set as a parameter of the program or learned from the data)
3) poisson_weighted_learn_hyper: the same as number 2) but with more options concerning the hyperparameters of the model (shape and rate of the gamma priors), you can set it as parameter of the program, also there is a parameter for verbose of the log


