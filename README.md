#False Vacuum Decay Simulation at Finite Temperature by Wigner Method in 1 Dimension

Overview
This repository contains the numerical code used in the paper: “Numerical simulation of the false vacuum decay at finite temperature” (arxiv:https://arxiv.org/abs/2506.18334)

The code implements a real-time approach based on the Wigner function to study the false vacuum decay of a scalar field with interactions. The framework allows one to compute the probability of remaining in the false vacuum and extract the decay rate. The method avoids direct evaluation of the path integral and can be naturally extended to finite temperature.

Features
1.Simulation of scalar field dynamics in 1D with periodic boundary conditions.
2.Initialization of fields from a Bose–Einstein distributed power spectrum.
3.Real-time evolution of the Wigner function via the classical Liouville equation.
4.Leapfrog algorithm for stable numerical integration.
5.Estimation of the false vacuum decay rate from probability distributions.




