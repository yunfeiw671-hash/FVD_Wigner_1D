#False Vacuum Decay Simulation at Finite Temperature by Wigner Method in 1 Dimension

Overview:

This repository contains the numerical code used in the paper: “Numerical simulation of the false vacuum decay at finite temperature” (arxiv:https://arxiv.org/abs/2506.18334)

The code implements a real-time approach based on the Wigner function to study the false vacuum decay of a scalar field with interactions. The framework allows one to compute the probability of remaining in the false vacuum and extract the decay rate. The method avoids direct evaluation of the path integral and can be naturally extended to finite temperature.

Features:

  1.Simulation of scalar field dynamics in 1D with periodic boundary conditions.
  2.Initialization of fields from a Bose–Einstein distributed power spectrum.
  3.Real-time evolution of the Wigner function via the classical Liouville equation.
  4.Leapfrog algorithm for stable numerical integration.
  5.Estimation of the false vacuum decay rate from probability distributions.

Requirements：

  Python >= 3.9
  numpy
  matplotlib
  numba
  plotly
  mpmath
  scipy

Installation：

Clone the repository:

  git clone https://github.com/yunfeiw671-hash/FVD_Wigner_1D.git

  cd false-vacuum-decay

Install the required dependencies

Usage：

1. Running the Main Program (main/)

The main script FVD_wigner_1D.py simulates the false vacuum decay in one dimension.

   Inputs: It accepts four parameters: the potential parameters a, b, c, and the inverse temperature beta.

   Other simulation parameters (e.g., lattice spacing, number of samples, number of steps) can be modified directly in the script.

Example 1: Single Run

  python FVD_wigner_1D.py --a 0.8 --b -1.84 --c 0.98 --beta 2.6

This produces output files in newly generated directories. The outputs include .txt data files and .png figures (the exact outputs can be configured by commenting/uncommenting the relevant parts in the script).The main quantity of interest is stored in gamma_values.txt.

Example 2: Batch Run

It is more convenient to use run_batch.py, which allows running multiple simulations for different values of β at once.

Before execution, set the potential parameters in run_batch.py. Then run:

  python run_batch.py --betas 0.6 0.8 1.0 1.2

This creates multiple output directories:

   First-level directories are labeled as dV=..., corresponding to the potential barrier height.

   Inside each dV folder, subdirectories labeled beta=... are generated.

   Each beta directory contains the same outputs as Example 1.

2. Data Processing (data_treating/)

The data_treating/ folder contains post-processing scripts to analyze simulation outputs.

 To directly extract decay rates from the generated data, use extract_gamma.py.

    You must edit the file paths in the script to match your own output directories.

    The expected folder structure is consistent with that produced by run_batch.py.

Running this script yields the dependence of decay rates on the inverse temperature for different barrier heights.

Other scripts are provided for analyzing quantities such as field evolution, spectra, and wave functions. These scripts are intended as reference implementations and can be flexibly adapted to specific research needs.













