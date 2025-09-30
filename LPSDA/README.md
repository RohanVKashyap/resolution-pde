# Forked Repository from Lie Point Symmetry Data Augmentation for Neural PDE Solvers

This is a forked repository from <a href=https://github.com/brandstetter-johannes/LPSDA> Lie Point Symmetry Data Augmentation for Neural PDE Solvers</a> (Johannes Brandstetter, Max Welling, Daniel Worrall) to enable generating PDEs with the LPSDA method but with two modifications:
1. The $\Delta t$ is not sampled randomly for each trajectory
2. The number of Fourier modes of the initial condition and the viscosity are arguments that can be passed through the command line.

An example command to generate trajectories for the KS dataset with viscosity $\nu=0.1$ and 8 Fourier modes in the initial condition is the following:
```
python generate/generate_data.py --experiment=KS --train_samples=2048 --valid__samples=256 --test__samples=0 --L=64 --nt=51  --nx=512 
--nt_effective=51 --viscosity=0.1 --end_time=5.0 --lmax=8
```
