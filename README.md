# Photoacoustic Tomography Simulation and Inverse Reconstruction

This project implements a simulation and reconstruction pipeline for **Quantitative Photoacoustic Tomography (QPAT)** using the **Discrete Ordinates Method (DOM)** for modeling light transport and **wave-based backprojection** for acoustic signal inversion.

## Key Components

- **Heney-Greenstein scattering model** for anisotropic light scattering.
- **Transport solver** (8 angular directions) to compute fluence via finite differences and sparse linear systems.
- **Phantom-based absorption map** from the Shepp-Logan model.
- **Wave propagation simulation** using spectral (FFT-based) method for forward modeling of photoacoustic signals.
- **Backprojection algorithm** (WavStar) for acoustic inversion.
- **Adjoint-based gradient descent** to reconstruct the spatial absorption coefficient (μₐ).
- **Huber loss** for robust optimization.
- Optional **total variation (TV)** denoising and **piecewise block projection**.

## Dependencies

- `numpy`
- `scipy`
- `matplotlib`
- `scikit-image`

## Results

The code simulates a synthetic forward problem and reconstructs the absorption distribution from simulated pressure data. It supports optional regularization for noise reduction.  
This is useful for research in QPAT, inverse problems, and hybrid imaging.
