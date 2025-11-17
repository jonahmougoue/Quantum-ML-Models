# Deep Learning Methods for Solving the time-independent Schrödinger Equation



## Introduction
The time-independent Schrödinger equation is defined in 2D as:
$$\hat{H} \psi(x,y) = E\psi(x,y)$$
Where $\hat{H}$ is the hamiltonian operator, defined as: 
$$\hat{H} = -\frac{\hbar^2}{2m} \nabla^2 + V(x,y)$$
Using atomic units, where $\hbar = m_e = 1$, the equation can be simplified to:
$$-\frac{1}{2}\nabla^2\psi(x,y) + V(x,y)\psi(x,y) = E\psi(x,y)$$
Each potential defines a set of eigenstates $\psi_i$ with eigenvalues $E_i$.
There is no analytic solution for $\psi$ in the time-independent Schrödinger equation for any arbitrary $V$, but it can be solved analytically for specific classes of $V$.
The energy of a wavefunction can be calculated using the equation:
$$E_i = \frac{<\psi_i|\hat{H}|\psi_i>}{<\psi_i|\psi_i>}$$

 Once the ground state wavefunction $\psi_0$ and ground state energy $E_0$ are known, numerous numerical methods in quantum mechanics can be used to approximate the larger eigenvalues and eigenstates.
Thus, finding accurate approximations for $\psi_0$ and $E_0$ is essential for approximating $\psi_i$ and $E_i$. 

## About
This project develops deep learning models for solving the Schrödinger equation in two dimensions from a discretized potential grid.

In 'Deep Learning and the Schrödinger Equation' (Mills, Spanner, and Tamblyn), a Convolutional Neural Network (CNN) is trained to predict ground-state energy directly using a discretized potential grid, bypassing the need to calculate the wavefunction.

This project compares models trained for direct predictions of $E_0$ with models trained for operator learning between $V$ and $\psi_0$, from which energy is numerically calculated.

The dataset used is the 8.51 GB sample of the 700.36 GB 'Quantum simulations of an electron in a two dimensional potential well' dataset from the National Research Council of Canada. The data was generated using a numerical simulation of an electron in multiple 256x256 discretized potential grids.

Each sample contains $V$, $|\psi_0|^2$, and $E_0$.

## Installation

```bash
git clone https://github.com/jonahmougoue/Quantum-ML-Models.git
cd Quantum-ML-Models
pip install -r requirements.txt
```
Run all cells in:
```
Part1/predict_energy.ipynb
Part2/predict_wavefunction.ipynb
```
## File Structure
```
.
├── Data
│   └── quantum_dataset.py #Pytorch Dataset class containing 25000 samples
├── HelperFunctions
│   ├── plot_energy.py #Helper for plotting energy accuracy
│   ├── plot_loss.py #Helper for displaying loss curves
│   └── print_image.py #Helper for displaying potentials and wavefunctions
├── Part1
│   ├── energy_cnn.py #CNN architecture for direct energy prediction
│   └── predict_energy.ipynb #Notebook for energy_cnn.py evaluation
├── Part2
│   ├── energy_loss.py #Custom loss for evaluating energy accuracy
│   ├── predict_wavefunction.ipynb #Notebook for wavefunction_unet.py evaluation
│   └── wavefunction_unet.py #U-Net architecture for wavefunction prediction
├── requirements.txt
└── README.md
```

## Part 1: Direct Energy Prediction
A CNN is trained to map a potential grid to the ground-state energy.
The architecture of this model is based off the model used in 'Deep Learning and the Schrödinger Equation'
### Results
The CNN achieved a median absolute error of 1.54 mHa with R^2 of 0.9985, larger than the median absolute error of 1.49 mHa found in 'Deep Learning and the Schrödinger Equation'.

## Part 2: Operator Learning
A U-Net model is created for operator learning, mapping potential grids to ground-state wavefunctions.
The predicted energy is then computed numerically from the wavefunction.
A custom loss function is used to penalize error in energy as opposed to error in wavefunction.
### Results
In energy prediction, the U-Net achieved a median absolute error of 0.88 mHa with R^2 = 0.9997 over 10 epochs, lower than the median absolute error of 1.49 mHa found in 'Deep Learning and the Schrödinger Equation'.

## Conclusion
Predicting the wavefunction and computing the energy numerically proved more accurate than directly predicting the energy.
Both models perform under the threshold for chemical accuracy (1.6 mHa).
These results indicate that learning the wavefunction and calculating the energy can outperform directly predicting the energy.
The custom loss function for the U-Net causes the model to produce wavefunctions that produce the correct energy, and thus make wavefunctions that satisfy the Shrödinger equation.
## Appendix

### Units
Length is measured in atomic units. 

Potential and energy are both measured in Hartrees. 

Energy differences are expressed in millihartrees.

### Notation
$V$: Potential

$\psi$: Wavefunction

$E_{True}$: Energy of the wavefunction as specified in the dataset

$E_{Pred}$: Energy of the wavefunction as calculated by the model


### dx
The dataset specifies that the potentials are defined on a grid from x,y = -20 to 20 a.u. on a 256x256 grid.

Therefore, $dx = \frac{40}{256} = 0.15625.$

This value is required to numerically calculate the correct energies and can vary by dataset.

## Credit
This research project is based on the publication:

Mills, Kyle, Michael Spanner, and Isaac Tamblyn. ‘Deep Learning and the Schrödinger Equation’. Physical Review A 96, no. 4 (18 October 2017). https://doi.org/10.1103/physreva.96.042113.

Data sourced from the National Research Council of Canada:
https://nrc-digital-repository.canada.ca/eng/view/object/?id=1343ae23-cebf-45c6-94c3-ddebdb2f23c6
	
### Licenses:
Open Government Licence - Canada (National Research Council of Canada)
https://open.canada.ca/en/open-government-licence-canada
Creative Commons, Attribution 2.0 Generic (CC BY 2.0) (University of Ontario Institute of Technology)
http://creativecommons.org/licenses/by/2.0/