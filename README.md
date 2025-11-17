# Deep Learning Methods for Solving the Schrödinger Equation
The goal of this project is to create deep learning models that can solve the Schrödinger equation in two dimensions.

In 'Deep Learning and the Schrödinger Equation' (Mills, Spanner, and Tamblyn), a Convolutional Neural Network (CNN) is trained to predict ground-state directly using a discretized potential grid, bypassing the need to calculate the wavefunction.

This project extends their idea by creating a model that uses discretized potential grids to first predict the wavefunction and then numerically solves for the predicted energy.

This dataset used is the 8.51 GB sample of the 700.36 GB 'Quantum simulations of an electron in a two dimensional potential well' dataset from the National Research Council of Canada. The data was generated using a numerical simulation of an electron in multiple 256x256 discretized potential grids.

Each sample contains the potential, wavefunction, and energy.

## Part 1
A CNN is trained to map the potential grid to ground-state energy.
The architecture of this model is based off the model used in 'Deep Learning and the Schrödinger Equation'
### Results
The CNN achieved a median absolute error of 1.54 mHa with R^2 of 0.9985, larger than the median absolute error of 1.49 mHa found in 'Deep Learning and the Schrödinger Equation'.

## Part 2
A U-Net model is create to map the potential grid to the ground-state wavefunction.
The predicted energy is then calculated using numerical approximation for the wavefunction's ground-state energy.
A custom loss function optimizes energy differences as opposed to predicted wavefunction differences.
### Results
In energy prediction, the U-Net achieved a median absolute error of 0.88 mHa with R^2 = 0.9997 over 10 epochs, lower than the median absolute error of 1.49 mHa found in 'Deep Learning and the Schrödinger Equation'.

## Conclusion
Predicting the wavefunction and numerically calculating the energy proved more accurate than directly calculating the energy.
Both models perform under the threshold for chemical accuracy (1.6 mHa).
These results indicate that learning the wavefunction and calculating the energy can outperform directly predicting the energy.
The custom loss function for the U-Net causes the model to produce wavefunctions that produce the correct energy, and thus make wavefunctions that satisfy the Shrödinger equation.
## Appendix

### Units
Length is measured in atomic units. 

Atomic units are used such that ℏ and m equal 1.

Potential and energy are both measured in Hartrees. 

Energy differences are expressed in millihartrees.

### Notation
V: Potential

Ψ: Wavefunction

E_True: Energy of the wavefunction as specified in the dataset

E_Pred: Energy of the wavefunction as calculated by the model


### dx
The dataset specifies that the potentials are defined ona grid from x,y = -20 to 20 a.u. on a 256x256 grid.

Therefore, dx = 40/256 = 0.15625.

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