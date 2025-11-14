# Deep Learning Methods for Calculating the Quantum Wavefunction
The goal of this project is to create deep learning models that can predict the ground-state energy and wavefunction of a quantum particle in an arbitrary potential.

In 'Deep Learning and the Schrödinger Equation' (Mills, Spanner, and Tamblyn), a Convolutional Neural Network (CNN) is used to calculate the ground-state energies using a discretized potential grid, bypassing the need to calculate the wavefunctions first.

This project extends their idea by using discretized potential grids to first predict the wavefunction and then numerically solving for the predicted energy.

This project is trained using an 8.51 GB sample of the 700.36 GB 'Quantum simulations of an electron in a two dimensional potential well' dataset from the National Research Council of Canada. The data was generated using a numerical simulation of an electron in multiple 256x256 discretized potential grids.

Each sample contains the potential, wavefunction, and energy.

## Part 1
A CNN is trained to take a potential grid as input and output the predicted ground-state energy.
The architecture of this model is based off the model used in 'Deep Learning and the Schrödinger Equation'
### Results

## Part 2
A U-Net Model is create to take a potential grid as input and output the ground-state wavefunction.
A custom loss function is used to measure the difference in energy of the predicted wavefunction compared to the true energy.
### Results

## Conclusion

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

E_Calc: Energy of the wavefunction as calculated by the model


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