# Deep Learning Methods for Calculating the Quantum Wavefunction
The goal of this project is to create a deep learning model that can predict the energy and wavefunction for a quantum particle in a potential well.
In 'Deep Learning and the Schrödinger Equation', the authors create a CNN model to calculate the energy of quantum wavefunctions using solely 256x256 maps of their potentials.
This project seeks to extend their method from predicting energy from 256x256 potential maps to predicting 256x256 wavefunction maps from potential maps and using numerical methods the wavefunction's energy.

## Part 1
A CNN model is created to take a 256x256 potential map as input and output the predicted ground-state energy of the wavefunction.
This model is based off the model produced in 'Deep Learning and the Schrödinger Equation'
## Part 2
A UNet Model is create to take a 256x256 potential map as input and output a 256x256 ground-state map for the predicted wavefunction.
A custom loss function is used to measure the difference in energy of the predicted wavefunction compared to the true energy.

## Conclusion

## Appendix

### Calculating dx:
The distance between each pixel in the potential and wavefunction images is not given by the dataset, but is required for EnergyLoss to calculate the correct energies. find_dx.ipynb iterates over multiple values of dx to find the value that minimizes the differences in the true energy of the wavefunction and the calculated energy.

## Credit
This research project is based on the publication:

Mills, Kyle, Michael Spanner, and Isaac Tamblyn. ‘Deep Learning and the Schrödinger Equation’. Physical Review A 96, no. 4 (18 October 2017). https://doi.org/10.1103/physreva.96.042113.

Data was sourced from the National Research Council of Canada:
https://nrc-digital-repository.canada.ca/eng/view/object/?id=1343ae23-cebf-45c6-94c3-ddebdb2f23c6
	
### Licenses:
Open Government Licence - Canada (National Research Council of Canada)
https://open.canada.ca/en/open-government-licence-canada
Creative Commons, Attribution 2.0 Generic (CC BY 2.0) (University of Ontario Institute of Technology)
http://creativecommons.org/licenses/by/2.0/