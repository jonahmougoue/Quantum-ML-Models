import matplotlib.pyplot as plt
import numpy as np
def plot_energy(E_pred,E_true,file_labels,potential_labels):

    unique_labels = sorted(set(potential_labels))
    E_true = np.array(E_true)
    E_pred = np.array(E_pred)
    V_labels = np.array(potential_labels)

    for label in unique_labels:
        mask = V_labels == label
        plt.scatter(E_true[mask], E_pred[mask], label=file_labels[label])

    line = np.linspace(0, max(max(E_true), max(E_pred)),2)
    plt.plot(line,line,label='y=x')
    plt.xlabel('True Energy (Ha)')
    plt.ylabel('Predicted Energy (Ha)')
    plt.title('Energy Accuracy')
    plt.legend()
    plt.show()