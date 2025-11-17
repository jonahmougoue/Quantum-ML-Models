import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def plot_energy(E_pred:list,E_true:list,file_labels:list,potential_indices:list)->None:

    '''
    :param E_pred: Predicted Energy from model
    :param E_true: True energies
    :param file_labels: Labels
    :param potential_indices: Index of each potential
    :return:
    '''

    unique_labels = sorted(set(potential_indices))
    E_true = np.array(E_true)
    E_pred = np.array(E_pred)
    V_labels = np.array(potential_indices)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(3, 2, figsize=(18, 27))

    axes = axes.flatten()  # flatten to 1D list of axes

    line = np.linspace(0, max(max(E_true), max(E_pred)), 2)
    axes[0].plot(line, line, label='Perfect Prediction',color='black')
    for i, label in enumerate(unique_labels):
        mask = V_labels == label
        axes[0].scatter(E_true[mask], E_pred[mask], label=file_labels[label],color=colors[i])
    r2 = r2_score(E_true, E_pred)
    axes[0].set_xlabel('True Energy (Ha)')
    axes[0].set_ylabel('Predicted Energy (Ha)')
    axes[0].set_title('Total Energy Accuracy')
    plot_text = rf'$R^2 = {r2:.4f}$'+f'\nMedian Absolute Error: {np.median(np.absolute(E_true - E_pred))*10**3:.2f} mHa'
    axes[0].text(0.95,0.05,plot_text,transform=axes[0].transAxes,ha='right',va='bottom')
    axes[0].legend(loc='upper left')

    for i, label in enumerate(unique_labels):
        mask = V_labels == label
        line = np.linspace(0, max(max(E_true[mask]), max(E_pred[mask])),2)
        axes[i+1].plot(line,line, label='Perfect Prediction',color='black')
        axes[i+1].scatter(E_true[mask], E_pred[mask],label=file_labels[label],color=colors[i])
        r2 = r2_score(E_true[mask],E_pred[mask])
        axes[i+1].set_xlabel('True Energy (Ha)')
        axes[i+1].set_ylabel('Predicted Energy (Ha)')
        axes[i+1].set_title(rf'{file_labels[label]} Energy Accuracy')
        plot_text = rf'$R^2 = {r2:.4f}$'+f'\nMedian Absolute Error: {np.median(np.absolute(E_true[mask] - E_pred[mask]))*10**3:.2f} mHa'
        axes[i+1].text(0.95,0.05,plot_text,transform=axes[i+1].transAxes,ha='right',va='bottom')
        axes[i+1].legend(loc='upper left')
    plt.tight_layout()
    plt.show()

