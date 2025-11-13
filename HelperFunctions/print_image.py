import torch
from torch import Tensor
from matplotlib import pyplot as plt
def print_img(image1:Tensor,
              image2:Tensor,
              title1:str=None,
              title2:str=None,
              energy:Tensor=None,
              potential_label:int=None,
              files:list=None,
              energy_pred:Tensor=None,
              energy_diff:bool=False)->None:
    """
    Function for Plotting Images
    :param image1: 256x256 map of potential or wavefunction
    :param image2: 256x256 map of potential or wavefunction
    :param title1: Title of the first image
    :param title2: Title of the second image
    :param energy: Energy of wavefunction
    :param potential_label: Label of potential
    :param files: List of potential files, required if potential_label is not None
    :param energy_pred: Predicted energy from the model
    :param energy_diff: True to display the difference in energy and energy_pred
    :return: None
    """
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    cmap = 'viridis'

    img = ax[0].imshow(image1.permute(1,2,0).cpu().numpy(),cmap=cmap)
    plt.colorbar(img)
    ax[0].set_title(title1)
    ax[0].axis('image')
    if potential_label is not None:
        ax[0].set_xlabel(files[potential_label])
    ax[0].tick_params(labelbottom=False,labelleft=False)

    img = ax[1].imshow(image2.permute(1,2,0).cpu().numpy(),cmap=cmap)
    plt.colorbar(img)
    ax[1].set_title(title2)
    ax[1].axis('image')
    if energy is not None:
        energy = energy.cpu()
        if energy_pred is None:
            ax[1].set_xlabel(rf'$E_{{True}} = {energy.item():.2f}$')
        elif not energy_diff:
            energy_pred = energy_pred.cpu()
            ax[1].set_xlabel(rf'$E_{{True}} = {energy.item():.2f}$'+'\n'+rf'$E_{{Calc}} = {energy_pred.item():.2f}$')
        else:
            energy_pred = energy_pred.cpu()
            ax[1].set_xlabel(rf'$E_{{True}} = {energy.item():.2f}$'+'\n'+rf'$E_{{Calc}} = {energy_pred.item():.2f}$'+'\n'+rf'$\Delta E = |E_{{True}} - E_{{Calc}}| = {torch.abs(energy-energy_pred).item():.4e}$')

    ax[1].tick_params(labelbottom=False,labelleft=False)

    plt.show()
    plt.close(fig)
