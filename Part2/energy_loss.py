import torch
from torch import nn
from torch import Tensor

class EnergyLoss(nn.Module):
    def __init__(self,alpha:float=1.,gamma:float=0.,dx:float=0.15625,loss_fn:nn.Module=nn.L1Loss()):
        """
        Loss function for calculating the difference in energy between true energy and energy predicted from a wavefunction\n
        Calculates energy using the time-independent schrodinger equation:\n
        H = -1/2 * Δ^2 + V\n
        <Ψ|H|Ψ> / <Ψ|Ψ> = E
        :param alpha: Value that linearly scales the output of the loss
        :param gamma: Determines type of kernel used for the laplacian
        :param dx: Distance between pixels
        :param loss_fn: Loss function used to evaluate energy accuracy
        """
        super().__init__()
        self.alpha = alpha
        self.loss_fn = loss_fn
        with torch.no_grad():
            self.laplacian = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding=1,padding_mode='reflect',bias=False)
            kernel = (1-gamma)*torch.tensor([[0,1,0],[1,-4,1],[0,1,0]])+gamma*torch.tensor([[1/2,0,1/2],[0,-2,0],[1/2,0,1/2]])
            self.laplacian.weight[:] = kernel.reshape(1,1,3,3)/(dx**2)

    def forward(self,potential:Tensor,wavefunction:Tensor,energy:Tensor)->Tensor:
        """
        Calculates loss between True and predicted energy
        :param potential: Potential Image
        :param wavefunction: Wavefunction Prediction
        :param energy: True Energy
        :return: Loss between True energy and energy from wavefunction
        """
        energy_pred = self.get_energy(potential,wavefunction)
        loss = self.loss_fn(energy_pred,energy)
        return loss*self.alpha

    def get_ke(self,wavefunction:Tensor)->Tensor:
        """
        Calculates Kinetic Energy of the wavefunction using 5-point stencil approximation method
        :param wavefunction: Wavefunction Prediction
        :return: Kinetic Energy
        """
        return -1/2 * self.laplacian(wavefunction)

    def get_energy(self,potential:Tensor,wavefunction:Tensor)->Tensor:
        """
        Calculates the energy of the wavefunction using the time-independent schrodinger equation:\n
        H = -1/2 * Δ^2 + V\n
        <Ψ|H|Ψ> / <Ψ|Ψ> = E
        :param potential: Potential Map
        :param wavefunction: Wavefunction Prediction
        :return: Predicted Energy
        """
        energy_pred = (self.get_ke(wavefunction) * wavefunction + potential * wavefunction**2).sum(dim=(2,3))/(wavefunction**2).sum(dim=(2,3))
        return energy_pred

