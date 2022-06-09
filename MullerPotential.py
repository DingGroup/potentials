import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from sys import exit
import pickle

class MullerPotential(nn.Module):
    """
    The Muller potential as defined in the paper:
    Müller, K., Brown, L.D. Location of saddle points and minimum energy paths by a constrained simplex optimization procedure. Theoret. Chim. Acta 53, 75–93 (1979).
    https://doi.org/10.1007/BF00547608
    """

    def __init__(self, alpha):
        """ Initialize the potential by setting a scalar variable alpha
        which is used to scale the potential.
        
        Args:
            alpha (scalar): a scalar to be multiplied to the original Muller potential
        """
        super(MullerPotential, self).__init__()
        self.alpha = alpha

        self.A = nn.Parameter(
            torch.tensor([-200.0, -100.0, -170.0, 15.0]), requires_grad=False
        )
        self.b = nn.Parameter(torch.tensor([0.0, 0.0, 11.0, 0.6]), requires_grad=False)
        self.ac = nn.Parameter(
            torch.tensor([[-1.0, -10.0], [-1.0, -10.0], [-6.5, -6.5], [0.7, 0.7]]),
            requires_grad=False,
        )
        self.x0 = nn.Parameter(
            torch.tensor([[1.0, 0.0], [0.0, 0.5], [-0.5, 1.5], [-1.0, 1.0]]),
            requires_grad=False,
        )

    def compute_potential(self, x):
        """ Compute the potential value on a batch of positions

        Args:
            x (tensor): positions where the potential is computed.

        Returns:
           u (tensor): potential values.

        """
        assert(x.shape[-1] == 2)
        
        x = x.unsqueeze(1)
        d = x - self.x0

        u = self.A * torch.exp(
            torch.sum(self.ac * d**2, -1) + self.b * torch.prod(d, -1)
        )
        u = torch.sum(u, -1)
        u = self.alpha * u

        return u

    def compute_force(self, x):
        """ Compute the force on a batch of positions

        Args:
            x (tensor): positions where the potential is computed.

        Returns:
           f (tensor): force.

        """

        x_new = x.clone().detach().requires_grad_(True)
        u = self.compute_potential(x_new).sum()
        u.backward()
        return x_new.grad
    
    def plot_to(self, file_name):
        x1_min, x1_max = -1.5, 1.0
        x2_min, x2_max = -0.5, 2.0
        steps = 100
        x1 = torch.linspace(x1_min, x1_max, steps=steps)
        x2 = torch.linspace(x2_min, x2_max, steps=steps)
        grid_x1, grid_x2 = torch.meshgrid(x1, x2)
        grid = torch.stack([grid_x1, grid_x2], dim=-1)
        x = grid.reshape((-1, 2))

        fig = plt.figure()
        fig.clf()
        U = self.compute_potential(x)
        U = U.reshape(steps, steps)
        U_min = U.min()
        U[U > U_min + 20] = U_min + 20
        U = U.T
        plt.contourf(
            U, levels=30, extent=(x1_min, x1_max, x2_min, x2_max), cmap=cm.viridis_r
        )
        plt.xlabel(r"$x_1$", fontsize=24)
        plt.ylabel(r"$x_2$", fontsize=24)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()
