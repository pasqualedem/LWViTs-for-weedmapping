import torch
import numpy as np

from ezdl.loss import ComposedLoss

class BranchAuxiliaryLoss(ComposedLoss):
    name = "BAuxLoss"
    """ Branch Auxiliary loss, wraps the task loss and auxiliary loss """
    def __init__(self, task_loss_fn, aux_loss_fn, branch_loss_fn, aux_loss_weights: tuple = (0.6, 0.2, 0.2), **kwargs):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.aux_loss_fn = aux_loss_fn
        self.branch_loss_fn = branch_loss_fn
        self.aux_loss_weights = torch.tensor([aux_loss_weights]).float()

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        return [
            self.name,
            self.task_loss_fn.__class__.__name__,
            f"A.{self.aux_loss_fn.__class__.__name__}",
            f"B.{self.aux_loss_fn.__class__.__name__}",
        ]   

    def forward(self, task_aux_output, target):
        out, (aux, branch) = task_aux_output
        task_loss = self.task_loss_fn(out, target)
        if isinstance(task_loss, tuple):  # SOME LOSS FUNCTIONS RETURNS LOSS AND LOG_ITEMS
            task_loss = task_loss[0]
        aux_loss = self.aux_loss_fn(aux, target)
        branch_loss = self.branch_loss_fn(branch, target)
        
        loss = (self.aux_loss_weights.to(target.device) * torch.cat((task_loss.unsqueeze(0), aux_loss.unsqueeze(0), branch_loss.unsqueeze(0)))).sum()

        return loss, torch.cat((loss.unsqueeze(0), task_loss.unsqueeze(0), aux_loss.unsqueeze(0), branch_loss.unsqueeze(0))).detach()
    

class BranchVariableLoss(ComposedLoss):
    name = "BAuxLoss"
    """ Branch Auxiliary loss, wraps the task loss and auxiliary loss """
    def __init__(self, task_loss_fn, aux_loss_fn, branch_loss_fn, aux_loss_weights: tuple = (0.6, 0.2, 0.2), min_threshold=0.2, max_threshold=1, **kwargs):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.aux_loss_fn = aux_loss_fn
        self.branch_loss_fn = branch_loss_fn
        self.aux_loss_weights = torch.tensor([aux_loss_weights]).float()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        return [
            self.name,
            self.task_loss_fn.__class__.__name__,
            f"A.{self.aux_loss_fn.__class__.__name__}",
            f"B.{self.aux_loss_fn.__class__.__name__}",
            f"sc:{self.task_loss_fn.__class__.__name__}",
            f"sc:A.{self.aux_loss_fn.__class__.__name__}",
            f"sc:B.{self.aux_loss_fn.__class__.__name__}",
        ]   
    
    def _get_weight(self, loss, device):
        loss = max(self.min_threshold, loss)
        loss = min(self.max_threshold, loss)
        indicator = 1 - ((loss - self.min_threshold) / (self.max_threshold - self.min_threshold)) # scale in [0, 1]
        indicator = (torch.e - 1)*(indicator - 0) + 1 # scale in [1, e]
        return torch.log(torch.tensor(indicator, device=device)) * 3 # scale in [0, 3]
    
    def rescale_weights(self, loss, device):
        ww_branch = self._get_weight(loss, device)
        ww_aux = ww_task = (3 - ww_branch) / 2
        aux_loss_weights = self.aux_loss_weights.to(device)
        wws = torch.tensor([[ww_task, ww_aux, ww_branch]], device=device)
        weights = aux_loss_weights * wws
        return weights / weights.sum() 

    def forward(self, task_aux_output, target):
        out, (aux, branch) = task_aux_output
        task_loss = self.task_loss_fn(out, target)
        if isinstance(task_loss, tuple):  # SOME LOSS FUNCTIONS RETURNS LOSS AND LOG_ITEMS
            task_loss = task_loss[0]
        aux_loss = self.aux_loss_fn(aux, target)
        branch_loss = self.branch_loss_fn(branch, target)

        weights = self.rescale_weights(task_loss, target.device)
        
        loss_comp = (weights * torch.cat((task_loss.unsqueeze(0), aux_loss.unsqueeze(0), branch_loss.unsqueeze(0)))).squeeze(0)
        loss = loss_comp.sum()

        return loss, \
            torch.cat((loss.unsqueeze(0), task_loss.unsqueeze(0), aux_loss.unsqueeze(0), branch_loss.unsqueeze(0),
                       loss_comp[0].unsqueeze(0), loss_comp[1].unsqueeze(0), loss_comp[2].unsqueeze(0)
                      )).detach()