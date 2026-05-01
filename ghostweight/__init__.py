from .ghostgate import GhostGate, apply_ghostgate, measure_sparsity

def ghost_surgery(model, threshold=0.05):
    model, replaced = apply_ghostgate(model, threshold), 0
    return model, replaced

__version__ = "0.1.0"
__author__ = "Manjit Pokhrel"