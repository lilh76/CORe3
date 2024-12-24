from .mlp_model import MLPModelEnsemble
from .table_model import TableModel

REGISTRY = {}

REGISTRY["mlp"] = MLPModelEnsemble
REGISTRY['table'] = TableModel