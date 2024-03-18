import similarflow.train
from .graph import Operation, Variable, Graph, Placeholder, Constant
from .operations import *
from .session import Session
from .gradients import RegisterGradient

import builtins

_default_graph = builtins._default_graph = Graph()

__all__ = [
    "Operation",
    "Variable",
    "Graph",
    "Placeholder",
    "Constant",
    "matmul",
    "add",
    "negative",
    "multiply",
    "sigmoid",
    "softmax",
    "log",
    "Session",
    "RegisterGradient",
]
