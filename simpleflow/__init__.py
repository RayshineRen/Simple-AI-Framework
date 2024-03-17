from .graph import Graph
from .operations import *
from .session import Session
from .train import GradientDescentOptimizer

# Create a default graph.
import builtins

DEFAULT_GRAPH = builtins.DEFAULT_GRAPH = Graph()
