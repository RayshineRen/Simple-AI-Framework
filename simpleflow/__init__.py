from .graph import Graph
from .operations import *
from .session import Session

# Create a default graph.
import builtins

DEFAULT_GRAPH = builtins.DEFAULT_GRAPH = Graph()
