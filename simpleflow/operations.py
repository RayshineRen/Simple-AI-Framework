import numpy as np


class Operation(object):
    """base class for an operation in the computational graph"""

    def __init__(self, *input_nodes, name=None):
        """
        :param input_nodes: input nodes to the operation
        :type input_nodes: Objects of `Operation`, `Variable` or `Placeholder`

        :param name: name of the operation
        :type name: str
        """
        # Nodes received by this operation.
        self.input_nodes = input_nodes
        # Nodes that receive this operation node as input.
        self.output_nodes = []
        # Output value of this operation in session execution.
        self.output_value = None
        self.name = name
        # Graph the operation belongs to.
        self.graph = DEFAULT_GRAPH
        # Add this operation node to destination lists in its input nodes.
        for node in input_nodes:
            node.output_nodes.append(self)
        # Add this operation to the graph.
        self.graph.operations.append(self)

    def compute_output(self):
        """Compute the output value of the operation."""
        raise NotImplementedError

    def compute_gradient(self, grad=None):
        """Compute the gradient of the operation."""
        raise NotImplementedError

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multiply(self, other)


class Add(Operation):
    def __init__(self, x, y, name=None):
        """Addition constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        """
        super().__init__(x, y, name=name)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.add(x.output_value, y.output_value)
        return self.output_value


def add(x, y, name=None):
    """Returns x + y element-wise."""
    return Add(x, y, name)


class Multiply(Operation):
    """Multiplication operation."""

    def __init__(self, x, y, name=None):
        """Multiplication constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        """
        super().__init__(x, y, name=name)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.multiply(x.output_value, y.output_value)
        return self.output_value


def multiply(x, y, name=None):
    """Returns x * y element-wise."""
    return Multiply(x, y, name)


class MatMul(Operation):
    """Matrix multiplication operation."""

    def __init__(self, x, y, name=None):
        """Matrix multiplication constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        """
        super().__init__(x, y, name=name)

    def compute_output(self):
        """Compute the output value of the matrix multiplication operation."""
        x, y = self.input_nodes
        self.output_value = np.dot(x.output_value, y.output_value)
        return self.output_value


def matmul(x, y, name=None):
    """Returns the matrix multiplication of two arrays."""
    return MatMul(x, y, name)


class Variable(object):
    """Variable node in computational graph."""

    def __init__(self, initial_value=None, name=None, trainable=True):
        """Variable constructor.

        :param initial_value: The initial value of the variable.
        :type initial_value: number or a ndarray.

        :param name: Name of the variable.
        :type name: str.
        """
        self.initial_value = initial_value
        # Output value of this operation in session execution.
        self.output_value = None
        # Nodes that receive this variable node as input.
        self.output_nodes = []
        self.name = name
        # Graph the variable belongs to.
        self.graph = DEFAULT_GRAPH
        # Add this variable to the graph.
        self.graph.variables.append(self)
        if trainable:
            self.graph.trainable_variables.append(self)

    def compute_output(self):
        """Compute the output value of the variable."""
        if self.output_value is None:
            self.output_value = self.initial_value
        return self.output_value


class Constant(object):
    """Constant node in computational graph."""

    def __init__(self, value, name=None):
        # Constant value.
        self.value = value
        # Output value of this operation in session.
        self.output_value = None
        # Nodes that receive this variable node as input.
        self.output_nodes = []
        self.name = name
        # Graph the constant belongs to.
        self.graph = DEFAULT_GRAPH
        # Add this constant to the graph.
        self.graph.constants.append(self)

    def compute_output(self):
        """Compute the output value of the constant."""
        if self.output_value is None:
            self.output_value = self.value
        return self.output_value

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multiply(self, other)


def constant(value, name=None):
    """Create a constant node."""
    return Constant(value, name=name)


class Placeholder(object):
    """Placeholder node in computational graph. It has to be provided a value when
    when computing the output of a graph.
    """

    def __init__(self, name=None):
        """Placeholder node in computational graph. It has to be provided a value when
        when computing the output of a graph.
        """
        # Output value of this operation in session execution.
        self.output_value = None
        # Nodes that receive this placeholder node as input.
        self.output_nodes = []
        self.name = name
        # Graph the placeholder node belongs to.
        self.graph = DEFAULT_GRAPH
        # Add to the currently active default graph.
        self.graph.placeholders.append(self)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multiply(self, other)


def placeholder(name=None):
    """Create a placeholder node."""
    return Placeholder(name=name)
