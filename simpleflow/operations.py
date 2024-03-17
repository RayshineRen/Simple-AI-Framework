import numpy as np
from queue import Queue


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

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

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

    def compute_gradient(self, grad=None):
        x, y = [node.output_value for node in self.input_nodes]
        if grad is None:
            grad = np.ones_like(self.output_value)
        grad_wrt_x = grad
        while np.ndim(grad_wrt_x) > len(np.shape(x)):
            grad_wrt_x = np.sum(grad_wrt_x, axis=0)
        for axis, size in enumerate(np.shape(x)):
            if size == 1:
                grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)

        grad_wrt_y = grad
        while np.ndim(grad_wrt_y) > len(np.shape(y)):
            grad_wrt_y = np.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(np.shape(y)):
            if size == 1:
                grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)
        return [grad_wrt_x, grad_wrt_y]


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

    def compute_gradient(self, grad=None):
        """Compute the gradient of the multiplication operation."""
        x, y = [node.output_value for node in self.input_nodes]
        if grad is None:
            grad = np.ones_like(self.output_value)
        grad_wrt_x = grad * y
        while np.ndim(grad_wrt_x) > len(np.shape(x)):
            grad_wrt_x = np.sum(grad_wrt_x, axis=0)
        for axis, size in enumerate(np.shape(x)):
            if size == 1:
                grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)

        grad_wrt_y = grad * x
        while np.ndim(grad_wrt_y) > len(np.shape(y)):
            grad_wrt_y = np.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(np.shape(y)):
            if size == 1:
                grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)

        return [grad_wrt_x, grad_wrt_y]


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

    def compute_gradient(self, grad=None):
        x, y = [node.output_value for node in self.input_nodes]
        if grad is None:
            grad = np.ones_like(self.output_value)
        dfdx = np.dot(grad, np.transpose(y))
        dfdy = np.dot(np.transpose(x), grad)

        return [dfdx, dfdy]


def matmul(x, y, name=None):
    """Returns the matrix multiplication of two arrays."""
    return MatMul(x, y, name)


class Sigmoid(Operation):
    def __init__(self, x, name=None):
        """Sigmoid constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        """
        super().__init__(x, name=name)

    def compute_output(self):
        """Compute the output value of the sigmoid operation."""
        x = self.input_nodes[0]
        self.output_value = 1 / (1 + np.exp(-x.output_value))
        return self.output_value

    def compute_gradient(self, grad=None):
        """Compute the gradient of the sigmoid operation."""
        if grad is None:
            grad = np.ones_like(self.output_value)
        return grad * self.output_value * (1 - self.output_value)


def sigmoid(x, name=None):
    """Returns the sigmoid of the input."""
    return Sigmoid(x, name)


class Log(Operation):
    def __init__(self, x, name=None):
        """Logarithm constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        """
        super().__init__(x, name=name)

    def compute_output(self):
        """Compute the output value of the logarithm operation."""
        x = self.input_nodes[0]
        self.output_value = np.log(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        """Compute the gradient of the logarithm operation."""
        x = self.input_nodes[0].output_value
        if grad is None:
            grad = np.ones_like(x)
        return grad / x


def log(x, name=None):
    """Returns the natural logarithm of the input."""
    return Log(x, name)


class Negative(Operation):
    """Negative operation."""

    def __init__(self, x, name=None):
        """Negative constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        """
        super().__init__(x, name=name)

    def compute_output(self):
        """Compute the output value of the negative operation."""
        x = self.input_nodes[0]
        self.output_value = -x.output_value
        return self.output_value

    def compute_gradient(self, grad=None):
        """Compute the gradient of the negative operation."""
        if grad is None:
            grad = np.ones_like(self.input_nodes[0].output_value)
        return -grad


class ReduceSum(Operation):
    def __init__(self, x, axis=None, name=None):
        """Reduce sum constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param axis: The axis along which the sum is computed.
        :type axis: int.

        :param name: The operation name.
        :type name: str.
        """
        super().__init__(x, name=name)
        self.axis = axis

    def compute_output(self):
        """Compute the output value of the reduce sum operation."""
        x = self.input_nodes[0]
        self.output_value = np.sum(x.output_value, axis=self.axis)
        return self.output_value

    def compute_gradient(self, grad=None):
        """compute the gradient of the reduce sum operation."""
        input_value = self.input_nodes[0].output_value
        if grad is None:
            grad = np.ones_like(self.output_value)
        output_shape = np.array(np.shape(input_value))
        output_shape[self.axis] = 1.0
        # tile_scaling = np.true_divide(np.shape(input_value), output_shape)
        tile_scaling = np.shape(input_value) // output_shape
        grad = np.reshape(grad, output_shape)
        return np.tile(grad, tile_scaling)


def reduce_sum(x, axis=None):
    """Computes the sum of elements across dimensions of a tensor."""
    return ReduceSum(x, axis=axis)


class Square(Operation):
    """Square operation."""

    def __init__(self, x, name=None):
        """Square constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        """
        super().__init__(x, name=name)

    def compute_output(self):
        """Compute the output value of the square operation."""
        x = self.input_nodes[0]
        self.output_value = np.square(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        """Compute the gradient of the square operation."""
        input_value = self.input_nodes[0].output_value
        if grad is None:
            grad = np.ones_like(input_value)
        return grad * np.multiply(2.0, input_value)


def square(x, name=None):
    """Returns the element-wise square of the input."""
    return Square(x, name)


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

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)


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

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

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

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)


def placeholder(name=None):
    """Create a placeholder node."""
    return Placeholder(name=name)


def compute_gradients(target_op):
    """Backpropagation implementation computing gradient of target operation wrt
        all the other connected nodes.

    :param target_op: The target operation whose gradient wrt other nodes would
                      be computed.
    :type target_op: Any operation type.

    :return grad_table: A table containing node objects and gradients.
    :type grad_table: dict.
    """
    # A dict containing a mapping between node and gradient value of target_op wrt the node's output.
    # NOTE: It is the gradient wrt the node's OUTPUT NOT input.
    grad_table = {}
    # The gradient wrt target_op itself is 1.
    grad_table[target_op] = np.ones_like(target_op.output_value)
    # perform a breadth-first search to compute the gradients.
    queue = Queue()
    queue.put(target_op)
    # Set for visited nodes.
    visited = set()
    visited.add(target_op)

    while not queue.empty():
        node = queue.get()
        # Compute gradient wrt the node's output.
        if node != target_op:
            grads_wrt_node_output = []
            for output_node in node.output_nodes:
                # Retrieve the gradient wrt output_node's OUTPUT.
                grad_wrt_output_node_output = grad_table[output_node]
                # Compute the gradient wrt current node's output.
                grad_wrt_node_output = output_node.compute_gradient(
                    grad_wrt_output_node_output
                )
                if len(output_node.input_nodes) > 1:
                    input_node_index = output_node.input_nodes.index(node)
                    grads_wrt_node_output.append(grad_wrt_node_output[input_node_index])
                else:
                    grads_wrt_node_output.append(grad_wrt_node_output)

            # Sum all the gradients wrt node's output.
            tot_grad_wrt_node_output = sum(grads_wrt_node_output)
            grad_table[node] = tot_grad_wrt_node_output

        # Put adjecent nodes to queue
        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)
    return grad_table
