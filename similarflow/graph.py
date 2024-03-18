class Graph(object):
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []
        self.constants = []

    def __enter__(self):
        global _default_graph
        self.graph = _default_graph
        _default_graph = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _default_graph
        _default_graph = self.graph

    def as_default(self):
        return self


class Operation(object):
    def __init__(self, *input_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

    def compute(self):
        pass

    def __add__(self, other):
        from .operations import add

        return add(self, other)

    def __neg__(self):
        from .operations import negative

        return negative(self)

    def __sub__(self, other):
        from .operations import add, negative

        return add(self, negative(other))

    def __mul__(self, other):
        from .operations import matmul

        return matmul(self, other)


class Placeholder(object):
    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)

    def __add__(self, other):
        from .operations import add

        return add(self, other)

    def __neg__(self):
        from .operations import negative

        return negative(self)

    def __sub__(self, other):
        from .operations import add, negative

        return add(self, negative(other))

    def __mul__(self, other):
        from .operations import matmul

        return matmul(self, other)


class Variable(object):
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        _default_graph.variables.append(self)

    def __add__(self, other):
        from .operations import add

        return add(self, other)

    def __neg__(self):
        from .operations import negative

        return negative(self)

    def __sub__(self, other):
        from .operations import add, negative

        return add(self, negative(other))

    def __mul__(self, other):
        from .operations import matmul

        return matmul(self, other)


class Constant(object):
    def __init__(self, value):
        self.value = value
        self.output_nodes = []
        _default_graph.constants.append(self)

    def __add__(self, other):
        from .operations import add

        return add(self, other)

    def __neg__(self):
        from .operations import negative

        return negative(self)

    def __sub__(self, other):
        from .operations import add, negative

        return add(self, negative(other))

    def __mul__(self, other):
        from .operations import matmul

        return matmul(self, other)
