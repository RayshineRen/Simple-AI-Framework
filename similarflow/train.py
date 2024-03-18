from queue import Queue
from .graph import Operation, Variable, Constant
from .gradients import _gradient_registry


def compute_gradients(loss):
    grad_table = {}  # 存放节点的梯度
    grad_table[loss] = 1

    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()

        # 该节点不是loss节点，先遍历进queue
        if node != loss:
            grad_table[node] = 0

            for output_node in node.output_nodes:
                lossgrad_wrt_output_node_output = grad_table[output_node]
                output_node_op_type = output_node.__class__
                bprop = _gradient_registry[output_node_op_type]

                lossgrads_wrt_output_node_inputs = bprop(
                    output_node, lossgrad_wrt_output_node_output
                )

                if len(output_node.input_nodes) == 1:
                    grad_table[node] += lossgrads_wrt_output_node_inputs
                else:
                    # 若一个节点有多个输出，则多个梯度求和
                    node_index_in_output_node_inputs = output_node.input_nodes.index(
                        node
                    )
                    lossgrad_wrt_node = lossgrads_wrt_output_node_inputs[
                        node_index_in_output_node_inputs
                    ]
                    grad_table[node] += lossgrad_wrt_node

        # 把节点存入到队列中
        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table


class GradientDescentOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, loss):
        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            def compute(self):
                grad_table = compute_gradients(loss)
                for node in grad_table:
                    if type(node) == Variable or type(node) == Constant:  # Constant?
                        grad = grad_table[node]
                        node.value -= learning_rate * grad

        return MinimizationOperation()
