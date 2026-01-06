from typing import Optional


class Graph:
    def __init__(self, nodes: list[tuple[str, str]]) -> None:
        self.nodes: list[tuple] = nodes

    def expand(self, node: str) -> list[str]:
        return [_node[1] for _node in self.nodes if _node[0] == node]


def dfs(graph: Graph, node: str, goal: str, discovered: Optional[set] = None) -> bool:
# def dfs(graph: Graph, node: str, goal: str, discovered: set = set()) -> bool:
    discovered = set() if discovered is None else discovered    # This line is unnecessary with the other function head.

    discovered.add(node)

    if node == goal:
        return True

    for _node in graph.expand(node):
        if _node not in discovered:
            return dfs(graph, _node, goal, discovered)

    return False


graph = Graph([
    ("A", "B"),
    ("A", "C"),
    ("A", "D"),
    ("B", "D"),
    ("B", "E"),
    ("C", "B"),
    ("C", "D"),
    ("D", "B"),
    ("D", "E"),
    ("F", "G"),
])


assert dfs(graph, "A", "E") is True     # There exists a path from A to E
assert dfs(graph, "A", "A") is True     # The start node is the target
assert dfs(graph, "A", "G") is False    # G is in a disjoint graph from A
assert dfs(graph, "A", "H") is False    # H is not in the Graph
assert dfs(graph, "D", "A") is False    # D and A are on the same graph but there exists no directed path from D to A
