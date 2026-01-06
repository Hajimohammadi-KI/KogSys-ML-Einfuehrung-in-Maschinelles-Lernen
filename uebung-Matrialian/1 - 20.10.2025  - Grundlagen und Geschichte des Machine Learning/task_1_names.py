from typing import Optional


class Graph:
    def __init__(self, edges: list[tuple], heuristic: dict[str, float]) -> None:
        self.edges: list[tuple] = edges
        self.heuristic: dict[str, float] = heuristic

    def h(self, node: str) -> float:
        return self.heuristic[node] if node in self.heuristic else float("inf")

    def g(self, path: tuple[str]) -> float:
        if len(path) < 2:
            return 0
        return (
            self.g(path[1:])
            + next(
                edge
                for edge in self.edges
                if edge[0] == path[0] and edge[1] == path[1]
            )[2]
        )

    def expand(self, node: str) -> list[str]:
        return [edge[1] for edge in self.edges if edge[0] == node]

    def print_queue(
        self, queue: list[tuple], with_cost: bool = True, with_heuristic: bool = True
    ) -> None:
        print(
            "(",
            " ".join(
                f"{''.join(path)}{'.' if with_cost or with_heuristic else ''}{str(self.g(path)) if with_cost else ''}{'+' if with_cost and with_heuristic else ''}{str(self.h(path[-1])) if with_heuristic else ''}"
                for path in queue
            ),
            ")",
        )


def a_star(start: str, goal: str, graph: Graph) -> Optional[tuple]:
    queue: list[tuple] = [(start,)]
    visited: set[str] = set()

    while queue:
        graph.print_queue(queue)

        current_path: tuple = queue.pop(0)
        current_node: str = current_path[-1]

        if current_node == goal:
            return current_path

        visited.add(current_node)
        for node in graph.expand(current_node):
            if node not in visited and node not in map(
                lambda x: x[-1], queue
            ):
                queue.append(current_path + (node,))
                continue
            elif node in map(lambda x: x[-1], queue):
                local_H = next(local_I for local_I in queue if local_I[-1] == node)
                if graph.g(current_path + (node,)) < graph.g(local_H):
                    queue.remove(local_H)
                    queue.append(current_path + (node,))
                    continue

        queue.sort(
            key=lambda x: graph.g(x) + graph.h(x[-1])
        )
    return None


graph: Graph = Graph(
    edges=[
        ("A", "B", 10),
        ("A", "C", 5),
        ("A", "D", 15),
        ("B", "D", 9),
        ("B", "E", 30),
        ("C", "B", 15),
        ("C", "D", 9),
        ("D", "B", 10),
        ("D", "E", 20),
    ],
    heuristic={
        "A": 30,
        "B": 25,
        "C": 20,
        "D": 15,
        "E": 0,
    },
)
