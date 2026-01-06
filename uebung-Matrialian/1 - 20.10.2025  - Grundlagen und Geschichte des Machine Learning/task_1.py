from typing import Optional


class Class_A:
    def __init__(self, param_A: list[tuple], param_B: dict[str, float]) -> None:
        self.attr_A: list[tuple] = param_A
        self.attr_B: dict[str, float] = param_B

    def meth_A(self, param_A: str) -> float:
        return self.attr_B[param_A] if param_A in self.attr_B else float("inf")

    def meth_B(self, param_A: tuple[str]) -> float:
        if len(param_A) < 2:
            return 0
        return (
            self.meth_B(param_A[1:])
            + next(
                local_A
                for local_A in self.attr_A
                if local_A[0] == param_A[0] and local_A[1] == param_A[1]
            )[2]
        )

    def meth_C(self, param_A: str) -> list[str]:
        return [local_A[1] for local_A in self.attr_A if local_A[0] == param_A]

    def meth_D(
        self, param_A: list[tuple], param_B: bool = True, param_C: bool = True
    ) -> None:
        print(
            "(",
            " ".join(
                f"{''.join(local_A)}{'.' if param_B or param_C else ''}{str(self.meth_B(local_A)) if param_B else ''}{'+' if param_B and param_C else ''}{str(self.meth_A(local_A[-1])) if param_C else ''}"
                for local_A in param_A
            ),
            ")",
        )


def func_A(param_A: str, param_B: str, param_C: Class_A) -> Optional[tuple]:
    local_A: list[tuple] = [(param_A,)]
    local_B: set[str] = set()

    while local_A:
        param_C.meth_D(local_A)

        local_C: tuple = local_A.pop(0)
        local_D: str = local_C[-1]

        if local_D == param_B:
            return local_C

        local_B.add(local_D)
        for local_E in param_C.meth_C(local_D):
            if local_E not in local_B and local_E not in map(
                lambda local_F: local_F[-1], local_A
            ):
                local_A.append(local_C + (local_E,))
                continue
            elif local_E in map(lambda local_G: local_G[-1], local_A):
                local_H = next(local_I for local_I in local_A if local_I[-1] == local_E)
                if param_C.meth_B(local_C + (local_E,)) < param_C.meth_B(local_H):
                    local_A.remove(local_H)
                    local_A.append(local_C + (local_E,))
                    continue

        local_A.sort(
            key=lambda local_J: param_C.meth_B(local_J) + param_C.meth_A(local_J[-1])
        )
    return None


obj_A: Class_A = Class_A(
    param_A=[
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
    param_B={
        "A": 30,
        "B": 25,
        "C": 20,
        "D": 15,
        "E": 0,
    },
)
