from typewire import as_type


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def test_caster_with_as_nonidempotent_init() -> None:
    p = Point(1.0, 2.0)
    assert as_type(p, Point) is p


def test_container_of_nonidempotent_init() -> None:
    p1 = Point(1.0, 2.0)
    p2 = Point(3.0, 4.0)

    q1, q2 = as_type([p1, p2], list[Point])
    assert q1 is p1
    assert q2 is p2
