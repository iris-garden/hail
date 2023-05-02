from dataclasses import dataclass, field, replace
from typing import Any, Optional


@dataclass
class Aes:
    x: Any = None
    y: Any = None
    rest: dict[str, Any] = field(default_factory=dict)


def aes(x: Any = None, y: Any = None, **kwargs: Any) -> Aes:
    return {**{"x": x, "y": y}, **kwargs}


@dataclass
class Geom:
    aes: Aes = field(default_factory=aes)


@dataclass
class GeomPoint(Geom):
    pass


def geom_point(aes: Aes = aes()):
    return GeomPoint(aes)


@dataclass
class GeomHistogram(Geom):
    pass


def geom_histogram(aes: Aes = aes()):
    return GeomHistogram(aes)


@dataclass
class GeomLine(Geom):
    pass


def geom_line(aes: Aes = aes()):
    return GeomHistogram(aes)


@dataclass
class Plot:
    data: Any = None
    aes: Aes = field(default_factory=aes)
    geoms: list[Geom] = field(default_factory=list)
    prev: Optional["Plot"] = None

    def __add__(self: "Plot", other: Any) -> "Plot":
        kwargs = None
        if isinstance(other, dict):
            kwargs = {"aes": {**self.aes, **other}}
        elif isinstance(other, Geom):
            kwargs = {"geoms": [*self.geoms, other]}

        if kwargs is not None:
            return replace(self, **kwargs, prev=self)
        else:
            return self


def undo(plot: Plot, *, depth: int = 1) -> Plot:
    curr = plot
    index = depth
    while curr.prev is not None and index > 0:
        curr = curr.prev
        index -= 1
    return curr


def ggplot(data, aes=aes()):
    return Plot(data, aes)


def show(plot):
    from pprint import pprint
    pprint(plot, width=1)


__all__ = [
    "aes",
    "geom_histogram",
    "geom_line",
    "geom_point",
    "ggplot",
    "show",
    "undo",
]
