def aes(x=None, y=None, **kwargs):
    return ("aes", x, y, kwargs)


def ggplot(data, aes=aes()):
    return ("ggplot", data, aes)


def geom_point(aes):
    return ("geom_point", aes)


def geom_histogram():
    return ("geom_histogram",)


def show(plot):
    from pprint import pprint
    pprint(plot, width=1)


__all__ = [
    "ggplot",
    "aes",
    "geom_point",
    "geom_histogram",
    "show",
]
