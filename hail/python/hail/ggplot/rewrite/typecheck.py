from functools import wraps
from inspect import getfullargspec

# TODO make into pytests

"""
# examples

@typecheck
class Foo(NamedTuple):
    bar: str
    baz: int
    bit: bool

@typecheck
def foo(bar: str, baz: int, bit: bool) -> (str, int, bool):
    return (bar, baz, bit)

# no error
Foo("1", 1, True)
Foo("1", bit=True, baz=1)
Foo(bit=True, bar="1", baz=1)
foo("1", 1, True)
foo("1", bit=True, baz=1)
foo(bit=True, bar="1", baz=1)

# error
Foo(1, 1, 1)
Foo("1", 1, 1)
Foo("1", bit=1, baz=True)
foo(1, 1, 1)
foo("1", 1, 1)
foo("1", bit=1, baz=True)
"""

# TODO this should handle e.g. the case where the function takes **kwargs as input but doesn't give individual kwargs specific types
# TODO fix for dataclasses
def typecheck(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        arg_names = func._fields if getattr(func, "_fields", None) is not None else getfullargspec(func).args
        for kw, arg in ({ k: v for k, v in zip(arg_names, args) } | kwargs).items():
            arg_type = func.__annotations__[kw]
            if not isinstance(arg, arg_type):
                raise TypeError(f"{func.__name__}: Argument '{kw}' is of type '{arg_type.__name__}', but got value '{arg}' of type '{type(arg).__name__}'.")
        return func(*args, **kwargs)
    return wrapper
