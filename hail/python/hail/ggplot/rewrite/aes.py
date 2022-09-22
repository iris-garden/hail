from hail import literal
from hail.expr import Expression
from hail.ggplot.utils import frozen_dataclass

Mapping = dict[str, Expression]

def aes(x: Optional[Expression] = None, y: Optional[Expression] = None, **kwargs: Any) -> Mapping:
    return (
        { "x": x } if x is not None else {}
        | { "y": y } if y is not None else {}
        | { k: v if isinstance(v, Expression) else literal(v) for k, v in kwargs.items() }
    )
