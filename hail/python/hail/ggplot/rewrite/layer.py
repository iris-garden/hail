from hail import Table
from hail.ggplot.aes import Mapping
from hail.ggplot.geom import Geom
from hail.ggplot.position import Position
from hail.ggplot.stat import Stat
from hail.ggplot.utils import frozen_dataclass

@frozen_dataclass
class Layer:
    data: Table
    mapping: Mapping
    geom: Geom
    stat: Stat
    position: Position
