import abc
from dataclasses import dataclass, asdict
import itertools
import math
from pprint import pprint
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from plotly.subplots import make_subplots

import hail as hl
from hail import literal
from hail import tstr
from hail import Table
from hail.context import get_reference
from hail.expr import Expression
from hail.expr import StructExpression
from hail.utils.java import warning


# dataclass utils ---------------------------------------------------------------------------------
frozen_dataclass = dataclass(frozen=True)


# TODO type params
def add_fields(base, fields):
    return base.__class__(**{**asdict(base), **(fields if isinstance(fields, dict) else as_nonempty_dict(fields))})


def as_nonempty_dict(data: Any) -> dict:
    return asdict(data, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})


# refactored aes ----------------------------------------------------------------------------------
Aesthetic = dict[str, Expression]


def aes(x: Any = None, y: Any = None, **kwargs: Any) -> Aesthetic:
    return {
        **({"x": x} if x is not None else {}),
        **({"y": y} if y is not None else {}),
        **{k: v if isinstance(v, Expression) else literal(v) for k, v in kwargs.items()}
    }


# refactored dataclasses --------------------------------------------------------------------------
@frozen_dataclass
class Labels:
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None


@frozen_dataclass
class CoordCartesian:
    xlim: Optional[Tuple[int, int]] = None
    ylim: Optional[Tuple[int, int]] = None


@frozen_dataclass
class FacetWrap:
    facets: StructExpression


def add_to_plot(plot, other):
    fields = None
    for typ, get_kwargs in [
        (dict, lambda plot, other: {"aes": aes({**plot.aes, **other})}),
        (CoordCartesian, lambda plot, other: {"coord_cartesian": other}),
        (FacetWrap, lambda plot, other: {"facet": other}),
        # TODO
        (Geom, lambda plot, other: {"geoms": plot.geoms + [other], "scales": add_default_scales(...)}),
        (Labels, lambda plot, other: {"labels": add_fields(plot.labels, other)}),
        (Scale, lambda plot, other: {"scales": {**plot.scales, other.aesthetic_name: other}}),
    ]:
        if isinstance(other, typ):
            fields = get_kwargs(other)
            break
    if fields is None:
        # TODO better error
        raise ValueError("not implemented")
    else:
        return add_fields(plot, fields)


@frozen_dataclass
class Plot:
    # TODO base class that could be table or matrixtable?
    ht: Table
    aes: Aesthetic
    geoms: list[Geom] = []
    labels: Labels = Labels()
    coord_cartesian: Optional[CoordCartesian] = None
    scales: dict[str, Scale] = {}
    facet: Optional[FacetWrap] = None

    # TODO
    # def __new__():
    #     self.scales = add_default_scales(plot, aes)

    __add__ = add_to_plot


# api ---------------------------------------------------------------------------------------------
def ggplot(table, mapping: dict[str, Any] = {}):
    return Plot(table, aes(**mapping))


def coord_cartesian(xlim=None, ylim=None):
    return CoordCartesian(xlim, ylim)


def facet_wrap(facets):
    return FacetWrap(facets)


def geom_area(mapping=aes(), fill=None, color=None):
    return GeomArea(mapping, fill=fill, color=color)


def geom_bar(mapping=aes(), *, fill=None, color=None, alpha=None, position="stack", size=None):
    return GeomBar(mapping, fill=fill, color=color, alpha=alpha, position=position, size=size)


def geom_col(mapping=aes(), *, fill=None, color=None, alpha=None, position="stack", size=None):
    return GeomBar(mapping, stat=StatIdentity(), fill=fill, color=color, alpha=alpha, position=position, size=size)


def geom_density(mapping=aes(), *, k=1000, smoothing=0.5, fill=None, color=None, alpha=None):
    return GeomDensity(mapping, k, smoothing, fill, color, alpha)


def geom_func(mapping=aes(), fun=None, color=None):
    return GeomFunction(mapping, fun=fun, color=color)


def geom_histogram(mapping=aes(), *, min_val=None, max_val=None, bins=None, fill=None, color=None, alpha=None, position='stack', size=None):
    return GeomHistogram(mapping, min_val=min_val, max_val=max_val, bins=bins, fill=fill, color=color, alpha=alpha, position=position, size=size)


def geom_hline(yintercept, *, linetype="solid", color=None):
    return GeomHLine(yintercept, linetype=linetype, color=color)


def geom_line(mapping=aes(), *, color=None, size=None, alpha=None):
    return GeomLine(mapping, color=color)


def geom_point(mapping=aes(), *, color=None, size=None, alpha=None):
    return GeomPoint(mapping, color=color, size=size, alpha=alpha)


def geom_ribbon(mapping=aes(), fill=None, color=None):
    return GeomRibbon(mapping, fill=fill, color=color)


def geom_text(mapping=aes(), *, color=None, size=None, alpha=None):
    return GeomText(mapping, color=color, size=size, alpha=alpha)


def geom_tile(mapping=aes()):
    return GeomTile(mapping)


def geom_vline(xintercept, *, linetype="solid", color=None):
    return GeomVLine(xintercept, linetype=linetype, color=color)


def ggtitle(label):
    return Labels(title=label)


def scale_color_continuous():
    return ScaleColorContinuous("color")


def scale_color_discrete():
    return scale_color_hue()


def scale_color_hue():
    return ScaleColorHue("color")


def scale_color_identity():
    return ScaleColorContinuousIdentity("color")


def scale_color_manual(*, values):
    return ScaleColorManual("color", values=values)


def scale_fill_continuous():
    return ScaleColorContinuous("fill")


def scale_fill_discrete():
    return scale_fill_hue()


def scale_fill_hue():
    return ScaleColorHue("fill")


def scale_fill_identity():
    return ScaleColorContinuousIdentity("fill")


def scale_fill_manual(*, values):
    return ScaleColorManual("fill", values=values)


def scale_x_continuous(name=None, breaks=None, labels=None, trans="identity"):
    return PositionScaleContinuous("x", name=name, breaks=breaks, labels=labels, transformation=trans)


def scale_x_discrete(name=None, breaks=None, labels=None):
    return PositionScaleDiscrete("x", name=name, breaks=breaks, labels=labels)


def scale_x_genomic(reference_genome, name=None):
    return PositionScaleGenomic("x", reference_genome, name=name)


def scale_x_log10(name=None):
    return PositionScaleContinuous("x", name=name, transformation="log10")


def scale_x_reverse(name=None):
    return PositionScaleContinuous("x", name=name, transformation="reverse")


def scale_y_continuous(name=None, breaks=None, labels=None, trans="identity"):
    return PositionScaleContinuous("y", name=name, breaks=breaks, labels=labels, transformation=trans)


def scale_y_discrete(name=None, breaks=None, labels=None):
    return PositionScaleDiscrete("y", name=name, breaks=breaks, labels=labels)


def scale_y_log10(name=None):
    return PositionScaleContinuous("y", name=name, transformation="log10")


def scale_y_reverse(name=None):
    return PositionScaleContinuous("y", name=name, transformation="reverse")


def vars(*args):
    return hl.struct(**{f"var_{i}": arg for i, arg in enumerate(args)})


def xlab(label):
    return Labels(xlabel=label)


def ylab(label):
    return Labels(ylabel=label)


# old classes -------------------------------------------------------------------------------------
@frozen_dataclass
class Scale:
    aesthetic_name: str

    @abc.abstractmethod
    def transform_data(self, field_expr):
        pass

    def create_local_transformer(self, groups_of_dfs):
        return lambda x: x

    @abc.abstractmethod
    def is_discrete(self):
        pass

    @abc.abstractmethod
    def is_continuous(self):
        pass

    def valid_dtype(self, dtype):
        pass


# TODO
def create_local_transformer(scale, groups_of_dfs):
    if isinstance(scale, ScaleColorManual):
        def categorical_strings_to_colors(string_set, color_values):
            if isinstance(color_values, list):
                if len(string_set) > len(color_values):
                    print(f"Not enough colors specified. Found {len(string_set)} distinct values of color aesthetic and only {len(color_values)} colors were provided.")
                color_dict = {}
                for idx, element in enumerate(string_set):
                    if element not in color_dict:
                        color_dict[element] = color_values[idx]
            else:
                color_dict = color_values
            return color_dict
        categorical_strings = set()
        for group_of_dfs in groups_of_dfs:
            for df in group_of_dfs:
                if scale.aesthetic_name in df.attrs:
                    categorical_strings.add(df.attrs[scale.aesthetic_name])

        unique_color_mapping = categorical_strings_to_colors(categorical_strings, scale.values)

        def transform(df):
            df.attrs[f"{scale.aesthetic_name}_legend"] = df.attrs[scale.aesthetic_name]
            df.attrs[scale.aesthetic_name] = unique_color_mapping[df.attrs[scale.aesthetic_name]]
            return df

        return transform
    else:
        return lambda x: x


class PositionScale(Scale):
    def __init__(self, aesthetic_name, name, breaks, labels):
        super().__init__(aesthetic_name)
        self.name = name
        self.breaks = breaks
        self.labels = labels

    def update_axis(self, fig):
        if self.aesthetic_name == "x":
            return fig.update_xaxes
        elif self.aesthetic_name == "y":
            return fig.update_yaxes

    # What else do discrete and continuous scales have in common?
    def apply_to_fig(self, parent, fig_so_far):
        if self.name is not None:
            self.update_axis(fig_so_far)(title=self.name)
        if self.breaks is not None:
            self.update_axis(fig_so_far)(tickvals=self.breaks)
        if self.labels is not None:
            self.update_axis(fig_so_far)(ticktext=self.labels)

    def valid_dtype(self, dtype):
        return True


class PositionScaleGenomic(PositionScale):
    def __init__(self, aesthetic_name, reference_genome, name=None):
        super().__init__(aesthetic_name, name, None, None)
        if isinstance(reference_genome, str):
            reference_genome = get_reference(reference_genome)
        self.reference_genome = reference_genome

    def apply_to_fig(self, parent, fig_so_far):
        contig_offsets = dict(list(self.reference_genome.global_positions_dict.items())[:24])
        breaks = list(contig_offsets.values())
        labels = list(contig_offsets.keys())
        self.update_axis(fig_so_far)(tickvals=breaks, ticktext=labels)

    def transform_data(self, field_expr):
        return field_expr.global_position()

    def is_discrete(self):
        return False

    def is_continuous(self):
        return False


class PositionScaleContinuous(PositionScale):

    def __init__(self, axis=None, name=None, breaks=None, labels=None, transformation="identity"):
        super().__init__(axis, name, breaks, labels)
        self.transformation = transformation

    def apply_to_fig(self, parent, fig_so_far):
        super().apply_to_fig(parent, fig_so_far)
        if self.transformation == "identity":
            pass
        elif self.transformation == "log10":
            self.update_axis(fig_so_far)(type="log")
        elif self.transformation == "reverse":
            self.update_axis(fig_so_far)(autorange="reversed")
        else:
            raise ValueError(f"Unrecognized transformation {self.transformation}")

    def transform_data(self, field_expr):
        return field_expr

    def is_discrete(self):
        return False

    def is_continuous(self):
        return True


class PositionScaleDiscrete(PositionScale):
    def __init__(self, axis=None, name=None, breaks=None, labels=None):
        super().__init__(axis, name, breaks, labels)

    def apply_to_fig(self, parent, fig_so_far):
        super().apply_to_fig(parent, fig_so_far)

    def transform_data(self, field_expr):
        return field_expr

    def is_discrete(self):
        return True

    def is_continuous(self):
        return False


class ScaleContinuous(Scale):
    def __init__(self, aesthetic_name):
        super().__init__(aesthetic_name)

    def transform_data(self, field_expr):
        return field_expr

    def is_discrete(self):
        return False

    def is_continuous(self):
        return True

    def valid_dtype(self, dtype):
        return dtype in [hl.tint32, hl.tint64, hl.tfloat32, hl.tfloat64]


class ScaleDiscrete(Scale):
    def __init__(self, aesthetic_name):
        super().__init__(aesthetic_name)

    def transform_data(self, field_expr):
        return field_expr

    def is_discrete(self):
        return True

    def is_continuous(self):
        return False

    def valid_dtype(self, dtype):
        def is_discrete_type(dtype):
            return dtype in [hl.tstr]
        return is_discrete_type(dtype)


class ScaleColorManual(ScaleDiscrete):

    def __init__(self, aesthetic_name, values):
        super().__init__(aesthetic_name)
        self.values = values


class ScaleColorContinuous(ScaleContinuous):

    def create_local_transformer(self, groups_of_dfs):
        overall_min = None
        overall_max = None
        for group_of_dfs in groups_of_dfs:
            for df in group_of_dfs:
                if self.aesthetic_name in df.columns:
                    series = df[self.aesthetic_name]
                    series_min = series.min()
                    series_max = series.max()
                    if overall_min is None:
                        overall_min = series_min
                    else:
                        overall_min = min(series_min, overall_min)

                    if overall_max is None:
                        overall_max = series_max
                    else:
                        overall_max = max(series_max, overall_max)

        def transform(df):
            df[self.aesthetic_name] = df[self.aesthetic_name].map(lambda input_color: plotly.colors.sample_colorscale(plotly.colors.sequential.Viridis, (input_color - overall_min) / overall_max - overall_min)[0])
            return df

        return transform


class ScaleColorHue(ScaleDiscrete):
    def create_local_transformer(self, groups_of_dfs):
        categorical_strings = set()
        for group_of_dfs in groups_of_dfs:
            for df in group_of_dfs:
                if self.aesthetic_name in df.attrs:
                    categorical_strings.add(df.attrs[self.aesthetic_name])

        num_categories = len(categorical_strings)
        step = 1.0 / num_categories
        interpolation_values = [step * i for i in range(num_categories)]
        hsv_scale = px.colors.get_colorscale("HSV")
        colors = px.colors.sample_colorscale(hsv_scale, interpolation_values)
        unique_color_mapping = dict(zip(categorical_strings, colors))

        def transform(df):
            df.attrs[f"{self.aesthetic_name}_legend"] = df.attrs[self.aesthetic_name]
            df.attrs[self.aesthetic_name] = unique_color_mapping[df.attrs[self.aesthetic_name]]
            return df

        return transform


class ScaleColorContinuousIdentity(ScaleContinuous):
    def valid_dtype(self, dtype):
        return dtype == tstr


def should_use_scale_for_grouping(scale):
    return (scale.aesthetic_name not in {"x", "tooltip", "label"}) and scale.is_discrete()


def get_precomputes(scale, mapping):
    if isinstance(scale, StatBin):
        return hl.struct(**{key: func(mapping.x) for key, func in [("min_val", hl.agg.min), ("max_val", hl.agg.max)] if scale.getattr(key) is None})
    else:
        return hl.struct()


def partition_dict(data, func):
    result = [{}, {}]
    for k, v in data.items():
        idx = int(not func(k, v))
        result[idx] = {**result[idx], k: v}
    return result


class Stat:
    @abc.abstractmethod
    def make_agg(self, mapping, precomputed, scales):
        pass

    @abc.abstractmethod
    def listify(self, agg_result):
        # Turns the agg result into a list of data frames to be plotted.
        pass


class StatIdentity(Stat):
    def make_agg(self, mapping, precomputed, scales):
        grouping, non_grouping = [hl.struct(**data) for data in partition_dict(mapping, lambda k, v: should_use_scale_for_grouping(scales[k]))]
        return hl.agg.group_by(grouping, hl.agg.collect(non_grouping))

    def listify(self, agg_result):
        result = []
        for grouped_struct, collected in agg_result.items():
            columns = list(collected[0].keys())
            data_dict = {}

            for column in columns:
                col_data = [row[column] for row in collected]
                data_dict[column] = pd.Series(col_data)

            df = pd.DataFrame(data_dict)
            df.attrs.update(**grouped_struct)
            result.append(df)
        return result


class StatFunction(StatIdentity):
    def __init__(self, fun):
        self.fun = fun

    def make_agg(self, mapping, precomputed, scales):
        with_y_value = mapping.annotate(y=self.fun(mapping.x))
        return super().make_agg(with_y_value, precomputed, scales)


class StatNone(Stat):
    def make_agg(self, mapping, precomputed, scales):
        return hl.agg.take(hl.struct(), 0)

    def listify(self, agg_result):
        return pd.DataFrame({})


class StatCount(Stat):
    def make_agg(self, mapping, precomputed, scales):
        grouping_variables = {aes_key: mapping[aes_key] for aes_key in mapping.keys() if should_use_scale_for_grouping(scales[aes_key])}
        if "weight" in mapping:
            return hl.agg.group_by(hl.struct(**grouping_variables), hl.agg.counter(mapping["x"], weight=mapping["weight"]))
        return hl.agg.group_by(hl.struct(**grouping_variables), hl.agg.group_by(mapping["x"], hl.agg.count()))

    def listify(self, agg_result):
        result = []
        for grouped_struct, count_by_x in agg_result.items():
            data_dict = {}
            xs, counts = zip(*count_by_x.items())
            data_dict["x"] = pd.Series(xs)
            data_dict["y"] = pd.Series(counts)

            df = pd.DataFrame(data_dict)
            df.attrs.update(**grouped_struct)
            result.append(df)

        return result


class StatBin(Stat):
    DEFAULT_BINS = 30

    def __init__(self, min_val, max_val, bins):
        self.min_val = min_val
        self.max_val = max_val
        self.bins = bins

    def make_agg(self, mapping, precomputed, scales):
        grouping_variables = {aes_key: mapping[aes_key] for aes_key in mapping.keys() if should_use_scale_for_grouping(scales[aes_key])}
        start = self.min_val if self.min_val is not None else precomputed.min_val
        end = self.max_val if self.max_val is not None else precomputed.max_val
        if self.bins is None:
            warning(f"No number of bins was specfied for geom_histogram, defaulting to {self.DEFAULT_BINS} bins")
            bins = self.DEFAULT_BINS
        else:
            bins = self.bins
        return hl.agg.group_by(hl.struct(**grouping_variables), hl.agg.hist(mapping["x"], start, end, bins))

    def listify(self, agg_result):
        items = list(agg_result.items())
        x_edges = items[0][1].bin_edges
        num_edges = len(x_edges)

        result = []

        for grouped_struct, hist in items:
            data_rows = []
            y_values = hist.bin_freq
            for i, x in enumerate(x_edges[:num_edges - 1]):
                data_rows.append({"x": x, "y": y_values[i]})
            df = pd.DataFrame.from_records(data_rows)
            df.attrs.update(**grouped_struct)
            result.append(df)
        return result


class StatCDF(Stat):
    def __init__(self, k):
        self.k = k

    def make_agg(self, mapping, precomputed, scales):
        grouping_variables = {aes_key: mapping[aes_key] for aes_key in mapping.keys() if should_use_scale_for_grouping(scales[aes_key])}
        return hl.agg.group_by(hl.struct(**grouping_variables), hl.agg.approx_cdf(mapping["x"], self.k))

    def listify(self, agg_result):
        result = []

        for grouped_struct, data in agg_result.items():
            n = data['ranks'][-1]
            weights = np.diff(data['ranks'][1:-1])
            min = data['values'][0]
            max = data['values'][-1]
            values = np.array(data['values'][1:-1])
            df = pd.DataFrame({'value': values, 'weight': weights})
            df.attrs.update(**grouped_struct)
            df.attrs.update({'min': min, 'max': max, 'n': n})

            result.append(df)
        return result


def get_trace_args(geom, df, facet_row, facet_col):
    base = {"x": df.x, "y": df.y, "row": facet_row, "col": facet_col}
    if isinstance(geom, (GeomLineBasic, GeomLine)):
        return {**base, "mode": "lines"}
    elif isinstance(geom, GeomPoint):
        return {**base, "mode": "markers"}
    elif isinstance(geom, GeomText):
        return {**base, "text": df.label, "mode": "text"}
    elif isinstance(geom, GeomBar):
        return base
    elif isinstance(geom, GeomArea):
        return {**base, "fill": 'tozeroy'}


def add_to_fig(fig, geom, trace_args):
    if isinstance(geom, (GeomLineBasic, GeomLine, GeomPoint, GeomText, GeomArea)):
        fig.add_scatter(**trace_args)
    elif isinstance(geom, GeomBar):
        fig.add_bar(**trace_args)
        fig.update_layout(barmode=bar_position_plotly_to_gg(geom.position))


@frozen_dataclass
class Geom:
    aes: Aesthetic

    # TODO
    def apply_to_fig(self, parent, grouped_data, fig, precomputed, facet_row, facet_col, legend_cache):
        for df in grouped_data:
            trace_args = get_trace_args(self, df, facet_row, facet_col)
            for aes_name, (plotly_name, default) in self.aes_to_arg.items():
                if hasattr(self, aes_name) and getattr(self, aes_name) is not None:
                    trace_args[plotly_name] = getattr(self, aes_name)
                elif aes_name in df.attrs:
                    trace_args[plotly_name] = df.attrs[aes_name]
                elif aes_name in df.columns:
                    trace_args[plotly_name] = df[aes_name]
                elif default is not None:
                    trace_args[plotly_name] = default
            if "name" in trace_args:
                trace_args["legendgroup"] = trace_args["name"]
                if trace_args["name"] in legend_cache:
                    trace_args["showlegend"] = False
                else:
                    trace_args["showlegend"] = True
                    legend_cache.add(trace_args["name"])
            add_to_fig(fig, self, trace_args)

    @abc.abstractmethod
    def get_stat(self):
        pass


def bar_position_plotly_to_gg(plotly_pos):
    ggplot_to_plotly = {'dodge': 'group', 'stack': 'stack', 'identity': 'overlay'}
    return ggplot_to_plotly[plotly_pos]


def linetype_plotly_to_gg(plotly_linetype):
    linetype_dict = {
        "solid": "solid",
        "dashed": "dash",
        "dotted": "dot",
        "longdash": "longdash",
        "dotdash": "dashdot"
    }
    return linetype_dict[plotly_linetype]


class GeomLineBasic(Geom):
    aes_to_arg = {
        "color": ("line_color", "black"),
        "size": ("marker_size", None),
        "tooltip": ("hovertext", None),
        "color_legend": ("name", None)
    }

    def __init__(self, aes, color):
        super().__init__(aes)
        self.color = color

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df):
            trace_args = {
                "x": df.x,
                "y": df.y,
                "mode": "lines",
                "row": facet_row,
                "col": facet_col
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_scatter(**trace_args)

        for group_df in grouped_data:
            plot_group(group_df)

    @abc.abstractmethod
    def get_stat(self):
        return ...


class GeomPoint(Geom):

    aes_to_arg = {
        "color": ("marker_color", "black"),
        "size": ("marker_size", None),
        "tooltip": ("hovertext", None),
        "color_legend": ("name", None),
        "alpha": ("marker_opacity", None)
    }

    def __init__(self, aes, color=None, size=None, alpha=None):
        super().__init__(aes)
        self.color = color
        self.size = size
        self.alpha = alpha

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df):
            trace_args = {
                "x": df.x,
                "y": df.y,
                "mode": "lines",
                "row": facet_row,
                "col": facet_col
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_scatter(**trace_args)

        for group_df in grouped_data:
            plot_group(group_df)

        def plot_group(df):
            trace_args = {
                "x": df.x,
                "y": df.y,
                "mode": "markers",
                "row": facet_row,
                "col": facet_col
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_scatter(**trace_args)

        for group_df in grouped_data:
            plot_group(group_df)

    def get_stat(self):
        return StatIdentity()


class GeomLine(GeomLineBasic):

    def __init__(self, aes, color=None):
        super().__init__(aes, color)
        self.color = color

    def get_stat(self):
        return StatIdentity()


class GeomFunction(GeomLineBasic):
    def __init__(self, aes, fun, color):
        super().__init__(aes, color)
        self.fun = fun

    def get_stat(self):
        return StatFunction(self.fun)


class GeomArea(Geom):
    aes_to_arg = {
        "fill": ("fillcolor", "black"),
        "color": ("line_color", "rgba(0, 0, 0, 0)"),
        "tooltip": ("hovertext", None),
        "fill_legend": ("name", None)
    }

    def __init__(self, aes, fill, color):
        super().__init__(aes)
        self.fill = fill
        self.color = color

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df):
            trace_args = {
                "x": df.x,
                "y": df.y,
                "row": facet_row,
                "col": facet_col,
                "fill": 'tozeroy'
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_scatter(**trace_args)

        for group_df in grouped_data:
            plot_group(group_df)

    def get_stat(self):
        return StatIdentity()


class GeomText(Geom):
    aes_to_arg = {
        "color": ("textfont_color", "black"),
        "size": ("marker_size", None),
        "tooltip": ("hovertext", None),
        "color_legend": ("name", None),
        "alpha": ("marker_opacity", None)
    }

    def __init__(self, aes, color=None, size=None, alpha=None):
        super().__init__(aes)
        self.color = color
        self.size = size
        self.alpha = alpha

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df):
            trace_args = {
                "x": df.x,
                "y": df.y,
                "text": df.label,
                "mode": "text",
                "row": facet_row,
                "col": facet_col
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_scatter(**trace_args)

        for group_df in grouped_data:
            plot_group(group_df)

    def get_stat(self):
        return StatIdentity()


class GeomBar(Geom):

    aes_to_arg = {
        "fill": ("marker_color", "black"),
        "color": ("marker_line_color", None),
        "tooltip": ("hovertext", None),
        "fill_legend": ("name", None),
        "alpha": ("marker_opacity", None)
    }

    def __init__(self, aes, fill=None, color=None, alpha=None, position="stack", size=None, stat=None):
        super().__init__(aes)
        self.fill = fill
        self.color = color
        self.position = position
        self.size = size
        self.alpha = alpha

        if stat is None:
            stat = StatCount()
        self.stat = stat

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df):
            trace_args = {
                "x": df.x,
                "y": df.y,
                "row": facet_row,
                "col": facet_col
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_bar(**trace_args)

        for group_df in grouped_data:
            plot_group(group_df)

        fig_so_far.update_layout(barmode=bar_position_plotly_to_gg(self.position))

    def get_stat(self):
        return self.stat


class GeomHistogram(Geom):
    aes_to_arg = {
        "fill": ("marker_color", "black"),
        "color": ("marker_line_color", None),
        "tooltip": ("hovertext", None),
        "fill_legend": ("name", None),
        "alpha": ("marker_opacity", None)
    }

    def __init__(self, aes, min_val=None, max_val=None, bins=None, fill=None, color=None, alpha=None, position='stack', size=None):
        super().__init__(aes)
        self.min_val = min_val
        self.max_val = max_val
        self.bins = bins
        self.fill = fill
        self.color = color
        self.alpha = alpha
        self.position = position
        self.size = size

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        min_val = self.min_val if self.min_val is not None else precomputed.min_val
        max_val = self.max_val if self.max_val is not None else precomputed.max_val
        # This assumes it doesn't really make sense to use another stat for geom_histogram
        bins = self.bins if self.bins is not None else self.get_stat().DEFAULT_BINS
        bin_width = (max_val - min_val) / bins

        num_groups = len(grouped_data)

        def plot_group(df, idx):
            left_xs = df.x

            if self.position == "dodge":
                x = left_xs + bin_width * (2 * idx + 1) / (2 * num_groups)
                bar_width = bin_width / num_groups

            elif self.position in {"stack", "identity"}:
                x = left_xs + bin_width / 2
                bar_width = bin_width
            else:
                raise ValueError(f"Histogram does not support position = {self.position}")

            right_xs = left_xs + bin_width

            trace_args = {
                "x": x,
                "y": df.y,
                "row": facet_row,
                "col": facet_col,
                "customdata": list(zip(left_xs, right_xs)),
                "width": bar_width,
                "hovertemplate":
                    "Range: [%{customdata[0]:.3f}-%{customdata[1]:.3f})<br>"
                    "Count: %{y}<br>"
                    "<extra></extra>",
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_bar(**trace_args)

        for idx, group_df in enumerate(grouped_data):
            plot_group(group_df, idx)

        fig_so_far.update_layout(barmode=bar_position_plotly_to_gg(self.position))

    def get_stat(self):
        return StatBin(self.min_val, self.max_val, self.bins)


class GeomDensity(Geom):
    aes_to_arg = {
        "fill": ("marker_color", "black"),
        "color": ("marker_line_color", None),
        "tooltip": ("hovertext", None),
        "fill_legend": ("name", None),
        "alpha": ("marker_opacity", None)
    }

    def __init__(self, aes, k=1000, smoothing=0.5, fill=None, color=None, alpha=None):
        super().__init__(aes)
        self.k = k
        self.smoothing = smoothing
        self.fill = fill
        self.color = color
        self.alpha = alpha

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df, idx):
            slope = 1.0 / (df.attrs['max'] - df.attrs['min'])
            n = df.attrs['n']
            min = df.attrs['min']
            max = df.attrs['max']
            values = df.value.to_numpy()
            weights = df.weight.to_numpy()

            def f(x, prev):
                inv_scale = (np.sqrt(n * slope) / self.smoothing) * np.sqrt(prev / weights)
                diff = x[:, np.newaxis] - values
                grid = (3 / (4 * n)) * weights * np.maximum(0, inv_scale - np.power(diff, 2) * np.power(inv_scale, 3))
                return np.sum(grid, axis=1)

            round1 = f(values, np.full(len(values), slope))
            x_d = np.linspace(min, max, 1000)
            final = f(x_d, round1)

            trace_args = {
                "x": x_d,
                "y": final,
                "mode": "lines",
                "fill": "tozeroy",
                "row": facet_row,
                "col": facet_col
            }

            self._add_aesthetics_to_trace_args(trace_args, df)
            self._update_legend_trace_args(trace_args, legend_cache)

            fig_so_far.add_scatter(**trace_args)

        for idx, group_df in enumerate(grouped_data):
            plot_group(group_df, idx)

    def get_stat(self):
        return StatCDF(self.k)


class GeomHLine(Geom):

    def __init__(self, yintercept, linetype="solid", color=None):
        self.yintercept = yintercept
        self.aes = aes()
        self.linetype = linetype
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        line_attributes = {
            "y": self.yintercept,
            "line_dash": linetype_plotly_to_gg(self.linetype)
        }
        if self.color is not None:
            line_attributes["line_color"] = self.color

        fig_so_far.add_hline(**line_attributes)

    def get_stat(self):
        return StatNone()


class GeomVLine(Geom):

    def __init__(self, xintercept, linetype="solid", color=None):
        self.xintercept = xintercept
        self.aes = aes()
        self.linetype = linetype
        self.color = color

    def apply_to_fig(self, parent, agg_result, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        line_attributes = {
            "x": self.xintercept,
            "line_dash": linetype_plotly_to_gg(self.linetype)
        }
        if self.color is not None:
            line_attributes["line_color"] = self.color

        fig_so_far.add_vline(**line_attributes)

    def get_stat(self):
        return StatNone()


class GeomTile(Geom):

    def __init__(self, aes):
        self.aes = aes

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df):

            for idx, row in df.iterrows():
                x_center = row['x']
                y_center = row['y']
                width = row['width']
                height = row['height']
                shape_args = {
                    "type": "rect",
                    "x0": x_center - width / 2,
                    "y0": y_center - height / 2,
                    "x1": x_center + width / 2,
                    "y1": y_center + height / 2,
                    "row": facet_row,
                    "col": facet_col,
                    "opacity": row.get('alpha', 1.0)
                }
                if "fill" in df.attrs:
                    shape_args["fillcolor"] = df.attrs["fill"]
                elif "fill" in row:
                    shape_args["fillcolor"] = row["fill"]
                else:
                    shape_args["fillcolor"] = "black"
                fig_so_far.add_shape(**shape_args)

        for group_df in grouped_data:
            plot_group(group_df)

    def get_stat(self):
        return StatIdentity()


class GeomRibbon(Geom):
    aes_to_arg = {
        "fill": ("fillcolor", "black"),
        "color": ("line_color", "rgba(0, 0, 0, 0)"),
        "tooltip": ("hovertext", None),
        "fill_legend": ("name", None)
    }

    def __init__(self, aes, fill, color):
        super().__init__(aes)
        self.fill = fill
        self.color = color

    def apply_to_fig(self, parent, grouped_data, fig_so_far, precomputed, facet_row, facet_col, legend_cache):
        def plot_group(df):

            trace_args_bottom = {
                "x": df.x,
                "y": df.ymin,
                "row": facet_row,
                "col": facet_col,
                "mode": "lines",
                "showlegend": False
            }
            self._add_aesthetics_to_trace_args(trace_args_bottom, df)
            self._update_legend_trace_args(trace_args_bottom, legend_cache)

            trace_args_top = {
                "x": df.x,
                "y": df.ymax,
                "row": facet_row,
                "col": facet_col,
                "mode": "lines",
                "fill": 'tonexty'
            }
            self._add_aesthetics_to_trace_args(trace_args_top, df)
            self._update_legend_trace_args(trace_args_top, legend_cache)

            fig_so_far.add_scatter(**trace_args_bottom)
            fig_so_far.add_scatter(**trace_args_top)

        for group_df in grouped_data:
            plot_group(group_df)

    def get_stat(self):
        return StatIdentity()


def _debug_print(plot: Plot) -> None:
    print("Ggplot Object:")
    print("Aesthetics")
    pprint(plot.aes)
    print("Scales:")
    pprint(plot.scales)
    print("Geoms:")
    pprint(plot.geoms)


def add_default_scales(plot, mapping):
    fields = None
    aesthetic = mapping

    # TODO
    for aesthetic_str, mapped_expr in aesthetic.items():
        dtype = mapped_expr.dtype
        if aesthetic_str not in plot.scales:
            is_continuous = mapped_expr.dtype in [hl.tint32, hl.tint64, hl.tfloat32, hl.tfloat64]
            # We only know how to come up with a few default scales.
            if aesthetic_str == "x":
                if is_continuous:
                    plot.scales["x"] = scale_x_continuous()
                elif isinstance(dtype, hl.tlocus):
                    plot.scales["x"] = scale_x_genomic(reference_genome=dtype.reference_genome)
                else:
                    plot.scales["x"] = scale_x_discrete()
            elif aesthetic_str == "y":
                if is_continuous:
                    plot.scales["y"] = scale_y_continuous()
                elif isinstance(dtype, hl.tlocus):
                    raise ValueError("Don't yet support y axis genomic")
                else:
                    plot.scales["y"] = scale_y_discrete()
            elif aesthetic_str == "color" and not is_continuous:
                plot.scales["color"] = scale_color_discrete()
            elif aesthetic_str == "color" and is_continuous:
                plot.scales["color"] = scale_color_continuous()
            elif aesthetic_str == "fill" and not is_continuous:
                plot.scales["fill"] = scale_fill_discrete()
            elif aesthetic_str == "fill" and is_continuous:
                plot.scales["fill"] = scale_fill_continuous()
            else:
                if is_continuous:
                    plot.scales[aesthetic_str] = ScaleContinuous(aesthetic_str)
                else:
                    plot.scales[aesthetic_str] = ScaleDiscrete(aesthetic_str)

    if fields is None:
        return plot
    else:
        return add_fields(plot, fields)


# TODO immutability (also this is spaghetti)
def to_plotly(plot):
    for aes_key in plot.aes.keys():
        if not plot.scales[aes_key].valid_dtype(plot.aes[aes_key].dtype):
            raise ValueError(f"Invalid scale for aesthetic {aes_key} of type {plot.aes[aes_key].dtype}")
    for geom in plot.geoms:
        aesthetic_dict = geom.aes
        for aes_key in aesthetic_dict.keys():
            if not plot.scales[aes_key].valid_dtype(aesthetic_dict[aes_key].dtype):
                raise ValueError(f"Invalid scale for aesthetic {aes_key} of type {aesthetic_dict[aes_key].dtype}")
    fields_to_select = {"figure_mapping": hl.struct(**plot.aes)}
    if plot.facet is not None:
        fields_to_select["facet"] = plot.facet.facets
    for geom_idx, geom in enumerate(plot.geoms):
        geom_label = f"geom{geom_idx}"
        fields_to_select[geom_label] = hl.struct(**geom.aes)
    selected = plot.ht.select(**fields_to_select)
    mapping_per_geom = []
    precomputes = {}
    for geom_idx, geom in enumerate(plot.geoms):
        geom_label = f"geom{geom_idx}"
        combined_mapping = selected["figure_mapping"].annotate(**selected[geom_label])
        for key in combined_mapping:
            if key in plot.scales:
                combined_mapping = combined_mapping.annotate(**{key: plot.scales[key].transform_data(combined_mapping[key])})
        mapping_per_geom.append(combined_mapping)
        precomputes[geom_label] = get_precomputes(geom.get_stat(), combined_mapping)
    # Is there anything to precompute?
    should_precompute = any([len(precompute) > 0 for precompute in precomputes.values()])
    if should_precompute:
        precomputed = selected.aggregate(hl.struct(**precomputes))
    else:
        precomputed = hl.Struct(**{key: hl.Struct() for key in precomputes.keys()})
    aggregators = {}
    labels_to_stats = {}
    use_faceting = plot.facet is not None
    for geom_idx, combined_mapping in enumerate(mapping_per_geom):
        stat = plot.geoms[geom_idx].get_stat()
        geom_label = f"geom{geom_idx}"
        if use_faceting:
            agg = hl.agg.group_by(selected.facet, stat.make_agg(combined_mapping, precomputed[geom_label], plot.scales))
        else:
            agg = stat.make_agg(combined_mapping, precomputed[geom_label], plot.scales)
        aggregators[geom_label] = agg
        labels_to_stats[geom_label] = stat
    all_agg_results = selected.aggregate(hl.struct(**aggregators))
    if use_faceting:
        facet_list = list(set(itertools.chain(*[list(x.keys()) for x in all_agg_results.values()])))
        facet_to_idx = {facet: idx for idx, facet in enumerate(facet_list)}
        aggregated = {geom_label: {facet_to_idx[facet]: agg_result for facet, agg_result in facet_to_agg_result.items()} for geom_label, facet_to_agg_result in all_agg_results.items()}
        num_facets = len(facet_list)
    else:
        aggregated = {geom_label: {0: agg_result} for geom_label, agg_result in all_agg_results.items()}
        num_facets = 1
        facet_list = None
    geoms_and_grouped_dfs_by_facet_idx = []
    for geom, (geom_label, agg_result_by_facet) in zip(plot.geoms, aggregated.items()):
        dfs_by_facet_idx = {facet_idx: labels_to_stats[geom_label].listify(agg_result) for facet_idx, agg_result in agg_result_by_facet.items()}
        geoms_and_grouped_dfs_by_facet_idx.append((geom, geom_label, dfs_by_facet_idx))
    # Create scaling functions based on all the data:
    transformers = {}
    for scale in plot.scales.values():
        all_dfs = list(itertools.chain(*[facet_to_dfs_dict.values() for _, _, facet_to_dfs_dict in geoms_and_grouped_dfs_by_facet_idx]))
        transformers[scale.aesthetic_name] = create_local_transformer(scale, all_dfs)
    if plot.facet is not None:
        n_facet_cols = int(math.ceil(math.sqrt(num_facets)))
        n_facet_rows = int(math.ceil(num_facets / n_facet_cols))
        subplot_args = {
            "rows": n_facet_rows,
            "cols": n_facet_cols,
            "shared_yaxes": True,
            "subplot_titles": [", ".join([str(fs_value) for fs_value in facet_struct.values()]) for facet_struct in facet_list]
        }
    else:
        n_facet_cols = 1
        subplot_args = {
            "rows": 1,
            "cols": 1,
        }
    fig = make_subplots(**subplot_args)
    # Need to know what I've added to legend already so we don't do it more than once.
    legend_cache = set()
    for geom, geom_label, facet_to_grouped_dfs in geoms_and_grouped_dfs_by_facet_idx:
        for facet_idx, grouped_dfs in facet_to_grouped_dfs.items():
            scaled_grouped_dfs = []
            for df in grouped_dfs:
                scales_to_consider = list(df.columns) + list(df.attrs)
                relevant_aesthetics = [scale_name for scale_name in scales_to_consider if scale_name in plot.scales]
                scaled_df = df
                for relevant_aesthetic in relevant_aesthetics:
                    scaled_df = transformers[relevant_aesthetic](scaled_df)
                scaled_grouped_dfs.append(scaled_df)
            facet_row = facet_idx // n_facet_cols + 1
            facet_col = facet_idx % n_facet_cols + 1
            geom.apply_to_fig(plot, scaled_grouped_dfs, fig, precomputed[geom_label], facet_row, facet_col, legend_cache)
    # Important to update axes after labels, axes names take precedence.
    fig.update_layout(**as_nonempty_dict(plot.labels))
    if plot.scales.get("x") is not None:
        plot.scales["x"].apply_to_fig(plot, fig)
    if plot.scales.get("y") is not None:
        plot.scales["y"].apply_to_fig(plot, fig)
    if plot.coord_cartesian is not None:
        if plot.coord_cartesian.xlim is not None:
            fig.update_xaxes(range=list(plot.coord_cartesian.xlim))
        if plot.coord_cartesian.ylim is not None:
            fig.update_yaxes(range=list(plot.coord_cartesian.ylim))
    fig = fig.update_xaxes(title_font_size=18, ticks="outside")
    fig = fig.update_yaxes(title_font_size=18, ticks="outside")
    fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(linecolor="black"),
        yaxis=dict(linecolor="black"),
        font_family='Arial, "Open Sans", verdana, sans-serif',
        title_font_size=26
    )
    return fig


def show(plot):
    to_plotly(plot).show()


def write_image(plot, path):
    to_plotly(plot).write_image(path)


def _repr_html_(plot):
    return to_plotly(plot)._repr_html_()
