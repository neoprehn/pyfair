"""Module for generating a FAIR tree graph."""

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


class FairTreeGraph(object):
    """Provides a tree diagram to summarize FAIR calculations.

    This tree class provides an image that describes those nodes that have
    been calculated, those nodes that have had data supplied, and those
    nodes which are not required.

    The implementation supports both:
    1. legacy flat parameter dictionaries
    2. structured distribution dictionaries with distribution/params/confidence
    """

    # Static layout for the FAIR tree.
    _DIMENSIONS = pd.DataFrame.from_dict(
        {
            'Contact Frequency'             : ['C'   ,    0,    0,  600,  800],
            'Threat Event Frequency'        : ['TEF' ,  600,  800, 1800, 1600],
            'Probability of Action'         : ['A'   , 1200,    0,  600,  800],
            'Threat Capability'             : ['TC'  , 2400,    0, 3000,  800],
            'Vulnerability'                 : ['V'   , 3000,  800, 1800, 1600],
            'Control Strength'              : ['CS'  , 3600,    0, 3000,  800],
            'Loss Magnitude'                : ['LM'  , 6600, 1600, 4200, 2400],
            'Loss Event Frequency'          : ['LEF' , 1800, 1600, 4200, 2400],
            'Risk'                          : ['R'   , 4200, 2400, 4200, 5000],
            'Primary Loss'                  : ['PL'  , 5400,  800, 6600, 1600],
            'Secondary Loss'                : ['SL'  , 7800,  800, 6600, 1600],
            'Secondary Loss Event Frequency': ['SLEF', 7200,    0, 7800,  800],
            'Secondary Loss Event Magnitude': ['SLEM', 8400,    0, 7800,  800],
        },
        orient='index',
        columns=['tag', 'self_x', 'self_y', 'parent_x', 'parent_y']
    )

    def __init__(self, model, format_strings):
        self._colormap = {
            'Not Required': 'grey',
            'Supplied': 'green',
            'Calculated': 'blue'
        }

        self._results = model.export_results().T
        self._format_strings = format_strings

        # Summary statistics for calculated nodes.
        self._result_summary = pd.DataFrame({
            'μ': self._results.mean(axis=1),
            'σ': self._results.std(axis=1),
            '↑': self._results.max(axis=1),
        })

        # FAIR node statuses from the dependency tree.
        self._statuses = model.get_node_statuses()
        self._process_statuses()

        # Parameter metadata aligned to the FAIR tree order.
        self._params = pd.DataFrame(model.export_params()).T.reindex(self._statuses.index)

        # Merge all data into one frame for plotting.
        self._data = pd.concat(
            [self._DIMENSIONS, self._statuses, self._params, self._result_summary],
            axis=1
        )

    def _process_statuses(self):
        """Transform node status information into display metadata."""

        # Handle both:
        # 1. {'node': 'Supplied'}
        # 2. {'node': {'status': 'Supplied'}}
        if isinstance(self._statuses, dict):
            first_val = next(iter(self._statuses.values()))

            # Case 1: flat dict → wrap into dict structure
            if isinstance(first_val, str):
                self._statuses = {
                    k: {'status': v}
                    for k, v in self._statuses.items()
                }

        self._statuses = pd.DataFrame(self._statuses).T
        self._statuses = self._statuses.reindex(self._DIMENSIONS.index)

        self._statuses['color'] = self._statuses['status'].map(self._colormap)
        self._statuses['color'] = self._statuses['color'].fillna('grey')

    def _tweak_axes(self, ax):
        """Apply general axis formatting for the FAIR tree image."""
        ax.set_xlim(-200, 9500)
        ax.set_ylim(-200, 3200)
        ax.axis('off')
        return ax

    def _generate_rects(self, ax):
        """Generate the background rectangles for each FAIR node."""
        rectangles = [
            Rectangle(
                (row['self_x'], row['self_y']),
                1100,   # slightly wider
                700     # taller to fit multiline parameter text
            )
            for _, row in self._data.iterrows()
        ]

        colors = self._data['color'].tolist()
        collection = PatchCollection(
            rectangles,
            facecolor=colors,
            edgecolor='black',
            alpha=.3,
            linewidths=2
        )
        ax.add_collection(collection)

    def _safe_format(self, target, value):
        """Format a value using the FAIR factor formatter if available."""
        if pd.isnull(value):
            return ''

        try:
            fmt = self._format_strings[target]
        except KeyError:
            return str(value)

        try:
            return fmt.format(value)
        except Exception:
            return str(value)

    def _is_structured_param_record(self, row):
        """Return True if the row uses the new structured parameter schema."""
        return (
            'distribution' in row.index and
            'params' in row.index and
            isinstance(row.get('params', None), dict)
        )

    def _stringify_params(self, params):
        """Convert a parameter dictionary into a compact multiline string."""
        if not isinstance(params, dict) or len(params) == 0:
            return ''

        parts = []
        for key, value in params.items():
            if isinstance(value, (int, float)):
                parts.append(f'{key}={value:.4f}')
            else:
                parts.append(f'{key}={value}')
        return '\n'.join(parts)

    def _build_supplied_output(self, row):
        """Build the text shown for supplied nodes."""
        target = row.name

        # Raw inputs are shown explicitly.
        if 'raw' in row.index and isinstance(row.get('raw', None), list):
            return 'Raw input'

        # Structured parameter path.
        if self._is_structured_param_record(row):
            params = row.get('params', {})
            confidence = row.get('confidence', None)

            lines = []

            # Show only the resolved parameters in the node.
            # Distribution type is already visible in the report table.
            param_text = self._stringify_params(params)
            if param_text:
                lines.append(param_text)

            # Show confidence only if explicitly present.
            if confidence not in [None, '']:
                lines.append(f'conf: {confidence}')

            return '\n'.join(lines)
        # Legacy flat parameter path.
        # Keep old behavior, but also support a few extra keys.
        legacy_order = ['high', 'mode', 'low', 'mean', 'stdev', 'constant']
        legacy_symbols = {
            'high': '↑',
            'mode': '-',
            'low': '↓',
            'mean': 'μ',
            'stdev': 'σ',
            'constant': 'c',
        }

        data = pd.Series({
            legacy_symbols[key]: row[key]
            for key in legacy_order
            if key in row.index and pd.notnull(row[key])
        })

        if len(data) == 0:
            return ''

        data = data.map(lambda x: self._safe_format(target, x))
        value_just = data.str.len().max()

        output = '\n'.join([
            key + '  ' + value.rjust(value_just)
            for key, value in data.items()
        ])
        return output

    def _build_calculated_output(self, row):
        """Build the text shown for calculated nodes."""
        target = row.name
        data = row.loc[['μ', 'σ', '↑']].dropna()

        if len(data) == 0:
            return ''

        data = data.map(lambda x: self._safe_format(target, x))
        value_just = data.str.len().max()

        output = '\n'.join([
            key + '  ' + value.rjust(value_just)
            for key, value in data.items()
        ])
        return output

    def _generate_text(self, row, ax):
        """Generate text content inside each FAIR node rectangle."""
        # Header
        plt.text(
            row['self_x'] + 550,
            row['self_y'] + 560,
            row['tag'],
            horizontalalignment='center',
            fontsize=14,
            fontweight='bold',
        )

        calculated = row['status'] == 'Calculated'
        supplied = row['status'] == 'Supplied'

        if calculated:
            output = self._build_calculated_output(row)
        elif supplied:
            output = self._build_supplied_output(row)
        else:
            output = ''

        plt.text(
            row['self_x'] + 30,
            row['self_y'] + 90,
            output,
            horizontalalignment='left',
            fontsize=8,
            fontfamily='monospace'
        )

    def _generate_lines(self, row, ax):
        """Generate connecting lines between FAIR nodes."""
        if (row['color'] != 'grey') and row.name != 'Risk':
            ax.annotate(
                None,
                xy=(row['parent_x'] + 500, row['parent_y']),
                xytext=(row['self_x'] + 500, row['self_y'] + 700),
                arrowprops=dict(
                    arrowstyle='-',
                    connectionstyle='angle3,angleA=0,angleB=-90',
                    ec=row['color'],
                    alpha=.3,
                    linestyle='--',
                    linewidth=3
                ),
            )

    def _generate_legend(self, ax):
        """Generate the FAIR tree legend."""
        patches = [
            Patch(color=color, label=label, alpha=.3)
            for label, color in self._colormap.items()
        ]
        plt.legend(handles=patches, frameon=False)

    def generate_image(self):
        """Create the FAIR tree figure and return (fig, ax)."""
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 6)
        ax = self._tweak_axes(ax)
        self._data.apply(self._generate_text, args=[ax], axis=1)
        self._generate_rects(ax)
        self._data.apply(self._generate_lines, args=[ax], axis=1)
        self._generate_legend(ax)
        return (fig, ax)