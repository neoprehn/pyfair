"""Distribution curve rendering for FAIR report visualisations."""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter
from scipy.stats import gaussian_kde

from ..report.base_curve import FairBaseCurve


class FairDistributionCurve(FairBaseCurve):
    """Render FAIR loss distributions as histogram and/or smooth density.

    The main report view is intended to show the simulated ``Risk`` results in
    a cleaner, less "bucket-heavy" style than a plain histogram. By default the
    chart combines a lightly transparent histogram with a smooth kernel density
    estimate (KDE). Both layers can be controlled independently.

    Parameters
    ----------
    model_or_iterable : FairModel, FairMetaModel, or iterable
        One or more calculated FAIR models.
    currency_prefix : str, optional
        Prefix used for x-axis currency formatting.
    show_histogram : bool, optional
        Whether to render the histogram layer.
    show_density : bool, optional
        Whether to render the smooth density curve.
    bins : int, optional
        Number of histogram bins.
    bandwidth_adjust : float, optional
        Multiplier applied to the default KDE bandwidth. Values below 1.0 make
        the curve more detailed, values above 1.0 make it smoother.
    histogram_alpha : float, optional
        Transparency of the histogram bars.
    fill_density : bool, optional
        Whether to softly fill the area beneath the density curve.
    """

    def __init__(
        self,
        model_or_iterable,
        currency_prefix='$',
        show_histogram=True,
        show_density=True,
        bins=40,
        bandwidth_adjust=1.0,
        histogram_alpha=0.25,
        fill_density=False,
    ):
        self._input = self._input_check(model_or_iterable)
        self._currency_prefix = currency_prefix
        self._show_histogram = show_histogram
        self._show_density = show_density
        self._bins = max(5, int(bins))
        self._bandwidth_adjust = float(bandwidth_adjust)
        self._histogram_alpha = float(histogram_alpha)
        self._fill_density = bool(fill_density)

    def _extract_series(self, model, target='Risk'):
        """Return a clean numeric series for plotting."""
        values = model.export_results().loc[:, target]
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return values

    def _build_x_grid(self, arrays):
        """Build a common x-grid across all model result sets."""
        finite_arrays = [arr for arr in arrays if arr.size > 0]
        if not finite_arrays:
            return np.linspace(0.0, 1.0, 200)

        xmin = min(float(arr.min()) for arr in finite_arrays)
        xmax = max(float(arr.max()) for arr in finite_arrays)

        if xmax <= xmin:
            pad = max(abs(xmax) * 0.05, 1.0)
            xmin -= pad
            xmax += pad
        else:
            pad = (xmax - xmin) * 0.03
            xmin = max(0.0, xmin - pad)
            xmax += pad

        return np.linspace(xmin, xmax, 512)

    def _get_kde(self, values):
        """Create a KDE object for the provided values.

        ``gaussian_kde`` can fail for degenerate or near-degenerate samples.
        In that case we return ``None`` and let the caller silently skip the
        smooth curve for that particular model.
        """
        if values.size < 2:
            return None

        if np.allclose(values, values[0]):
            return None

        try:
            kde = gaussian_kde(values)
        except Exception:
            return None

        if self._bandwidth_adjust != 1.0:
            default_factor = kde.factor
            kde.set_bandwidth(bw_method=default_factor * self._bandwidth_adjust)

        return kde

    def generate_icon(self, model_name, target):
        """Generate a minimalist histogram icon for a model parameter."""
        model = self._input[model_name]
        data = self._extract_series(model, target)

        fig, ax = plt.subplots(figsize=(6, 1))
        xmax = float(data.max()) if data.size else 1.0
        ax.set_xlim(0, xmax)

        for spine in ['left', 'right', 'top', 'bottom']:
            ax.spines[spine].set_visible(False)
        plt.tick_params(bottom=False)
        ax.yaxis.set_visible(False)

        if xmax <= 1:
            ax.axes.xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            plt.xticks([0, 1])
        else:
            ax.axes.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            plt.xticks([0, xmax])

        plt.hist(data, bins=100, range=(0, xmax), alpha=.4)
        if data.size:
            plt.vlines(data.mean(), 0, plt.ylim()[1], linestyle='--')
        plt.tight_layout()
        return (fig, ax)

    def generate_image(self):
        """Create the main risk distribution chart."""
        fig, ax = plt.subplots(figsize=(16, 6))
        plt.subplots_adjust(bottom=.2)
        ax.axes.set_title('Risk Distribution', fontsize=20)

        ax.axes.xaxis.set_major_formatter(
            StrMethodFormatter(self._currency_prefix + '{x:,.0f}')
        )
        ax.axes.xaxis.set_tick_params(rotation=-45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('left')

        risk_arrays = []
        for _, model in self._input.items():
            risk_arrays.append(self._extract_series(model, 'Risk'))

        x_grid = self._build_x_grid(risk_arrays)

        # Plot on density scale so histogram and smooth curve are directly
        # comparable on the same axis.
        ax.set_ylabel('Density')

        legend_handles = []
        legend_labels = []

        for name, model in self._input.items():
            risk = self._extract_series(model, 'Risk')
            if risk.size == 0:
                continue

            histogram_container = None
            if self._show_histogram:
                histogram_container = ax.hist(
                    risk,
                    bins=self._bins,
                    density=True,
                    alpha=self._histogram_alpha,
                    edgecolor='none',
                )

            density_line = None
            if self._show_density:
                kde = self._get_kde(risk)
                if kde is not None:
                    density_values = kde(x_grid)
                    density_line, = ax.plot(x_grid, density_values, linewidth=2.2)
                    if self._fill_density:
                        ax.fill_between(x_grid, 0, density_values, alpha=0.08)

            if density_line is not None:
                legend_handles.append(density_line)
                legend_labels.append(name)
            elif histogram_container is not None and histogram_container[2]:
                legend_handles.append(histogram_container[2][0])
                legend_labels.append(name)

        if legend_handles:
            ax.legend(legend_handles, legend_labels, frameon=False)

        plt.margins(x=0)
        return (fig, ax)
