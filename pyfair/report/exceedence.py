"""Exceedence curve for viewing threshold values"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
from scipy.interpolate import PchipInterpolator

from ..utility.fair_exception import FairException
from ..report.base_curve import FairBaseCurve


class FairExceedenceCurves(FairBaseCurve):
    """Plots one or more exceedence curves."""

    def __init__(self, model_or_iterable, currency_prefix='$', risk_tolerance=None):
        super().__init__()
        self._currency_prefix = currency_prefix
        self._input = self._input_check(model_or_iterable)
        self._risk_tolerance = risk_tolerance
        self._intersection_cache = None

    def generate_image(self):
        """Main function for generating plots."""
        fig, axes = plt.subplots(2, 1, figsize=(16, 11))
        plt.subplots_adjust(hspace=.6, bottom=.12, top=.92, right=.82)
        ax1, ax2 = axes

        # Add common percentile guide lines only once on the LEC.
        reference_levels = [20, 50, 80]
        for level in reference_levels:
            ax2.axhline(
                y=level,
                color='grey',
                linestyle='--',
                linewidth=1,
                alpha=0.5
            )

        ax2.text(
            0.015,
            level,
            f'{level}%',
            transform=ax2.get_yaxis_transform(),
            fontsize=8,
            color='grey',
            verticalalignment='center',
            horizontalalignment='left',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
        )

        curve_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        legend_labels = []
        legend_handles = []
        curve_data = {}

        for idx, (name, model) in enumerate(self._input.items()):
            data = model.export_results()
            risk = np.asarray(data['Risk'], dtype=float)

            percentiles = [0.2, 0.5, 0.8]
            quantile_values = {
                p: np.quantile(risk, p)
                for p in percentiles
            }

            space = self._build_space(risk)
            prob_xy = self._get_prob_data(space, risk)
            loss_xy = self._get_loss_data(space, risk)
            curve_data[name] = {
                'space': np.asarray(loss_xy[0], dtype=float),
                'loss_expectancy': np.asarray(loss_xy[1], dtype=float),
            }

            color = curve_colors[idx % len(curve_colors)]

            prob_handle = self._generate_prob_curve(name, ax1, *prob_xy, color=color)
            self._generate_loss_curve(name, ax2, *loss_xy, quantile_values, color=color)

            legend_labels.append(name)
            legend_handles.append(prob_handle)

        # Add optional risk tolerance overlay on the LEC.
        tolerance_handle, tolerance_label = self._draw_risk_tolerance(ax2)
        if tolerance_handle is not None:
            legend_handles.append(tolerance_handle)
            legend_labels.append(tolerance_label)

        intersections = self._compute_intersections(curve_data)
        intersection_handle = self._draw_intersections(ax2, intersections)
        if intersection_handle is not None:
            legend_handles.append(intersection_handle)
            legend_labels.append('Tolerance Intersection')

        # Create one shared legend only once.
        self._add_shared_legend(fig, legend_handles, legend_labels)

        return (fig, (ax1, ax2))

    def get_tolerance_intersections(self):
        """Return tolerance/LEC intersections as a dataframe.

        Returns
        -------
        pandas.DataFrame
            One row per intersection with model name, loss, exceedance,
            tolerance type and an ordinal intersection number.
        """
        if self._intersection_cache is None:
            curve_data = {}
            for name, model in self._input.items():
                risk = np.asarray(model.export_results()['Risk'], dtype=float)
                space = self._build_space(risk)
                loss_xy = self._get_loss_data(space, risk)
                curve_data[name] = {
                    'space': np.asarray(loss_xy[0], dtype=float),
                    'loss_expectancy': np.asarray(loss_xy[1], dtype=float),
                }
            self._compute_intersections(curve_data)

        rows = []
        for name, intersections in (self._intersection_cache or {}).items():
            for idx, point in enumerate(intersections, start=1):
                rows.append({
                    'Model': name,
                    'Intersection #': idx,
                    'Tolerance Type': point['tolerance_type'],
                    'Loss': point['loss'],
                    'Exceedance %': point['exceedance_percent'],
                    'Tolerance %': point['tolerance_percent'],
                })

        if not rows:
            return pd.DataFrame(
                columns=['Model', 'Intersection #', 'Tolerance Type', 'Loss', 'Exceedance %', 'Tolerance %']
            )

        return pd.DataFrame(rows)

    def _build_space(self, risk):
        """Create a stable geometric x-grid for LEC plotting and analysis.

        The grid should cover both the observed simulated losses and the
        configured tolerance domain. Otherwise a valid curve intersection may
        sit outside the default risk-only display range and be missed.
        """
        positive_risk = risk[risk > 0]
        if positive_risk.size == 0:
            positive_risk = np.array([1e-6], dtype=float)

        risk_min_display = float(np.min(positive_risk))
        risk_max_display = float(np.quantile(risk, 0.99))
        if not np.isfinite(risk_max_display) or risk_max_display <= 0:
            risk_max_display = max(float(positive_risk.max()), 1e-6)

        tolerance = self._normalize_tolerance_points()
        tolerance_min = None
        tolerance_max = None
        if tolerance is not None:
            if tolerance['type'] == 'constant':
                tolerance_min = float(tolerance['value'])
                tolerance_max = float(tolerance['value'])
            elif tolerance['type'] in ('curve', 'distribution'):
                tolerance_min = float(np.min(tolerance['x']))
                tolerance_max = float(np.max(tolerance['x']))

        lower_bound = risk_min_display
        upper_bound = risk_max_display
        if tolerance_min is not None:
            lower_bound = min(lower_bound, tolerance_min)
        if tolerance_max is not None:
            upper_bound = max(upper_bound, tolerance_max)

        space_min = max(lower_bound / 10.0, 1e-6)
        space_max = max(upper_bound, space_min * 10.0)
        return pd.Series(np.geomspace(space_min, space_max, 500))

    def _level_to_exceedance_percent(self, level):
        """Convert a percentile-style label like P90 to exceedance percent."""
        if isinstance(level, str) and level.upper().startswith('P'):
            percentile = float(level[1:])
            return max(0.0, 100.0 - percentile)

        if isinstance(level, (int, float)):
            if 0.0 <= level <= 1.0:
                return level * 100.0
            return float(level)

        raise FairException(f'Unsupported tolerance level "{level}".')

    def _normalize_tolerance_points(self):
        """Normalize tolerance points for downstream plotting and analysis."""
        if not self._risk_tolerance:
            return None

        tol_type = self._risk_tolerance.get('type')
        if tol_type == 'constant':
            value = float(self._risk_tolerance.get('value'))
            if value <= 0:
                raise FairException('Risk tolerance value must be positive.')
            return {
                'type': 'constant',
                'value': value,
            }

        if tol_type == 'curve':
            points = self._risk_tolerance.get('points', [])
            if not isinstance(points, list) or len(points) < 2:
                raise FairException('Risk tolerance curve requires at least two points.')

            normalized = []
            for idx, point in enumerate(points):
                if not isinstance(point, dict):
                    raise FairException(f'Risk tolerance curve point #{idx + 1} must be a dictionary.')

                if 'loss' not in point:
                    raise FairException(f'Risk tolerance curve point #{idx + 1} is missing "loss".')
                if 'level' not in point:
                    raise FairException(f'Risk tolerance curve point #{idx + 1} is missing "level".')

                loss = float(point['loss'])
                if loss <= 0:
                    raise FairException('Risk tolerance curve loss values must be positive.')

                try:
                    exceedance = float(self._level_to_exceedance_percent(point['level']))
                except Exception as exc:
                    raise FairException(
                        f'Invalid risk tolerance curve level "{point["level"]}" '
                        f'at point #{idx + 1}.'
                    ) from exc

                normalized.append((loss, exceedance))

            normalized.sort(key=lambda item: item[0])
            x_vals = np.array([p[0] for p in normalized], dtype=float)
            y_vals = np.array([p[1] for p in normalized], dtype=float)
            y_vals = np.clip(y_vals, 0.0, 100.0)

            if np.unique(x_vals).size != x_vals.size:
                raise FairException('Risk tolerance curve loss values must be unique.')

            # Higher loss must not imply a more permissive tolerance.
            # Therefore tolerated exceedance must be non-increasing as loss grows.
            if np.any(np.diff(y_vals) > 1e-12):
                raise FairException(
                    'Risk tolerance curve must be monotonic: tolerated exceedance must be '
                    'non-increasing as loss increases.'
                )

            return {
                'type': 'curve',
                'x': x_vals,
                'y': y_vals,
            }

        if tol_type == 'distribution':
            distribution = self._risk_tolerance.get('distribution')
            params = self._risk_tolerance.get('params', {})
            sample_count_raw = self._risk_tolerance.get('samples', 100000)

            if not distribution:
                raise FairException('Risk tolerance distribution requires a "distribution" value.')

            if not isinstance(params, dict) or not params:
                raise FairException('Risk tolerance distribution requires non-empty "params".')

            try:
                sample_count = int(sample_count_raw)
            except (TypeError, ValueError):
                raise FairException('Risk tolerance distribution "samples" must be an integer.')

            if sample_count <= 0:
                raise FairException('Risk tolerance distribution "samples" must be greater than zero.')

            samples = self._sample_tolerance_distribution(
                distribution=distribution,
                params=params,
                count=sample_count
            )

            samples = np.asarray(samples, dtype=float)
            samples = samples[np.isfinite(samples)]
            samples = samples[samples > 0]

            if samples.size == 0:
                raise FairException('Risk tolerance distribution produced no positive finite samples.')

            x_vals = np.sort(samples)
            y_vals = (1.0 - (np.arange(1, len(x_vals) + 1) / len(x_vals))) * 100.0

            return {
                'type': 'distribution',
                'distribution': str(distribution).lower(),
                'params': params,
                'samples': sample_count,
                'x': x_vals,
                'y': np.clip(y_vals, 0.0, 100.0),
            }

        raise FairException(f'Unsupported risk tolerance type "{tol_type}".')

    def _sample_tolerance_distribution(self, distribution, params, count):
        """Sample direct loss values for a distribution-based risk tolerance."""
        distribution = str(distribution).lower()

        if distribution == 'pert':
            from ..utility.beta_pert import FairBetaPert

            required = ['low', 'mode', 'high']
            missing = [key for key in required if key not in params]
            if missing:
                raise FairException(
                    f'Risk tolerance pert distribution missing required params: {missing}.'
                )

            low = float(params['low'])
            mode = float(params['mode'])
            high = float(params['high'])
            gamma = float(params.get('gamma', 4.0))

            if low <= 0 or mode <= 0 or high <= 0:
                raise FairException(
                    'Risk tolerance pert distribution requires positive low/mode/high values.'
                )

            if not (low < mode < high):
                raise FairException(
                    'Risk tolerance pert distribution requires low < mode < high.'
                )

            if gamma <= 0:
                raise FairException('Risk tolerance pert distribution gamma must be greater than zero.')

            pert = FairBetaPert(
                low=low,
                mode=mode,
                high=high,
                gamma=gamma
            )
            return pert.random_variates(count)

        if distribution == 'lognormal':
            required = ['mean', 'sigma']
            missing = [key for key in required if key not in params]
            if missing:
                raise FairException(
                    f'Risk tolerance lognormal distribution missing required params: {missing}.'
                )

            mean = float(params['mean'])
            sigma = float(params['sigma'])

            if mean <= 0:
                raise FairException('Risk tolerance lognormal mean must be greater than zero.')
            if sigma <= 0:
                raise FairException('Risk tolerance lognormal sigma must be greater than zero.')

            mu = np.log(mean) - 0.5 * (sigma ** 2)
            return np.random.lognormal(mean=mu, sigma=sigma, size=count)

        raise FairException(
            f'Unsupported risk tolerance distribution "{distribution}". Supported: pert, lognormal.'
        )

    def get_risk_tolerance_details(self):
        """Return report-friendly details for the configured risk tolerance."""
        if not self._risk_tolerance:
            return None

        normalized = self._normalize_tolerance_points()
        if normalized is None:
            return None

        tol_type = normalized['type']

        if tol_type == 'constant':
            return {
                'type': 'constant',
                'value': float(normalized['value']),
            }

        if tol_type == 'curve':
            points = []
            x_vals = normalized.get('x', [])
            y_vals = normalized.get('y', [])
            for x, y in zip(x_vals, y_vals):
                points.append({
                    'loss': float(x),
                    'exceedance_percent': float(y),
                })
            return {
                'type': 'curve',
                'points': points,
                'n_points': len(points),
            }

        if tol_type == 'distribution':
            return {
                'type': 'distribution',
                'distribution': normalized.get('distribution'),
                'params': normalized.get('params', {}),
                'samples': int(normalized.get('samples', 0)),
            }

        return {'type': tol_type}

    def _evaluate_tolerance(self, x_values):
        """Evaluate risk tolerance on supplied x-values.

        Parameters
        ----------
        x_values : array-like
            Positive loss values.

        Returns
        -------
        np.ndarray or None
            Tolerance exceedance percentages for the provided losses.
        """
        normalized = self._normalize_tolerance_points()
        if normalized is None:
            return None

        x_values = np.asarray(x_values, dtype=float)

        if normalized['type'] == 'constant':
            return np.full_like(x_values, np.nan, dtype=float)

        x_vals = normalized['x']
        y_vals = normalized['y']
        log_x = np.log10(x_vals)
        smooth_log_x = np.log10(np.clip(x_values, 1e-12, None))

        interpolator = PchipInterpolator(log_x, y_vals)
        smooth_y = np.empty_like(smooth_log_x)

        inside = (smooth_log_x >= log_x.min()) & (smooth_log_x <= log_x.max())
        smooth_y[inside] = interpolator(smooth_log_x[inside])

        if np.any(smooth_log_x < log_x.min()):
            left_mask = smooth_log_x < log_x.min()
            left_slope = (y_vals[1] - y_vals[0]) / (log_x[1] - log_x[0])
            smooth_y[left_mask] = y_vals[0] + left_slope * (smooth_log_x[left_mask] - log_x[0])

        if np.any(smooth_log_x > log_x.max()):
            right_mask = smooth_log_x > log_x.max()
            right_slope = (y_vals[-1] - y_vals[-2]) / (log_x[-1] - log_x[-2])
            smooth_y[right_mask] = y_vals[-1] + right_slope * (smooth_log_x[right_mask] - log_x[-1])

        return np.clip(smooth_y, 0.0, 100.0)

    def _draw_risk_tolerance(self, ax):
        """Draw the risk tolerance line/curve on the LEC axis."""
        normalized = self._normalize_tolerance_points()
        if normalized is None:
            return None, None

        if normalized['type'] == 'constant':
            line = ax.axvline(
                x=normalized['value'],
                color='red',
                linestyle='-',
                linewidth=1.5,
                alpha=0.9
            )
            return line, 'Risk Tolerance'

        x_min_plot, x_max_plot = ax.get_xlim()
        smooth_x = np.geomspace(max(x_min_plot, 1e-6), max(x_max_plot, x_min_plot * 1.01), 400)
        smooth_y = self._evaluate_tolerance(smooth_x)

        line, = ax.plot(
            smooth_x,
            smooth_y,
            color='red',
            linestyle='-',
            linewidth=1.8,
            alpha=0.9
        )
        label = 'Risk Tolerance'
        if normalized['type'] == 'distribution':
            label = 'Risk Tolerance (Distribution)'
        return line, label

    def _compute_intersections(self, curve_data):
        """Compute tolerance/LEC intersections for all models."""
        results = {}
        normalized = self._normalize_tolerance_points()
        if normalized is None:
            self._intersection_cache = results
            return results

        for name, data in curve_data.items():
            x_vals = np.asarray(data['space'], dtype=float)
            lec_y = np.asarray(data['loss_expectancy'], dtype=float)

            if normalized['type'] == 'constant':
                x0 = normalized['value']
                y0 = self._interpolate_curve_at_x(x_vals, lec_y, x0)
                results[name] = [{
                    'loss': float(x0),
                    'exceedance_percent': float(y0),
                    'tolerance_percent': float(y0),
                    'tolerance_type': 'constant',
                }]
                continue

            tol_y = self._evaluate_tolerance(x_vals)
            diff = lec_y - tol_y
            intersections = []
            eps = 1e-9

            for idx in range(len(diff) - 1):
                d1 = diff[idx]
                d2 = diff[idx + 1]
                x1, x2 = x_vals[idx], x_vals[idx + 1]
                y1, y2 = lec_y[idx], lec_y[idx + 1]
                t1, t2 = tol_y[idx], tol_y[idx + 1]

                if abs(d1) <= eps:
                    intersections.append({
                        'loss': float(x1),
                        'exceedance_percent': float(y1),
                        'tolerance_percent': float(t1),
                        'tolerance_type': normalized['type'],
                    })
                    continue

                if d1 * d2 < 0 or abs(d2) <= eps:
                    frac = d1 / (d1 - d2) if abs(d1 - d2) > eps else 0.0
                    frac = float(np.clip(frac, 0.0, 1.0))
                    log_x = np.log10(x1) + frac * (np.log10(x2) - np.log10(x1))
                    loss = 10 ** log_x
                    exceedance = y1 + frac * (y2 - y1)
                    tolerance = t1 + frac * (t2 - t1)
                    intersections.append({
                        'loss': float(loss),
                        'exceedance_percent': float(exceedance),
                        'tolerance_percent': float(tolerance),
                        'tolerance_type': normalized['type'],
                    })

            intersections = self._deduplicate_intersections(intersections)
            if not intersections:
                nearest = self._find_nearest_tolerance_point(x_vals, lec_y, tol_y, normalized['type'])
                if nearest is not None:
                    intersections = [nearest]
            results[name] = intersections

        self._intersection_cache = results
        return results


    def _find_nearest_tolerance_point(self, x_vals, lec_y, tol_y, tolerance_type='curve'):
        """Return the closest approach between LEC and tolerance curve.

        This acts as a robust fallback when sampled curves do not produce a
        strict sign change, for example when the curves nearly touch within the
        displayed domain or when the tolerance curve stays consistently above or
        below the LEC over the sampled grid.
        """
        x_vals = np.asarray(x_vals, dtype=float)
        lec_y = np.asarray(lec_y, dtype=float)
        tol_y = np.asarray(tol_y, dtype=float)

        if x_vals.size == 0 or lec_y.size == 0 or tol_y.size == 0:
            return None

        diff = np.abs(lec_y - tol_y)
        idx = int(np.argmin(diff))
        if not np.isfinite(diff[idx]):
            return None

        return {
            'loss': float(x_vals[idx]),
            'exceedance_percent': float(lec_y[idx]),
            'tolerance_percent': float(tol_y[idx]),
            'tolerance_type': tolerance_type,
        }

    def _deduplicate_intersections(self, intersections, rel_tol=1e-6):
        """Remove duplicate intersection points from dense sampling."""
        deduped = []
        for point in intersections:
            if not deduped:
                deduped.append(point)
                continue

            prev = deduped[-1]
            same_loss = np.isclose(point['loss'], prev['loss'], rtol=rel_tol, atol=0.0)
            same_ex = np.isclose(
                point['exceedance_percent'],
                prev['exceedance_percent'],
                rtol=rel_tol,
                atol=1e-6
            )
            if not (same_loss and same_ex):
                deduped.append(point)
        return deduped

    def _interpolate_curve_at_x(self, x_vals, y_vals, x_target):
        """Interpolate the LEC on log-x scale for a target loss."""
        x_vals = np.asarray(x_vals, dtype=float)
        y_vals = np.asarray(y_vals, dtype=float)
        x_target = float(x_target)

        log_x = np.log10(np.clip(x_vals, 1e-12, None))
        log_target = np.log10(max(x_target, 1e-12))

        if log_target <= log_x.min():
            return float(y_vals[0])
        if log_target >= log_x.max():
            return float(y_vals[-1])

        return float(np.interp(log_target, log_x, y_vals))

    def _draw_intersections(self, ax, intersections_by_model):
        """Draw computed intersection points on the LEC axis."""
        any_points = False
        for intersections in intersections_by_model.values():
            if not intersections:
                continue
            any_points = True
            losses = [point['loss'] for point in intersections]
            exceedances = [point['exceedance_percent'] for point in intersections]
            ax.scatter(
                losses,
                exceedances,
                marker='o',
                s=36,
                facecolors='white',
                edgecolors='black',
                linewidths=1.0,
                zorder=6
            )

        if not any_points:
            return None

        return Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor='white',
            markeredgecolor='black',
            markersize=6
        )

    def _add_shared_legend(self, fig, legend_handles, legend_labels):
        """Render a single shared legend for both EPC and LEC."""
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='center left',
            bbox_to_anchor=(0.84, 0.5),
            ncol=1,
            frameon=False,
            fontsize=9,
            handlelength=1.5,
            handletextpad=0.6,
            labelspacing=0.4,
            borderaxespad=0.0
        )

    def get_var_table(self):
        """Return a VaR table for key percentiles.

        The table is based on the simulated Risk values for each model.
        """
        var_data = {}
        for name, model in self._input.items():
            risk = model.export_results()['Risk']
            var_data[name] = {
                'P20': np.quantile(risk, 0.2),
                'P50': np.quantile(risk, 0.5),
                'P80': np.quantile(risk, 0.8),
            }

        return pd.DataFrame(var_data).T

    def _get_prob_data(self, space, risk):
        """Get the percentile score for each risk value."""
        quantiles = space.map(lambda x: stats.percentileofscore(risk, x))
        return (quantiles, space)

    def _get_loss_data(self, space, risk):
        """Get percentage of values exceeding each loss value."""
        loss_ex = space.map(lambda value: (value < risk).mean())
        return (space, loss_ex * 100)

    def _generate_prob_curve(self, name, ax, quantiles, space, color):
        """For each percentile, what is the expected loss?"""
        line, = ax.plot(quantiles, space, color=color)

        y_formatter = matplotlib.ticker.StrMethodFormatter(self._currency_prefix + '{x:,.0f}')
        ax.axes.yaxis.set_major_formatter(y_formatter)
        x_formatter = matplotlib.ticker.StrMethodFormatter('{x:,.0f}%')
        ax.axes.xaxis.set_major_formatter(x_formatter)
        ax.axes.set_title('Exceedence Probability Curve', fontsize=20)

        return line

    def _generate_loss_curve(self, name, ax, space, loss_expectancy, quantile_values, color):
        """For each dollar amount, what's the probability loss was exceeded?"""
        ax.plot(space, loss_expectancy, color=color)

        for p, value in quantile_values.items():
            ax.axvline(
                x=value,
                color=color,
                linestyle=':',
                linewidth=0.9,
                alpha=0.45
            )

        ax.set_xscale('log')
        ax.set_xlim(space.min(), space.max())
        ax.set_ylim(0, 100)
        ax.margins(x=0, y=0)

        ax.axes.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}%'))
        ax.axes.xaxis.set_major_formatter(StrMethodFormatter(self._currency_prefix + '{x:,.0f}'))
        ax.axes.xaxis.set_tick_params(rotation=-35)

        for label in ax.get_xticklabels():
            label.set_horizontalalignment('left')

        ax.axes.set_title('Loss Exceedence Curve', fontsize=20)
