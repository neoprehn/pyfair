"""Base report class for creating HTML reports."""

import base64
import datetime
import getpass
import io
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import VERSION
from ..utility.fair_exception import FairException
from .distribution import FairDistributionCurve
from .exceedence import FairExceedenceCurves
from .tree_graph import FairTreeGraph
from .violin import FairViolinPlot


class FairBaseReport(object):
    """A base class for creating FairModel and FairMetaModel reports.

    This class provides shared reporting utilities for HTML-based reports.
    It supports both legacy flat parameter inputs and the newer structured
    distribution metadata introduced for richer FAIR parameter definitions.
    """

    def __init__(self, currency_prefix='$'):
        # Basic formatting configuration.
        self._currency_prefix = currency_prefix
        self._model_or_models = None
        self._currency_format_string = currency_prefix + '{0:,.02f}'
        self._float_format_string = '{0:.4f}'

        # FAIR factor specific formatting.
        self._format_strings = {
            'Risk': self._currency_format_string,
            'Loss Event Frequency': self._float_format_string,
            'Threat Event Frequency': self._float_format_string,
            'Vulnerability': self._float_format_string,
            'Contact Frequency': self._float_format_string,
            'Probability of Action': self._float_format_string,
            'Threat Capability': self._float_format_string,
            'Control Strength': self._float_format_string,
            'Loss Magnitude': self._currency_format_string,
            'Primary Loss': self._currency_format_string,
            'Secondary Loss': self._currency_format_string,
            'Secondary Loss Event Frequency': self._float_format_string,
            'Secondary Loss Event Magnitude': self._currency_format_string,
        }

        # Static/report resources.
        self._fair_location = pathlib.Path(__file__).parent.parent
        self._static_location = self._fair_location / 'static'
        self._logo_location = self._static_location / 'white_python_logo.png'
        self._template_paths = {
            'css': self._static_location / 'fair.css',
            'simple': self._static_location / 'simple.html',
        }

        # Legacy parameter columns retained for backward compatibility.
        self._legacy_param_cols = [
            'low',
            'most_likely',
            'high',
            'constant',
            'mean',
            'stdev',
        ]

        # Preferred column order for the model parameter table.
        self._structured_param_cols = [
            'distribution',
            'distribution_name',
            'parameters',
            'confidence',
            'mean',
            'stdev',
            'min',
            'max',
        ]

    def _input_check(self, value):
        """Check whether the input is a model or iterable of models.

        Raises
        ------
        FairException
            If an inappropriate object or iterable of objects is supplied.
        """
        rv = {}

        # Single FairModel / FairMetaModel.
        if value.__class__.__name__ in ['FairModel', 'FairMetaModel']:
            rv[value.get_name()] = value
            return rv

        # Iterable check.
        if not hasattr(value, '__iter__'):
            raise FairException(
                'Input is not a FairModel, FairMetaModel, or an iterable.'
            )

        if len(value) == 0:
            raise FairException(
                'Empty iterable where iterable of models expected.'
            )

        # Validate iterable members.
        for purported_model in value:
            if purported_model.__class__.__name__ in ['FairModel', 'FairMetaModel']:
                if purported_model.calculation_completed():
                    rv[purported_model.get_name()] = purported_model
                else:
                    raise FairException(
                        'Model or FairModel has not been calculated.'
                    )
            else:
                raise FairException(
                    'Iterable member is not a FairModel or FairMetaModel'
                )

        return rv

    def get_format_strings(self):
        """Return FAIR factor specific formatting strings."""
        return self._format_strings

    def base64ify(self, image, alternative_text='', options=''):
        """Convert binary data into an embeddable HTML <img> tag."""
        if type(image) == str or isinstance(image, pathlib.Path):
            with open(image, 'rb') as f:
                binary_data = f.read()
        elif type(image) == bytes:
            binary_data = image
        else:
            raise TypeError(
                str(image) + ' is not a string, path, or bytes.'
            )

        base64_string = base64.b64encode(binary_data).decode('utf8')
        tag = (
            f'<img {options} '
            f'src="data:image/png;base64, {base64_string}" '
            f'alt="{alternative_text}"/>'
        )
        return tag

    def _safe_filename(self, value):
        """Convert a model/report name into a filesystem-friendly filename."""
        return ''.join(
            char if char.isalnum() or char in ('-', '_') else '_'
            for char in str(value)
        )

    def _construct_output(self):
        """Stub to be overridden by subclasses."""
        raise NotImplementedError()

    def to_html(self, output_path, export_csv=False, csv_dir=None):
        """Write the generated HTML report to disk.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Output HTML file path.
        export_csv : bool, optional
            If True, export CSV files for contained models/metamodels.
        csv_dir : str or pathlib.Path, optional
            Target directory for CSV export. If omitted, the HTML file's
            parent directory is used.
        """
        output = self._construct_output()
        output_path = pathlib.Path(output_path)

        with open(output_path, 'w+', encoding='utf-8') as f:
            f.write(output)

        if export_csv:
            target_dir = output_path.parent if csv_dir is None else pathlib.Path(csv_dir)
            self.to_csv(output_dir=target_dir)

    def to_csv(self, output_dir='.', sep=';', decimal=',', index=False):
        """Export report model results as CSV files.

        One CSV file is written per contained model/metamodel.

        Parameters
        ----------
        output_dir : str or pathlib.Path, optional
            Target directory for CSV files.
        sep : str, optional
            CSV separator.
        decimal : str, optional
            Decimal separator.
        index : bool, optional
            Whether to include the dataframe index.

        Returns
        -------
        list[str]
            List of written CSV file paths.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written_files = []

        for name, model in self._model_or_models.items():
            filename = self._safe_filename(name) + '.csv'
            output_path = output_dir / filename

            # FairModel may provide a dedicated CSV export method.
            if hasattr(model, 'export_results_csv'):
                model.export_results_csv(
                    output_path=output_path,
                    sep=sep,
                    decimal=decimal,
                    index=index
                )
            else:
                model.export_results().to_csv(
                    output_path,
                    sep=sep,
                    decimal=decimal,
                    index=index
                )

            written_files.append(str(output_path))

        return written_files

    def _fig_to_img_tag(self, fig):
        """Convert a matplotlib figure to a base64 HTML image tag.

        The figure is explicitly closed after conversion to avoid
        accumulating open matplotlib figures during reports or tests.
        """
        data = io.BytesIO()
        fig.savefig(data, format='png', transparent=True)
        data.seek(0)
        img_tag = self.base64ify(data.read())

        # Close the figure explicitly to avoid matplotlib warnings
        # about too many open figures.
        plt.close(fig)

        return img_tag

    def _get_data_table(self, model):
        """Generate an HTML table from a model's results."""
        data = model.export_results().dropna(axis=1)
        table = data.to_html(
            border=0,
            justify='left',
            classes='fair_metadata_table',
        )
        return table

    def _get_parameter_table(self, model):
        """Return raw parameter metadata from the model."""
        # Keep this legacy hook as-is because external code may rely on it.
        if hasattr(model, 'export_parameters'):
            return model.export_parameters()
        if hasattr(model, 'export_params'):
            return model.export_params()
        raise FairException('Model does not support parameter export.')

    def _get_metadata_table(self):
        """Generate report-level metadata."""
        try:
            username = getpass.getuser()
        except Exception:
            username = 'Unknown'

        metadata = pd.Series({
            'Author': username,
            'Created': str(datetime.datetime.now()).partition('.')[0],
            'PyFair Version': VERSION,
            'Type': type(self).__name__,
        }).to_frame().to_html(
            border=0,
            header=None,
            justify='left',
            classes='fair_metadata_table',
        )
        return metadata


    def _model_has_risk(self, model):
        """Return True if the exported results include a Risk column."""
        try:
            return 'Risk' in model.export_results().columns
        except Exception:
            return False

    def _get_risk_models(self, model_or_models):
        """Return only models whose exported results contain a Risk column."""
        if isinstance(model_or_models, dict):
            return {
                name: model
                for name, model in model_or_models.items()
                if self._model_has_risk(model)
            }

        return [model for model in model_or_models if self._model_has_risk(model)]

    def _is_compare_metamodel(self, model):
        """Return True for FairMetaModel instances in compare mode."""
        return (
            model.__class__.__name__ == 'FairMetaModel' and
            getattr(model, '_mode', 'sum') == 'compare'
        )

    def _get_tree(self, model):
        """Create a base64 image string using FairTreeGraph."""
        ftg = FairTreeGraph(model, self._format_strings)
        fig, ax = ftg.generate_image()
        img_tag = self._fig_to_img_tag(fig)
        return img_tag

    def _get_distribution(self, model_or_models, currency_prefix):
        """Create a base64 image string using FairDistributionCurve."""
        fdc = FairDistributionCurve(model_or_models, currency_prefix)
        fig, ax = fdc.generate_image()
        img_tag = self._fig_to_img_tag(fig)
        return img_tag

    def _get_distribution_icon(self, model, target):
        """Create a miniature distribution icon for a given target."""
        fdc = FairDistributionCurve(model, self._currency_prefix)
        fig, ax = fdc.generate_icon(model.get_name(), target)
        img_tag = self._fig_to_img_tag(fig)
        return img_tag

    def _get_exceedence_curves(self, model_or_models, currency_prefix, risk_tolerance=None):
        """Create a base64 image string using FairExceedenceCurves."""
        fec = FairExceedenceCurves(
            model_or_models,
            currency_prefix,
            risk_tolerance=risk_tolerance
        )
        fig, ax = fec.generate_image()
        img_tag = self._fig_to_img_tag(fig)
        return img_tag

    def _get_exceedence_curve_object(self, model_or_models, currency_prefix, risk_tolerance=None):
        """Return a FairExceedenceCurves object for table/report helpers."""
        fec = FairExceedenceCurves(
            model_or_models,
            currency_prefix,
            risk_tolerance=risk_tolerance
        )
        return fec

    def _get_tolerance_intersection_table(self, curve_obj_or_models, risk_tolerance=None):
        """Create an HTML table for LEC/tolerance intersections.

        Supports both:
        1. new style: _get_tolerance_intersection_table(curve_obj)
        2. old style: _get_tolerance_intersection_table(model_or_models, risk_tolerance=...)
        """
        if hasattr(curve_obj_or_models, 'get_tolerance_intersections'):
            fec = curve_obj_or_models
        else:
            if not risk_tolerance:
                return ''

            fec = FairExceedenceCurves(
                curve_obj_or_models,
                self._currency_prefix,
                risk_tolerance=risk_tolerance
            )

        intersections = fec.get_tolerance_intersections()
        if intersections.empty:
            return (
                "<div class='tolerance-intersection-block'>"
                "<h3>Risk Tolerance Intersection</h3>"
                "<p>No intersection found within the displayed LEC range.</p>"
                "</div>"
            )

        intersections = intersections.copy()
        intersections['Loss'] = intersections['Loss'].map(
            lambda x: self._format_strings['Risk'].format(x)
        )
        intersections['Exceedance %'] = intersections['Exceedance %'].map(
            lambda x: f'{x:,.2f}%'
        )
        intersections['Tolerance %'] = intersections['Tolerance %'].map(
            lambda x: f'{x:,.2f}%'
        )

        table_html = intersections.to_html(
            border=0,
            header=True,
            index=False,
            justify='left',
            classes='fair_table tolerance_intersection_table',
        )

        return (
            "<div class='tolerance-intersection-block'>"
            "<h3>Risk Tolerance Intersection</h3>"
            f"{table_html}"
            "</div>"
        )

    def _get_var_table(self, model_or_models):
        """Create an HTML VaR table for key percentiles."""
        var_data = {}

        for model in model_or_models:
            risk = model.export_results()['Risk']
            var_data[model.get_name()] = {
                'VaR 10%': np.quantile(risk, 0.10),
                'VaR 20%': np.quantile(risk, 0.20),
                'VaR 50%': np.quantile(risk, 0.50),
                'VaR 80%': np.quantile(risk, 0.80),
                'VaR 90%': np.quantile(risk, 0.90),
                'VaR 95%': np.quantile(risk, 0.95),
                'VaR 99%': np.quantile(risk, 0.99),
            }

        var_df = pd.DataFrame(var_data).T
        var_df.index.name = 'Model'

        # Model index into a real first column
        var_df = var_df.reset_index()

        # Format only numeric VaR columns, not the Model column
        for col in var_df.columns:
            if col != 'Model':
                var_df[col] = var_df[col].map(
                    lambda x: self._format_strings['Risk'].format(x)
                )

        table_html = var_df.to_html(
            border=0,
            header=True,
            index=False,
            justify='left',
            classes='fair_table var_table',
        )
        return table_html

    def _get_violins(self, metamodel):
        """Create a base64 image string using FairViolinPlot."""
        vplot = FairViolinPlot(metamodel)
        fig, ax = vplot.generate_image()
        img_tag = self._fig_to_img_tag(fig)
        return img_tag

    def _get_overview_table(self, model_or_models):
        """Create a risk overview table using one or more models.

        Compare-mode metamodels are excluded here because they do not expose
        an aggregate Risk column by design.
        """
        risk_models = self._get_risk_models(model_or_models)
        if not risk_models:
            return (
                "<p>No aggregate Risk column available for overview. "
                "Compare-mode metamodels are shown in the comparison section below.</p>"
            )

        try:
            risk_results = pd.DataFrame({
                name: model.export_results()['Risk']
                for name, model in risk_models.items()
            })
        except KeyError:
            raise FairException(
                "No 'Risk' key. Model likely uncalculated."
            )

        agg = risk_results.agg(['mean', 'std', 'min', 'max'])
        agg.index = ['Mean', 'Stdev', 'Minimum', 'Maximum']
        risk_results = agg.copy()

        overview_df = risk_results.map(
            lambda x: self._format_strings['Risk'].format(x)
        )

        overview_df.loc['Simulations'] = [
            '{0:,.0f}'.format(len(model.export_results()))
            for model in risk_models.values()
        ]
        overview_df.loc['Identifier'] = [
            model.get_uuid()
            for model in risk_models.values()
        ]
        overview_df.loc['Model Type'] = [
            model.__class__.__name__
            for model in risk_models.values()
        ]

        skipped = [
            name for name, model in model_or_models.items()
            if not self._model_has_risk(model)
        ]
        if skipped:
            overview_df.loc['Excluded From Overview'] = [
                '' for _ in risk_models.values()
            ]
            overview_df.iloc[-1, 0] = ', '.join(skipped)

        overview_html = overview_df.to_html(
            border=0,
            header=True,
            justify='left',
            classes='fair_table',
        )
        return overview_html

    def _is_structured_param_record(self, value):
        """Return True if a parameter record uses the new structured format."""
        return (
            isinstance(value, dict) and
            'distribution' in value and
            'params' in value
        )

    def _stringify_params(self, params):
        """Convert a parameter dictionary into a readable string.

        Numeric values are formatted with four decimal places.
        """
        if not isinstance(params, dict) or len(params) == 0:
            return ''

        parts = []
        for key, value in params.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                parts.append(f'{key}={value:.2f}')
            else:
                parts.append(f'{key}={value}')
        return ', '.join(parts)

    def _flatten_param_record(self, target, value):
        """Flatten one parameter record into a report-friendly row.

        Supports:
        1. legacy flat parameter dictionaries
        2. new structured dictionaries with distribution/params/confidence
        3. raw-supplied values
        """
        row = {
            'target': target,
            'distribution_name': '',
            'confidence': '',
            'parameters': '',
        }

        # Structured new API.
        if self._is_structured_param_record(value):
            distribution = value.get('distribution', '')
            params = value.get('params', {}) or {}
            confidence = value.get('confidence', None)

            row['distribution_name'] = distribution
            row['confidence'] = '' if confidence is None else str(confidence)
            row['parameters'] = self._stringify_params(params)
            return row

        # Raw data supplied directly.
        if isinstance(value, dict) and 'raw' in value:
            raw_values = value.get('raw', [])
            row['distribution_name'] = 'raw'
            row['confidence'] = ''
            row['parameters'] = f'n={len(raw_values)} raw values'
            return row

        # Legacy flat record.
        if isinstance(value, dict):
            row['distribution_name'] = 'legacy'

            # Map a legacy record to a compact string.
            visible_items = []
            for col in self._legacy_param_cols:
                if col in value and pd.notnull(value[col]):
                    visible_items.append(f'{col}={value[col]}')

            # Include any additional legacy keys that are not in the standard list.
            extra_keys = [
                key for key in value.keys()
                if key not in self._legacy_param_cols and key != 'raw'
            ]
            for key in extra_keys:
                item = value.get(key)
                if pd.notnull(item):
                    visible_items.append(f'{key}={item}')

            row['parameters'] = ', '.join(visible_items)
            return row

        # Fallback for unexpected values.
        row['distribution_name'] = 'unknown'
        row['parameters'] = str(value)
        return row

    def _safe_format(self, target, value):
        """Format a value using the FAIR-factor-specific formatter."""
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

    def _decorate_target_label(self, target, node_statuses=None):
        """Append a calculated marker for FAIR nodes derived by the model."""
        if node_statuses is None:
            return target

        status = node_statuses.get(target)
        if status == 'Calculated':
            return f'{target} (calculated)'
        return target

    def _get_model_parameter_table(self, model):
        """Generate a model parameter table with statistics and icons.

        This supports both:
        - legacy flat parameter storage
        - structured distribution metadata storage

        In addition, calculated-only FAIR nodes that exist in export_results()
        but not in export_params() are added to the output table and marked as
        '(calculated)' when applicable.
        """
        # Obtain parameter metadata from the model.
        if hasattr(model, 'export_params'):
            params = dict(**model.export_params())
        elif hasattr(model, 'export_parameters'):
            params = dict(**model.export_parameters())
        else:
            raise FairException('Model does not support parameter export.')

        # Keep only FAIR nodes that have report formatting rules.
        params = {
            key: value
            for key, value in params.items()
            if key in self._format_strings.keys()
        }

        # Create one row per FAIR factor from exported parameters.
        rows = []
        for target, value in params.items():
            rows.append(self._flatten_param_record(target, value))

        # Start with parameter dataframe.
        if len(rows) == 0:
            param_df = pd.DataFrame(columns=['target'])
        else:
            param_df = pd.DataFrame(rows)

        # Ensure structured fields exist explicitly.
        if 'distribution_name' not in param_df.columns:
            param_df['distribution_name'] = ''
        if 'parameters' not in param_df.columns:
            param_df['parameters'] = ''
        if 'confidence' not in param_df.columns:
            param_df['confidence'] = ''

        # Obtain simulation results.
        results = model.export_results()

        # Only keep result targets that have FAIR formatting rules.
        result_targets = [
            target for target in results.columns
            if target in self._format_strings.keys()
        ]

        summary_df = pd.DataFrame({
            'target': result_targets,
            'mean': [results[target].mean() for target in result_targets],
            'stdev': [results[target].std() for target in result_targets],
            'min': [results[target].min() for target in result_targets],
            'max': [results[target].max() for target in result_targets],
        })

        # Add rows for calculated-only targets that are present in results
        # but absent from exported input parameters.
        existing_targets = set(param_df['target']) if 'target' in param_df.columns else set()
        missing_targets = [
            target for target in result_targets
            if target not in existing_targets
        ]

        if missing_targets:
            missing_df = pd.DataFrame({
                'target': missing_targets,
                'distribution_name': [''] * len(missing_targets),
                'parameters': [''] * len(missing_targets),
                'confidence': [''] * len(missing_targets),
            })
            param_df = pd.concat([param_df, missing_df], ignore_index=True)

        # Merge statistics on original target names.
        param_df = param_df.merge(summary_df, on='target', how='left')

        # Get node statuses for calculated markers.
        node_statuses = {}
        if hasattr(model, 'get_node_statuses'):
            try:
                node_statuses = model.get_node_statuses() or {}
            except Exception:
                node_statuses = {}

        # Format numeric values according to FAIR factor formatting.
        for stat_col in ['mean', 'stdev', 'min', 'max']:
            param_df[stat_col] = [
                self._safe_format(target, value)
                for target, value in zip(param_df['target'], param_df[stat_col])
            ]

        # Decorate target labels only after merge/formatting.
        if 'target' in param_df.columns:
            param_df['target'] = [
                self._decorate_target_label(target, node_statuses)
                for target in param_df['target']
            ]

        # Keep only rows that were either supplied or calculated.
        # A row is considered "supplied" if at least one metadata field is non-empty.
        # A row is considered "calculated" if at least one summary statistic is present.
        def _has_content(value):
            if pd.isnull(value):
                return False
            if isinstance(value, str):
                return value.strip() != ''
            return True

        supplied_mask = param_df.apply(
            lambda row: any(
                _has_content(row.get(col, ''))
                for col in ['distribution_name', 'parameters', 'confidence']
            ),
            axis=1
        )

        calculated_mask = param_df.apply(
            lambda row: any(
                _has_content(row.get(col, ''))
                for col in ['mean', 'stdev', 'min', 'max']
            ),
            axis=1
        )

        param_df = param_df[supplied_mask | calculated_mask].copy()

        # Preserve full base64 image strings in HTML.
        pd.set_option('display.max_colwidth', None)

        display_cols = [
            'target',
            'distribution_name',
            'parameters',
            'confidence',
            'mean',
            'stdev',
            'min',
            'max',
        ]

        for col in display_cols:
            if col not in param_df.columns:
                param_df[col] = ''

        param_df = param_df[display_cols]

        param_df = param_df.rename(columns={
            'target': 'Target',
            'distribution_name': 'Distribution',
            'parameters': 'Parameters',
            'confidence': 'Confidence',
            'mean': 'Mean',
            'stdev': 'Stdev',
            'min': 'Min',
            'max': 'Max',
        })

        detail_table = param_df.to_html(
            border=0,
            header=True,
            index=False,
            justify='left',
            classes='fair_table',
            escape=False,
        )
        return detail_table

    def _get_metamodel_parameter_table(self, metamodel):
        """Create a table for a metamodel."""
        risk_df = metamodel.export_results().T
        risk_df = pd.DataFrame({
            'mean': risk_df.mean(axis=1),
            'stdev': risk_df.std(axis=1),
            'min': risk_df.min(axis=1),
            'max': risk_df.max(axis=1),
        })

        risk_df = risk_df.apply(
            lambda row: pd.Series(
                [
                    self._format_strings['Risk'].format(item)
                    for item in row
                ],
                index=row.index.values,
            ),
            axis=1,
        )

        detail_table = risk_df.to_html(
            border=0,
            header=True,
            justify='left',
            classes='fair_table',
            escape=False,
        )
        return detail_table

    def _get_metamodel_compare_table(self, metamodel):
        """Create a delta summary table for compare-mode metamodels."""
        results = metamodel.export_results().copy()
        delta_columns = [c for c in results.columns if str(c).startswith('Delta::')]
        if not delta_columns:
            return ''

        rows = []
        for col in delta_columns:
            series = results[col]
            rows.append({
                'Comparison': str(col).replace('Delta::', ''),
                'Mean Delta': series.mean(),
                'Stdev Delta': series.std(),
                'Min Delta': series.min(),
                'Max Delta': series.max(),
                'VaR 50 Delta': np.quantile(series, 0.50),
                'VaR 90 Delta': np.quantile(series, 0.90),
                'VaR 95 Delta': np.quantile(series, 0.95),
                'VaR 99 Delta': np.quantile(series, 0.99),
            })

        df = pd.DataFrame(rows)
        numeric_cols = [c for c in df.columns if c != 'Comparison']
        for col in numeric_cols:
            df[col] = df[col].map(lambda x: self._format_strings['Risk'].format(x))

        baseline = getattr(metamodel, '_baseline_model', None)
        header = '<h3>Meta Risk Comparison</h3>'
        if baseline:
            header += f'<p><strong>Baseline model:</strong> {baseline}</p>'

        return header + df.to_html(
            border=0,
            header=True,
            index=False,
            justify='left',
            classes='fair_table',
            escape=False,
        )

    def _get_metamodel_compare_summary_box(self, metamodel):
        """Render a compact summary box when only one comparison exists."""
        results = metamodel.export_results().copy()
        delta_columns = [c for c in results.columns if str(c).startswith('Delta::')]
        if len(delta_columns) != 1:
            return ''

        col = delta_columns[0]
        label = str(col).replace('Delta::', '')
        series = results[col]
        baseline = getattr(metamodel, '_baseline_model', None)

        def fmt(value):
            return self._format_strings['Risk'].format(value)

        header = '<h3>Compare Summary</h3>'
        if baseline:
            header += f'<p><strong>Baseline model:</strong> {baseline}</p>'

        html = (
            f"{header}"
            "<table class='fair_table'>"
            "<tr><th>Comparison</th><td>{}</td></tr>"
            "<tr><th>Mean Delta</th><td>{}</td></tr>"
            "<tr><th>VaR 95 Delta</th><td>{}</td></tr>"
            "<tr><th>Min Delta</th><td>{}</td></tr>"
            "<tr><th>Max Delta</th><td>{}</td></tr>"
            "</table>"
        ).format(
            label,
            fmt(series.mean()),
            fmt(np.quantile(series, 0.95)),
            fmt(series.min()),
            fmt(series.max()),
        )
        return html

    def _get_metamodel_compare_bar_plot(self, metamodel, metric='VaR 95 Delta'):
        """Create a sorted horizontal bar plot for compare-mode metamodel deltas."""
        results = metamodel.export_results().copy()
        delta_columns = [c for c in results.columns if str(c).startswith('Delta::')]
        # A bar plot is only helpful when at least two comparisons exist.
        if len(delta_columns) < 2:
            return ''

        metric_map = {
            'Mean Delta': lambda s: s.mean(),
            'VaR 95 Delta': lambda s: np.quantile(s, 0.95),
        }
        reducer = metric_map.get(metric, metric_map['VaR 95 Delta'])

        rows = []
        for col in delta_columns:
            label = str(col).replace('Delta::', '')
            value = reducer(results[col])
            rows.append({
                'Comparison': label,
                'Value': value,
                'AbsValue': abs(value),
            })

        plot_df = pd.DataFrame(rows)
        if plot_df.empty:
            return ''

        plot_df = plot_df.sort_values('AbsValue', ascending=True).reset_index(drop=True)

        labels = plot_df['Comparison'].tolist()
        values = plot_df['Value'].tolist()

        baseline = getattr(metamodel, '_baseline_model', None)
        if baseline:
            title = f'{metric} vs. baseline: {baseline}'
        else:
            title = f'{metric} comparison'

        fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(labels) + 1.5)))
        y_pos = np.arange(len(labels))

        ax.barh(y_pos, values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel(metric)
        ax.set_title(title)
        ax.axvline(0.0, linewidth=1)

        ax.xaxis.set_major_formatter(
            plt.matplotlib.ticker.StrMethodFormatter(self._currency_prefix + '{x:,.0f}')
        )

        fig.tight_layout()
        return self._fig_to_img_tag(fig)

    def _get_risk_tolerance_details_table(self, curve_obj):
        """Return an HTML table with risk tolerance details."""
        details = curve_obj.get_risk_tolerance_details()

        if not details:
            return ''

        rows = []

        tol_type = details.get('type', '')
        rows.append(
            "<tr><th>Tolerance Type</th><td>{}</td></tr>".format(tol_type)
        )

        if tol_type == 'constant':
            value = details.get('value')
            rows.append(
                "<tr><th>Value</th><td>{}{:,.2f}</td></tr>".format(
                    self._currency_prefix,
                    value
                )
            )

        elif tol_type == 'curve':
            n_points = details.get('n_points', 0)
            rows.append(
                "<tr><th>Points</th><td>{}</td></tr>".format(n_points)
            )

            points = details.get('points', [])
            if points:
                point_strings = []
                for p in points:
                    point_strings.append(
                        "{}{:,.2f} @ {:,.2f}%".format(
                            self._currency_prefix,
                            p['loss'],
                            p['exceedance_percent']
                        )
                    )

                rows.append(
                    "<tr><th>Curve Points</th><td>{}</td></tr>".format(
                        "<br>".join(point_strings)
                    )
                )

        elif tol_type == 'distribution':
            distribution = details.get('distribution', '')
            params = details.get('params', {})
            samples = details.get('samples', '')

            rows.append(
                "<tr><th>Distribution</th><td>{}</td></tr>".format(distribution)
            )

            if params:
                param_str = self._stringify_params(params)
                rows.append(
                    "<tr><th>Parameters</th><td>{}</td></tr>".format(param_str)
                )

            rows.append(
                "<tr><th>Samples</th><td>{}</td></tr>".format(samples)
            )

        return (
            "<div class='tolerance-details-block'>"
            "<h3>Risk Tolerance Details</h3>"
            "<table class='fair_table tolerance_details_table'>"
            "<tbody>"
            f"{''.join(rows)}"
            "</tbody>"
            "</table>"
            "</div>"
        )
    def _get_loss_frequency_magnitude_scatter(self, model):
        """Create a log-log scatter plot for LEF vs. primary/secondary loss.

        Blue points represent Primary Loss, red points represent Secondary Loss.
        A small multiplicative x-jitter is applied to reduce vertical striping
        when LEF values occur in coarse/discrete steps.
        """
        if model.__class__.__name__ != 'FairModel':
            return ''

        try:
            results = model.export_results().copy()
        except Exception:
            return ''

        required_cols = ['Loss Event Frequency', 'Primary Loss', 'Secondary Loss']
        if any(col not in results.columns for col in required_cols):
            return ''

        lef = pd.to_numeric(results['Loss Event Frequency'], errors='coerce')
        primary = pd.to_numeric(results['Primary Loss'], errors='coerce')
        secondary = pd.to_numeric(results['Secondary Loss'], errors='coerce')

        mask_primary = lef.gt(0) & primary.gt(0)
        mask_secondary = lef.gt(0) & secondary.gt(0)

        if not mask_primary.any() and not mask_secondary.any():
            return ''

        # Small multiplicative jitter for log-scale x-axis.
        # This preserves order and scale much better than additive jitter.
        rng = np.random.default_rng(42)
        jitter_strength = 0.08  # ~6% log-scale jitter

        def _jitter_x(values):
            values = np.asarray(values, dtype=float)
            factors = np.exp(rng.normal(loc=0.0, scale=jitter_strength, size=len(values)))
            jittered = values * factors

            # Keep values strictly positive for log-scale plotting.
            tiny = np.finfo(float).tiny
            jittered[jittered <= 0] = tiny
            return jittered

        fig, ax = plt.subplots(figsize=(5.6, 4.6))

        if mask_primary.any():
            lef_primary = lef[mask_primary].to_numpy(dtype=float)
            primary_vals = primary[mask_primary].to_numpy(dtype=float)
            lef_primary_jittered = _jitter_x(lef_primary)

            ax.scatter(
                lef_primary_jittered,
                primary_vals,
                s=6,
                alpha=0.22,
                color='blue',
                label='Primary',
            )
            ax.scatter(
                [lef_primary.mean()],
                [primary_vals.mean()],
                s=28,
                color='black',
                zorder=5,
            )

        if mask_secondary.any():
            lef_secondary = lef[mask_secondary].to_numpy(dtype=float)
            secondary_vals = secondary[mask_secondary].to_numpy(dtype=float)
            lef_secondary_jittered = _jitter_x(lef_secondary)

            ax.scatter(
                lef_secondary_jittered,
                secondary_vals,
                s=6,
                alpha=0.18,
                color='red',
                label='Secondary',
            )
            ax.scatter(
                [lef_secondary.mean()],
                [secondary_vals.mean()],
                s=28,
                color='black',
                zorder=5,
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Loss Event Frequency')
        ax.set_ylabel('Loss Magnitude')
        ax.set_title('Loss Frequency vs. Loss Magnitude')
        ax.legend(loc='upper right')
        ax.grid(True, which='both', alpha=0.25)

        fig.tight_layout()
        return self._fig_to_img_tag(fig)
