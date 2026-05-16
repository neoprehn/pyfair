"""Simple report for demonstrating aggregate risk"""

import pandas as pd

from .base_report import FairBaseReport


class FairSimpleReport(FairBaseReport):
    """A subclass for creating the report HTML

    This class is responsible for implementing the _construct_output()
    method. The method is takes the template and css for the simple report
    and plugs in the appropriate data base on the models supplied.

    Parameters
    ----------
    currency_prefix : str
        The currency symbol in front of your (default: $)

    Examples
    --------
    >>> m1 = pyfair.model.FairModel.from_json('model_1.json')
    >>> m2 = pyfair.model.FairModel.from_json('model_2.json')
    >>> fsr = FairSimpleReport([m1, m2], currency_prefix='元')
    >>> fsr.generate_html('output.html')

    """
    def __init__(self, model_or_models, currency_prefix='$', risk_tolerance=None):
        super().__init__(currency_prefix=currency_prefix)
        self._currency_prefix = currency_prefix
        self._model_or_models = self._input_check(model_or_models)
        self._risk_tolerance = risk_tolerance
        self._css = self._template_paths['css'].read_text()
        self._template = self._template_paths['simple'].read_text()

    def _construct_output(self):
        """HTML creation function called by FairBaseReport.to_html()."""
        t = self._template
        t = t.replace('{STYLE}', self._css)
        t = t.replace('{METADATA}', self._get_metadata_table())
        b64 = self.base64ify(self._logo_location)
        t = t.replace('{PYTHON_LOGO}', b64)

        risk_models = self._get_risk_models(self._model_or_models)

        overview_html = self._get_overview_table(self._model_or_models)
        overview_block = (
            "<h3>Risk Summary</h3>"
            f"{overview_html}"
        )
        t = t.replace('{OVERVIEW_DATAFRAME}', overview_block)

        if risk_models:
            hist = self._get_distribution(
                risk_models.values(),
                currency_prefix=self._currency_prefix
            )

            exceed = self._get_exceedence_curves(
                risk_models.values(),
                currency_prefix=self._currency_prefix,
                risk_tolerance=self._risk_tolerance
            )

            var_table = self._get_var_table(
                risk_models.values()
            )

            curve_obj = self._get_exceedence_curve_object(
                risk_models.values(),
                currency_prefix=self._currency_prefix,
                risk_tolerance=self._risk_tolerance
            )

            tolerance_intersection_table = self._get_tolerance_intersection_table(curve_obj)
            tolerance_details_table = self._get_risk_tolerance_details_table(curve_obj)

            exceed_block = (
                "<div class='exceedence-block'>"
                f"{exceed}"
                "<div class='var-table-block'>"
                f"{var_table}"
                f"{tolerance_intersection_table}"
                f"{tolerance_details_table}"
                "</div>"
                "</div>"
            )
        else:
            hist = '<p>No aggregate Risk distribution available for the selected objects.</p>'
            exceed_block = '<p>No aggregate Risk exceedance output available for the selected objects.</p>'

        t = t.replace('{HIST}', hist)
        t = t.replace('{EXCEEDENCE}', exceed_block)
        t = t.replace('{VAR_TABLE}', '')

        parameter_html = ''
        for name, model in self._model_or_models.items():
            parameter_html += "<h1>{}</h1>".format(name)

            if model.__class__.__name__ == 'FairModel':
                tree_html = self._get_tree(model)
                table_html = self._get_model_parameter_table(model)
                scatter_html = self._get_loss_frequency_magnitude_scatter(model)

                if tree_html:
                    parameter_html += "<h3>FAIR Tree</h3>"
                    parameter_html += tree_html

                if scatter_html:
                    parameter_html += (
                        "<div style='display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap;'>"
                        "<div style='flex:1 1 460px; min-width:420px;'>"
                        f"{table_html}"
                        "</div>"
                        "<div style='flex:1 1 460px; min-width:420px;'>"
                        "<h3>Loss Frequency vs. Loss Magnitude</h3>"
                        f"{scatter_html}"
                        "</div>"
                        "</div>"
                    )
                else:
                    parameter_html += table_html
            if model.__class__.__name__ == 'FairMetaModel':
                parameter_html += self._get_violins(model)
            if model.__class__.__name__ == 'FairMetaModel':
                parameter_html += self._get_metamodel_parameter_table(model)
                if self._is_compare_metamodel(model):
                    parameter_html += self._get_metamodel_compare_table(model)

                    summary_box = self._get_metamodel_compare_summary_box(model)
                    if summary_box:
                        parameter_html += summary_box

                    mean_plot = self._get_metamodel_compare_bar_plot(model, metric='Mean Delta')
                    if mean_plot:
                        parameter_html += "<h3>Compare Bar Plot - Mean Delta</h3>"
                        parameter_html += mean_plot

                    var95_plot = self._get_metamodel_compare_bar_plot(model, metric='VaR 95 Delta')
                    if var95_plot:
                        parameter_html += "<h3>Compare Bar Plot - VaR 95 Delta</h3>"
                        parameter_html += var95_plot

            parameter_html += "<br>"

        t = t.replace('{PARAMETER_HTML}', parameter_html)
        return t
