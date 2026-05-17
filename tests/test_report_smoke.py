from pathlib import Path

from pyfair import FairModel, FairSimpleReport
from pyfair.report.tree_graph import FairTreeGraph


def _build_report_model():
    """Create a model suitable for report and tree smoke tests."""
    model = FairModel(name="report_model", n_simulations=1000)

    model.input_data(
        'TEF',
        distribution='poisson',
        params={'lambda': 1.0}
    )

    model.input_data(
        'TC',
        distribution='beta',
        params={'mean': 0.3},
        confidence='high'
    )

    model.input_data(
        'CS',
        distribution='beta',
        params={'mean': 0.4}
    )

    model.input_data(
        'PL',
        distribution='lognormal',
        params={'mean': 100000}
    )

    model.input_data(
        'SLEF',
        distribution='beta',
        params={'mean': 0.05}
    )

    model.input_data(
        'SLEM',
        distribution='pert',
        params={'low': 10000, 'mode': 25000, 'high': 100000}
    )

    model.calculate_all()
    return model


def test_tree_graph_renders_without_error():
    """Tree graph should render successfully for structured inputs."""
    model = _build_report_model()

    format_strings = {
        'Risk': 'EUR {0:,.4f}',
        'Loss Event Frequency': '{0:.4f}',
        'Threat Event Frequency': '{0:.4f}',
        'Vulnerability': '{0:.4f}',
        'Contact Frequency': '{0:.4f}',
        'Probability of Action': '{0:.4f}',
        'Threat Capability': '{0:.4f}',
        'Control Strength': '{0:.4f}',
        'Loss Magnitude': 'EUR {0:,.4f}',
        'Primary Loss': 'EUR {0:,.4f}',
        'Secondary Loss': 'EUR {0:,.4f}',
        'Secondary Loss Event Frequency': '{0:.4f}',
        'Secondary Loss Event Magnitude': 'EUR {0:,.4f}',
    }

    ftg = FairTreeGraph(model, format_strings)
    fig, ax = ftg.generate_image()

    assert fig is not None
    assert ax is not None


def test_html_report_is_created(tmp_path):
    """HTML report should be written successfully."""
    model = _build_report_model()

    output_file = tmp_path / "report_output.html"

    fsr = FairSimpleReport([model], currency_prefix='EUR ')
    fsr.to_html(str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_html_report_contains_confidence_and_parameter_markers(tmp_path):
    """HTML report should contain key structured parameter markers.

    This test is intentionally robust against formatting changes.
    It checks for the presence of confidence and parameter names,
    but does not depend on exact numeric formatting or layout.
    """
    model = _build_report_model()

    output_file = tmp_path / "report_output.html"

    fsr = FairSimpleReport([model], currency_prefix='EUR ')
    fsr.to_html(output_file)

    html = output_file.read_text(encoding='utf-8')

    # Confidence should appear in the report.
    assert 'high' in html

    # Structured parameter markers should appear.
    assert 'mean=' in html
    assert 'k=' in html


def test_html_report_contains_key_elements(tmp_path):
    """HTML report should contain basic report structure and key FAIR labels.

    This test intentionally checks only stable report markers and avoids
    brittle assumptions about exact tree or template wording.
    """
    model = _build_report_model()

    output_file = tmp_path / "report_output.html"

    fsr = FairSimpleReport([model], currency_prefix='EUR ')
    fsr.to_html(output_file)

    html = output_file.read_text(encoding='utf-8')

    # Basic HTML/report structure
    assert '<html' in html.lower()
    assert '<table' in html.lower()

    # Stable FAIR content markers
    assert 'Threat Capability' in html
    assert 'Control Strength' in html
    assert 'Primary Loss' in html
    
def test_html_report_can_optionally_export_csv(tmp_path):
    """HTML export should optionally write CSV files."""
    model = _build_report_model()

    output_file = tmp_path / "report_output.html"

    fsr = FairSimpleReport([model], currency_prefix='EUR ')
    fsr.to_html(str(output_file), export_csv=True)

    expected_csv = tmp_path / "report_model.csv"

    assert output_file.exists()
    assert expected_csv.exists()
    assert expected_csv.stat().st_size > 0
    
def test_html_report_with_constant_risk_tolerance(tmp_path):
    """HTML report should render with a constant risk tolerance line."""
    model = _build_report_model()

    output_file = tmp_path / "report_output.html"

    risk_tolerance = {
        "type": "constant",
        "value": 50000
    }

    fsr = FairSimpleReport(
        [model],
        currency_prefix='EUR ',
        risk_tolerance=risk_tolerance
    )
    fsr.to_html(str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_html_report_with_curve_risk_tolerance(tmp_path):
    """HTML report should render with a curve-based risk tolerance line."""
    model = _build_report_model()

    output_file = tmp_path / "report_output.html"

    risk_tolerance = {
        "type": "curve",
        "points": [
            {"loss": 20000, "level": "P0"},
            {"loss": 50000, "level": "P50"},
            {"loss": 200000, "level": "P90"},
        ]
    }

    fsr = FairSimpleReport(
        [model],
        currency_prefix='EUR ',
        risk_tolerance=risk_tolerance
    )
    fsr.to_html(str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0    

def test_html_report_contains_risk_tolerance_intersection_block(tmp_path):
    """HTML report should contain the tolerance intersection summary."""
    model = _build_report_model()

    output_file = tmp_path / "report_output.html"

    risk_tolerance = {
        "type": "curve",
        "points": [
            {"loss": 1000, "level": "P0"},
            {"loss": 50000, "level": "P50"},
            {"loss": 250000, "level": "P95"},
        ]
    }

    fsr = FairSimpleReport(
        [model],
        currency_prefix='EUR ',
        risk_tolerance=risk_tolerance
    )
    fsr.to_html(str(output_file))

    html = output_file.read_text(encoding='utf-8')

    assert 'Risk Tolerance Intersection' in html
    assert 'Tolerance Type' in html
    assert 'Exceedance %' in html
