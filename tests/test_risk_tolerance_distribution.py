from pyfair import FairModel, FairSimpleReport
from pyfair.report.exceedence import FairExceedenceCurves


def _build_model():
    model = FairModel(name="dist_tol_model", n_simulations=3000)

    model.input_data('TEF', distribution='poisson', params={'lambda': 1.2})
    model.input_data('TC', distribution='beta', params={'mean': 0.35}, confidence='high')
    model.input_data('CS', distribution='beta', params={'mean': 0.45})
    model.input_data('PL', distribution='lognormal', params={'mean': 120000})
    model.input_data('SLEF', distribution='beta', params={'mean': 0.08})
    model.input_data('SLEM', distribution='pert', params={'low': 10000, 'mode': 30000, 'high': 120000})

    model.calculate_all()
    return model


def test_distribution_risk_tolerance_renders():
    model = _build_model()

    risk_tolerance = {
        "type": "distribution",
        "distribution": "pert",
        "params": {
            "low": 15000,
            "mode": 50000,
            "high": 180000,
            "gamma": 4.0,
        },
        "samples": 4000,
    }

    fec = FairExceedenceCurves(model, currency_prefix='EUR ', risk_tolerance=risk_tolerance)
    fig, axes = fec.generate_image()

    assert fig is not None
    assert axes is not None


def test_distribution_risk_tolerance_intersections_not_empty():
    model = _build_model()

    risk_tolerance = {
        "type": "distribution",
        "distribution": "lognormal",
        "params": {
            "mean": 70000,
            "sigma": 0.55,
        },
        "samples": 4000,
    }

    fec = FairExceedenceCurves(model, currency_prefix='EUR ', risk_tolerance=risk_tolerance)
    intersections = fec.get_tolerance_intersections()

    assert intersections is not None
    assert not intersections.empty
    assert set(['Model', 'Tolerance Type', 'Loss', 'Exceedance %']).issubset(intersections.columns)
    assert 'distribution' in set(intersections['Tolerance Type'])


def test_html_report_with_distribution_risk_tolerance(tmp_path):
    model = _build_model()

    risk_tolerance = {
        "type": "distribution",
        "distribution": "pert",
        "params": {
            "low": 20000,
            "mode": 60000,
            "high": 200000,
            "gamma": 4.0,
        },
        "samples": 4000,
    }

    output_file = tmp_path / "report_output.html"

    fsr = FairSimpleReport(
        [model],
        currency_prefix='EUR ',
        risk_tolerance=risk_tolerance,
    )
    fsr.to_html(output_file)

    html = output_file.read_text(encoding='utf-8')

    assert output_file.exists()
    assert 'Risk Tolerance Details' in html
    assert 'Tolerance Type' in html
    assert 'distribution' in html
    assert 'Distribution' in html
    assert 'Parameters' in html
    assert 'Samples' in html
    assert 'pert' in html
    assert 'low=' in html
    assert 'mode=' in html
    assert 'high=' in html
