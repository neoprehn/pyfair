from pyfair import FairModel, FairMetaModel, FairSimpleReport


def _build_compare_models(n_simulations=1500):
    baseline = FairModel(name="Baseline", n_simulations=n_simulations)
    baseline.input_data('TEF', distribution='poisson', params={'lambda': 1.0})
    baseline.input_data('TC', distribution='beta', params={'mean': 0.30})
    baseline.input_data('CS', distribution='beta', params={'mean': 0.35})
    baseline.input_data('PL', distribution='lognormal', params={'mean': 75000})
    baseline.input_data('SLEF', distribution='beta', params={'mean': 0.05})
    baseline.input_data('SLEM', distribution='pert', params={'low': 10000, 'mode': 25000, 'high': 120000})
    baseline.calculate_all()

    variant = FairModel(name="Laptop", n_simulations=n_simulations)
    variant.input_data('CF', distribution='poisson', params={'lambda': 2.0, 'range': 0.25})
    variant.input_data(
        'PoA',
        distribution='beta',
        input_mode='confidence_interval',
        params={'low': 0.15, 'high': 0.60, 'confidence': 0.90},
    )
    variant.input_data('TC', distribution='beta', params={'mean': 0.50, 'k': 25})
    variant.input_data('CS', distribution='beta', params={'mean': 0.70, 'k': 25})
    variant.input_data('PL', distribution='lognormal', params={'mean': 76750}, confidence='high')
    variant.input_data('SLEF', distribution='beta', params={'mean': 0.06, 'k': 30})
    variant.input_data(
        'SLEM',
        distribution='pert',
        params={'low': 100000, 'mode': 250000, 'high': 1800000, 'gamma': 3},
    )
    variant.calculate_all()

    mm = FairMetaModel(
        name='Vergleich neue Inputs',
        models=[baseline, variant],
        mode='compare',
        baseline_model=baseline.get_name(),
    )
    mm.calculate_all()
    return baseline, variant, mm


def test_compare_metamodel_report_is_created(tmp_path):
    baseline, variant, mm = _build_compare_models()
    output_file = tmp_path / 'compare_report.html'

    fsr = FairSimpleReport([baseline, variant, mm], currency_prefix='EUR ')
    fsr.to_html(output_file)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_compare_metamodel_report_contains_delta_markers(tmp_path):
    baseline, variant, mm = _build_compare_models()
    output_file = tmp_path / 'compare_report.html'

    fsr = FairSimpleReport([baseline, variant, mm], currency_prefix='EUR ')
    fsr.to_html(output_file)

    html = output_file.read_text(encoding='utf-8')

    assert 'Meta Risk Comparison' in html
    assert 'Delta::Laptop' in html or 'Laptop' in html
    assert 'Mean Delta' in html
    assert 'VaR 95 Delta' in html


def test_compare_metamodel_report_can_render_with_risk_tolerance(tmp_path):
    baseline, variant, mm = _build_compare_models()
    output_file = tmp_path / 'compare_report_with_tolerance.html'

    risk_tolerance = {
        'type': 'distribution',
        'distribution': 'lognormal',
        'params': {
            'mean': 10000,
            'sigma': 0.5,
        },
        'samples': 4000,
    }

    fsr = FairSimpleReport(
        [baseline, variant, mm],
        currency_prefix='EUR ',
        risk_tolerance=risk_tolerance,
    )
    fsr.to_html(output_file)

    html = output_file.read_text(encoding='utf-8')

    assert output_file.exists()
    assert 'Risk Tolerance Details' in html
    assert 'Meta Risk Comparison' in html


def test_compare_metamodel_has_delta_columns_and_no_aggregated_risk():
    _, _, mm = _build_compare_models()
    results = mm.export_results()

    assert 'Baseline' in results.columns
    assert 'Laptop' in results.columns
    assert 'Delta::Laptop' in results.columns
    assert 'Risk' not in results.columns
    assert mm.calculation_completed() is True
