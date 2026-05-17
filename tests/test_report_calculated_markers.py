from pyfair import FairModel, FairSimpleReport


def _build_report_model():
    model = FairModel(name="calculated_marker_model", n_simulations=1000)

    model.input_data(
        'CF',
        distribution='poisson',
        params={'lambda': 4.0}
    )

    model.input_data(
        'PoA',
        distribution='beta',
        params={'mean': 0.25}
    )

    model.input_data(
        'TC',
        distribution='beta',
        params={'mean': 0.30}
    )

    model.input_data(
        'CS',
        distribution='beta',
        params={'mean': 0.40}
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


def test_html_report_marks_calculated_nodes_in_parameter_table(tmp_path):
    model = _build_report_model()

    output_file = tmp_path / 'report_output.html'

    fsr = FairSimpleReport([model], currency_prefix='EUR ')
    fsr.to_html(output_file)

    html = output_file.read_text(encoding='utf-8')

    assert 'Risk (calculated)' in html
    assert 'Threat Event Frequency (calculated)' in html
    assert 'Vulnerability (calculated)' in html
    assert 'Loss Event Frequency (calculated)' in html

    # Supplied inputs should remain unmarked.
    assert 'Contact Frequency (calculated)' not in html
    assert 'Probability of Action (calculated)' not in html
    assert 'Threat Capability (calculated)' not in html
    assert 'Control Strength (calculated)' not in html
