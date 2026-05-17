from pyfair import FairModel, FairSimpleReport


def _build_report_model():
    model = FairModel(name="report_model", n_simulations=1000)

    model.input_data('TEF', distribution='poisson', params={'lambda': 1.0})
    model.input_data('TC', distribution='beta', params={'mean': 0.3})
    model.input_data('CS', distribution='beta', params={'mean': 0.4})
    model.input_data('PL', distribution='lognormal', params={'mean': 100000})
    model.input_data('SLEF', distribution='beta', params={'mean': 0.05})
    model.input_data(
        'SLEM',
        distribution='pert',
        params={'low': 10000, 'mode': 25000, 'high': 100000}
    )
    model.calculate_all()
    return model


def test_html_report_contains_loss_frequency_magnitude_scatter(tmp_path):
    model = _build_report_model()
    output_file = tmp_path / "report_output.html"

    fsr = FairSimpleReport([model], currency_prefix='EUR ')
    fsr.to_html(output_file)

    html = output_file.read_text(encoding='utf-8')

    assert 'Loss Frequency vs. Loss Magnitude' in html
    assert 'Primary' in html
    assert 'Secondary' in html
