from pyfair import FairModel
from pyfair.report.distribution import FairDistributionCurve



def _build_distribution_model(name='distribution_model'):
    model = FairModel(name=name, n_simulations=1500)

    model.input_data(
        'TEF',
        distribution='poisson',
        params={'lambda': 1.2}
    )

    model.input_data(
        'TC',
        distribution='beta',
        params={'mean': 0.35},
        confidence='high'
    )

    model.input_data(
        'CS',
        distribution='beta',
        params={'mean': 0.45}
    )

    model.input_data(
        'PL',
        distribution='lognormal',
        params={'mean': 120000, 'sigma': 0.9}
    )

    model.input_data(
        'SLEF',
        distribution='beta',
        params={'mean': 0.08}
    )

    model.input_data(
        'SLEM',
        distribution='pert',
        params={'low': 15000, 'mode': 35000, 'high': 180000}
    )

    model.calculate_all()
    return model



def test_distribution_curve_renders_with_histogram_and_density():
    model = _build_distribution_model()

    curve = FairDistributionCurve(model, currency_prefix='EUR ')
    fig, ax = curve.generate_image()

    assert fig is not None
    assert ax is not None
    assert ax.get_ylabel() == 'Density'
    assert len(ax.lines) >= 1



def test_distribution_curve_can_render_density_only():
    model = _build_distribution_model()

    curve = FairDistributionCurve(
        model,
        currency_prefix='EUR ',
        show_histogram=False,
        show_density=True,
    )
    fig, ax = curve.generate_image()

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) >= 1



def test_distribution_curve_can_render_multiple_models():
    model_a = _build_distribution_model('distribution_model_a')
    model_b = _build_distribution_model('distribution_model_b')

    curve = FairDistributionCurve([model_a, model_b], currency_prefix='EUR ')
    fig, ax = curve.generate_image()

    legend = ax.get_legend()

    assert fig is not None
    assert ax is not None
    assert legend is not None
    assert len(legend.texts) == 2
