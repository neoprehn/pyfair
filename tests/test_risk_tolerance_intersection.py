from pyfair import FairModel
from pyfair.report.exceedence import FairExceedenceCurves


def _build_model():
    model = FairModel(name="intersection_model", n_simulations=2000)

    model.input_data('TEF', distribution='poisson', params={'lambda': 1.2})
    model.input_data('TC', distribution='beta', params={'mean': 0.35}, confidence='high')
    model.input_data('CS', distribution='beta', params={'mean': 0.45})
    model.input_data('PL', distribution='lognormal', params={'mean': 75000})
    model.input_data('SLEF', distribution='beta', params={'mean': 0.08})
    model.input_data('SLEM', distribution='pert', params={'low': 15000, 'mode': 40000, 'high': 150000})

    model.calculate_all()
    return model


def test_constant_risk_tolerance_intersection_returns_one_point():
    model = _build_model()
    fec = FairExceedenceCurves(
        [model],
        currency_prefix='EUR ',
        risk_tolerance={
            'type': 'constant',
            'value': 50000,
        }
    )

    intersections = fec.get_tolerance_intersections()

    assert len(intersections) == 1
    assert intersections.iloc[0]['Tolerance Type'] == 'constant'
    assert intersections.iloc[0]['Loss'] == 50000
    assert 0 <= intersections.iloc[0]['Exceedance %'] <= 100


def test_curve_risk_tolerance_intersection_returns_at_least_one_point():
    model = _build_model()
    fec = FairExceedenceCurves(
        [model],
        currency_prefix='EUR ',
        risk_tolerance={
            'type': 'curve',
            'points': [
                {'loss': 1000, 'level': 1.00},
                {'loss': 10000, 'level': 0.70},
                {'loss': 50000, 'level': 0.30},
                {'loss': 150000, 'level': 0.05},
            ]
        }
    )

    intersections = fec.get_tolerance_intersections()

    assert not intersections.empty
    assert set(intersections['Tolerance Type']) == {'curve'}
    assert intersections['Intersection #'].min() == 1
    assert intersections['Loss'].min() > 0
    assert intersections['Exceedance %'].between(0, 100).all()


def test_generate_image_with_intersections_renders_successfully():
    model = _build_model()
    fec = FairExceedenceCurves(
        [model],
        currency_prefix='EUR ',
        risk_tolerance={
            'type': 'curve',
            'points': [
                {'loss': 1000, 'level': 'P0'},
                {'loss': 30000, 'level': 'P50'},
                {'loss': 250000, 'level': 'P95'},
            ]
        }
    )

    fig, axes = fec.generate_image()

    assert fig is not None
    assert axes is not None
