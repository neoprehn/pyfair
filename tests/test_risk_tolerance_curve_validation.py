import pytest

from pyfair import FairModel
from pyfair.report.exceedence import FairExceedenceCurves
from pyfair.utility.fair_exception import FairException


def _build_model():
    model = FairModel(name="curve_validation_model", n_simulations=2000)
    model.input_data('CF', distribution='poisson', params={'lambda': 2.0, 'range': 0.25})
    model.input_data('PoA', distribution='beta', params={'mean': 0.35, 'k': 20})
    model.input_data('TC', distribution='beta', params={'mean': 0.50, 'k': 20})
    model.input_data('CS', distribution='beta', params={'mean': 0.60, 'k': 20})
    model.input_data('PL', distribution='lognormal', params={'mean': 40000, 'sigma': 0.5})
    model.input_data('SLEF', distribution='beta', params={'mean': 0.05, 'k': 20})
    model.input_data('SLEM', distribution='pert', params={'low': 15000, 'mode': 50000, 'high': 150000, 'gamma': 4.0})
    model.calculate_all()
    return model


def _build_fec(risk_tolerance):
    return FairExceedenceCurves(_build_model(), currency_prefix='EUR ', risk_tolerance=risk_tolerance)


def test_curve_requires_at_least_two_points():
    fec = _build_fec({
        'type': 'curve',
        'points': [{'loss': 10000, 'level': 'P50'}],
    })

    with pytest.raises(FairException, match='at least two points'):
        fec.get_tolerance_intersections()


def test_curve_rejects_duplicate_loss_values():
    fec = _build_fec({
        'type': 'curve',
        'points': [
            {'loss': 10000, 'level': 'P50'},
            {'loss': 10000, 'level': 'P20'},
        ],
    })

    with pytest.raises(FairException, match='unique'):
        fec.get_tolerance_intersections()


def test_curve_requires_positive_loss_values():
    fec = _build_fec({
        'type': 'curve',
        'points': [
            {'loss': -1000, 'level': 'P80'},
            {'loss': 10000, 'level': 'P50'},
        ],
    })

    with pytest.raises(FairException, match='positive'):
        fec.get_tolerance_intersections()


def test_curve_rejects_invalid_level():
    fec = _build_fec({
        'type': 'curve',
        'points': [
            {'loss': 1000, 'level': 'bad-level'},
            {'loss': 10000, 'level': 'P50'},
        ],
    })

    with pytest.raises(Exception):
        fec.get_tolerance_intersections()


def test_curve_requires_monotonic_tolerance():
    fec = _build_fec({
        'type': 'curve',
        'points': [
            {'loss': 1000, 'level': 'P80'},
            {'loss': 10000, 'level': 'P50'},
            {'loss': 100000, 'level': 'P90'},
        ],
    })

    with pytest.raises(FairException, match='monotonic|non-increasing'):
        fec.get_tolerance_intersections()


def test_curve_accepts_valid_monotonic_points():
    fec = _build_fec({
        'type': 'curve',
        'points': [
            {'loss': 1000, 'level': 'P10'},
            {'loss': 10000, 'level': 'P50'},
            {'loss': 100000, 'level': 'P95'},
        ],
    })

    intersections = fec.get_tolerance_intersections()
    assert intersections is not None
