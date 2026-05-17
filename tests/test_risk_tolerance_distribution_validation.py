import pytest

from pyfair import FairModel
from pyfair.report.exceedence import FairExceedenceCurves
from pyfair.utility.fair_exception import FairException


def _build_model():
    model = FairModel(name="dist_tol_validation_model", n_simulations=2000)

    model.input_data('TEF', distribution='poisson', params={'lambda': 1.1})
    model.input_data('TC', distribution='beta', params={'mean': 0.35}, confidence='high')
    model.input_data('CS', distribution='beta', params={'mean': 0.45})
    model.input_data('PL', distribution='lognormal', params={'mean': 90000})
    model.input_data('SLEF', distribution='beta', params={'mean': 0.08})
    model.input_data('SLEM', distribution='pert', params={'low': 10000, 'mode': 25000, 'high': 90000})

    model.calculate_all()
    return model


def _build_fec(risk_tolerance):
    model = _build_model()
    return FairExceedenceCurves(model, currency_prefix='EUR ', risk_tolerance=risk_tolerance)


def test_distribution_requires_distribution_name():
    fec = _build_fec({
        'type': 'distribution',
        'params': {'low': 10000, 'mode': 30000, 'high': 90000},
        'samples': 2000,
    })

    with pytest.raises(FairException, match='distribution'):
        fec.get_tolerance_intersections()



def test_distribution_requires_non_empty_params():
    fec = _build_fec({
        'type': 'distribution',
        'distribution': 'pert',
        'params': {},
        'samples': 2000,
    })

    with pytest.raises(FairException, match='params'):
        fec.get_tolerance_intersections()



def test_distribution_rejects_unknown_distribution():
    fec = _build_fec({
        'type': 'distribution',
        'distribution': 'gamma',
        'params': {'mean': 10000},
        'samples': 2000,
    })

    with pytest.raises(FairException, match='Unsupported risk tolerance distribution'):
        fec.get_tolerance_intersections()



def test_distribution_rejects_non_positive_samples():
    fec = _build_fec({
        'type': 'distribution',
        'distribution': 'pert',
        'params': {'low': 10000, 'mode': 30000, 'high': 90000},
        'samples': 0,
    })

    with pytest.raises(FairException, match='samples'):
        fec.get_tolerance_intersections()



def test_pert_distribution_requires_ordered_positive_values():
    fec = _build_fec({
        'type': 'distribution',
        'distribution': 'pert',
        'params': {'low': 50000, 'mode': 40000, 'high': 90000},
        'samples': 2000,
    })

    with pytest.raises(FairException):
        fec.get_tolerance_intersections()



def test_lognormal_distribution_requires_positive_mean():
    fec = _build_fec({
        'type': 'distribution',
        'distribution': 'lognormal',
        'params': {'mean': 0, 'sigma': 0.5},
        'samples': 2000,
    })

    with pytest.raises(FairException, match='mean'):
        fec.get_tolerance_intersections()



def test_lognormal_distribution_requires_positive_sigma():
    fec = _build_fec({
        'type': 'distribution',
        'distribution': 'lognormal',
        'params': {'mean': 50000, 'sigma': 0},
        'samples': 2000,
    })

    with pytest.raises(FairException, match='sigma'):
        fec.get_tolerance_intersections()
