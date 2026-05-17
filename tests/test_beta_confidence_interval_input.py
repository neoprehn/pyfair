import math

import pytest
import scipy.stats

from pyfair.model.model import FairModel
from pyfair.utility.fair_exception import FairException


def _build_model():
    model = FairModel('beta_ci_test', n_simulations=5000)

    model.input_data(
        'CF',
        distribution='poisson',
        params={
            'lambda': 1.0,
            'range': 0.25,
        }
    )

    model.input_data(
        'PoA',
        distribution='beta',
        params={'mean': 0.4},
        confidence='high'
    )
    model.input_data(
        'TC',
        distribution='beta',
        params={'mean': 0.5},
        confidence='moderate'
    )
    model.input_data(
        'CS',
        distribution='beta',
        params={'mean': 0.7},
        confidence='moderate'
    )
    model.input_data(
        'PL',
        distribution='lognormal',
        params={'mean': 76750},
        confidence='high'
    )
    model.input_data(
        'SLEF',
        distribution='beta',
        params={'mean': 0.06},
        confidence='moderate'
    )
    model.input_data(
        'SLEM',
        distribution='pert',
        params={
            'low': 100000,
            'mode': 250000,
            'high': 1000000,
        }
    )

    return model


def test_beta_confidence_interval_input_is_accepted():
    model = _build_model()

    model.input_data(
        'PoA',
        distribution='beta',
        input_mode='confidence_interval',
        params={
            'low': 0.15,
            'high': 0.30,
            'confidence': 0.90,
        }
    )

    params = model.export_params()
    poa = params['Probability of Action']

    assert poa['distribution'] == 'beta'
    assert 'mean' in poa['params']
    assert 'k' in poa['params']
    assert poa['params']['k'] > 0
    assert 0 < poa['params']['mean'] < 1


def test_beta_confidence_interval_hits_target_interval_approximately():
    model = _build_model()

    model.input_data(
        'PoA',
        distribution='beta',
        input_mode='confidence_interval',
        params={
            'low': 0.15,
            'high': 0.30,
            'confidence': 0.90,
        }
    )

    params = model.export_params()['Probability of Action']
    mean = params['params']['mean']
    k = params['params']['k']

    alpha = mean * k
    beta = (1.0 - mean) * k

    q_low = scipy.stats.beta.ppf(0.05, alpha, beta)
    q_high = scipy.stats.beta.ppf(0.95, alpha, beta)

    assert math.isclose(q_low, 0.15, rel_tol=0.0, abs_tol=0.03)
    assert math.isclose(q_high, 0.30, rel_tol=0.0, abs_tol=0.03)


def test_confidence_interval_only_supported_for_beta():
    model = _build_model()

    with pytest.raises(FairException, match='only supported for "beta"'):
        model.input_data(
            'PL',
            distribution='lognormal',
            input_mode='confidence_interval',
            params={
                'low': 0.15,
                'high': 0.30,
                'confidence': 0.90,
            }
        )


def test_confidence_interval_rejects_invalid_bounds():
    model = _build_model()

    with pytest.raises(FairException, match='0 <= low < high <= 1'):
        model.input_data(
            'PoA',
            distribution='beta',
            input_mode='confidence_interval',
            params={
                'low': 0.40,
                'high': 0.20,
                'confidence': 0.90,
            }
        )


def test_confidence_interval_rejects_out_of_range_values():
    model = _build_model()

    with pytest.raises(FairException, match='0 <= low < high <= 1'):
        model.input_data(
            'PoA',
            distribution='beta',
            input_mode='confidence_interval',
            params={
                'low': -0.10,
                'high': 0.30,
                'confidence': 0.90,
            }
        )


def test_confidence_interval_rejects_invalid_confidence():
    model = _build_model()

    with pytest.raises(FairException, match='0 < confidence < 1'):
        model.input_data(
            'PoA',
            distribution='beta',
            input_mode='confidence_interval',
            params={
                'low': 0.15,
                'high': 0.30,
                'confidence': 1.0,
            }
        )


def test_confidence_interval_rejects_conflicting_mean_or_k():
    model = _build_model()

    with pytest.raises(FairException, match='may not be combined'):
        model.input_data(
            'PoA',
            distribution='beta',
            input_mode='confidence_interval',
            params={
                'low': 0.15,
                'high': 0.30,
                'confidence': 0.90,
                'mean': 0.22,
            }
        )


def test_confidence_interval_requires_all_fields():
    model = _build_model()

    with pytest.raises(FairException, match='requires "high"'):
        model.input_data(
            'PoA',
            distribution='beta',
            input_mode='confidence_interval',
            params={
                'low': 0.15,
                'confidence': 0.90,
            }
        )
