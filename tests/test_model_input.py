import pytest

from pyfair import FairModel
from pyfair.utility.fair_exception import FairException


def test_legacy_pert_input_still_works():
    """Legacy low/mode/high input should still be accepted."""
    model = FairModel(name="legacy_pert", n_simulations=1000)

    model.input_data('TEF', low=0, mode=1, high=5)
    params = model.export_params()

    assert 'Threat Event Frequency' in params
    assert params['Threat Event Frequency']['distribution'] == 'pert'
    assert params['Threat Event Frequency']['params']['low'] == 0
    assert params['Threat Event Frequency']['params']['mode'] == 1
    assert params['Threat Event Frequency']['params']['high'] == 5
    assert params['Threat Event Frequency']['params']['gamma'] == 4
    assert params['Threat Event Frequency']['confidence'] is None


def test_legacy_normal_input_still_works():
    """Legacy mean/stdev input should still be accepted."""
    model = FairModel(name="legacy_normal", n_simulations=1000)

    model.input_data('PL', mean=100000, stdev=25000)
    params = model.export_params()

    assert params['Primary Loss']['distribution'] == 'normal'
    assert params['Primary Loss']['params']['mean'] == 100000
    assert params['Primary Loss']['params']['stdev'] == 25000
    assert params['Primary Loss']['confidence'] is None


def test_legacy_constant_input_still_works():
    """Legacy constant input should still be accepted."""
    model = FairModel(name="legacy_constant", n_simulations=1000)

    model.input_data('SLEF', constant=0.05)
    params = model.export_params()

    assert params['Secondary Loss Event Frequency']['distribution'] == 'constant'
    assert params['Secondary Loss Event Frequency']['params']['constant'] == 0.05
    assert params['Secondary Loss Event Frequency']['confidence'] is None


def test_structured_beta_uses_default_k_when_no_confidence_and_no_explicit_k():
    """Structured beta input should use default k=15 if nothing else is supplied."""
    model = FairModel(name="beta_default_k", n_simulations=1000)

    model.input_data(
        'TC',
        distribution='beta',
        params={'mean': 0.3}
    )
    params = model.export_params()

    assert params['Threat Capability']['distribution'] == 'beta'
    assert params['Threat Capability']['params']['mean'] == 0.3
    assert params['Threat Capability']['params']['k'] == 15
    assert params['Threat Capability']['confidence'] is None


def test_structured_beta_high_confidence_sets_k_40():
    """Structured beta input with confidence='high' should resolve to k=40."""
    model = FairModel(name="beta_conf_high", n_simulations=1000)

    model.input_data(
        'TC',
        distribution='beta',
        params={'mean': 0.3},
        confidence='high'
    )
    params = model.export_params()

    assert params['Threat Capability']['distribution'] == 'beta'
    assert params['Threat Capability']['params']['mean'] == 0.3
    assert params['Threat Capability']['params']['k'] == 40
    assert params['Threat Capability']['confidence'] == 'high'


def test_structured_lognormal_uses_default_sigma():
    """Structured lognormal input should use default sigma=0.6."""
    model = FairModel(name="lognormal_default_sigma", n_simulations=1000)

    model.input_data(
        'PL',
        distribution='lognormal',
        params={'mean': 100000}
    )
    params = model.export_params()

    assert params['Primary Loss']['distribution'] == 'lognormal'
    assert params['Primary Loss']['params']['mean'] == 100000
    assert params['Primary Loss']['params']['sigma'] == 0.6
    assert params['Primary Loss']['confidence'] is None


def test_structured_poisson_uses_default_range():
    """Structured poisson input should use default range=0.4."""
    model = FairModel(name="poisson_default_range", n_simulations=1000)

    model.input_data(
        'TEF',
        distribution='poisson',
        params={'lambda': 2.0}
    )
    params = model.export_params()

    assert params['Threat Event Frequency']['distribution'] == 'poisson'
    assert params['Threat Event Frequency']['params']['lambda'] == 2.0
    assert params['Threat Event Frequency']['params']['range'] == 0.4
    assert params['Threat Event Frequency']['confidence'] is None


def test_structured_pert_uses_default_gamma():
    """Structured pert input should use default gamma=4."""
    model = FairModel(name="pert_default_gamma", n_simulations=1000)

    model.input_data(
        'SLEM',
        distribution='pert',
        params={'low': 10000, 'mode': 25000, 'high': 100000}
    )
    params = model.export_params()

    assert params['Secondary Loss Event Magnitude']['distribution'] == 'pert'
    assert params['Secondary Loss Event Magnitude']['params']['low'] == 10000
    assert params['Secondary Loss Event Magnitude']['params']['mode'] == 25000
    assert params['Secondary Loss Event Magnitude']['params']['high'] == 100000
    assert params['Secondary Loss Event Magnitude']['params']['gamma'] == 4
    assert params['Secondary Loss Event Magnitude']['confidence'] is None


def test_explicit_shape_parameter_overrides_default():
    """Explicit shape/range parameters should be preserved as supplied."""
    model = FairModel(name="explicit_shape", n_simulations=1000)

    model.input_data(
        'PL',
        distribution='lognormal',
        params={'mean': 100000, 'sigma': 0.9}
    )
    params = model.export_params()

    assert params['Primary Loss']['distribution'] == 'lognormal'
    assert params['Primary Loss']['params']['mean'] == 100000
    assert params['Primary Loss']['params']['sigma'] == 0.9
    assert params['Primary Loss']['confidence'] is None


def test_confidence_and_explicit_shape_parameter_conflict_for_beta():
    """confidence and explicit k must not be used together for beta."""
    model = FairModel(name="beta_conflict", n_simulations=1000)

    with pytest.raises(FairException) as exc:
        model.input_data(
            'TC',
            distribution='beta',
            params={'mean': 0.4, 'k': 20},
            confidence='moderate'
        )

    assert 'confidence' in str(exc.value)
    assert 'k' in str(exc.value)


def test_confidence_and_explicit_shape_parameter_conflict_for_lognormal():
    """confidence and explicit sigma must not be used together for lognormal."""
    model = FairModel(name="lognormal_conflict", n_simulations=1000)

    with pytest.raises(FairException) as exc:
        model.input_data(
            'PL',
            distribution='lognormal',
            params={'mean': 100000, 'sigma': 0.9},
            confidence='moderate'
        )

    assert 'confidence' in str(exc.value)
    assert 'sigma' in str(exc.value)


def test_confidence_and_explicit_shape_parameter_conflict_for_poisson():
    """confidence and explicit range must not be used together for poisson."""
    model = FairModel(name="poisson_conflict", n_simulations=1000)

    with pytest.raises(FairException) as exc:
        model.input_data(
            'TEF',
            distribution='poisson',
            params={'lambda': 2.0, 'range': 0.25},
            confidence='moderate'
        )

    assert 'confidence' in str(exc.value)
    assert 'range' in str(exc.value)


def test_confidence_and_explicit_shape_parameter_conflict_for_pert():
    """confidence and explicit gamma must not be used together for pert."""
    model = FairModel(name="pert_conflict", n_simulations=1000)

    with pytest.raises(FairException) as exc:
        model.input_data(
            'SLEM',
            distribution='pert',
            params={'low': 10000, 'mode': 25000, 'high': 100000, 'gamma': 3},
            confidence='moderate'
        )

    assert 'confidence' in str(exc.value)
    assert 'gamma' in str(exc.value)


def test_alias_cf_and_poa_work():
    """Alias handling should support CF and PoA."""
    model = FairModel(name="alias_test", n_simulations=1000)

    model.input_data(
        'CF',
        distribution='poisson',
        params={'lambda': 2.0}
    )
    model.input_data(
        'PoA',
        distribution='beta',
        params={'mean': 0.3}
    )

    params = model.export_params()

    assert 'Contact Frequency' in params
    assert 'Probability of Action' in params
    assert params['Contact Frequency']['distribution'] == 'poisson'
    assert params['Probability of Action']['distribution'] == 'beta'


def test_model_can_calculate_with_structured_inputs():
    """A minimal structured model should calculate successfully."""
    model = FairModel(name="structured_calc", n_simulations=1000)

    model.input_data('TEF', distribution='poisson', params={'lambda': 1.0})
    model.input_data('TC', distribution='beta', params={'mean': 0.3})
    model.input_data('CS', distribution='beta', params={'mean': 0.4})
    model.input_data('PL', distribution='lognormal', params={'mean': 100000})
    model.input_data('SLEF', distribution='beta', params={'mean': 0.05})
    model.input_data('SLEM', distribution='pert', params={'low': 10000, 'mode': 25000, 'high': 100000})

    model.calculate_all()
    results = model.export_results()

    assert results is not None
    assert 'Risk' in results.columns
    assert len(results) == 1000