import json

from pyfair import FairModel


def _build_structured_model():
    """Create a reusable structured FAIR model for roundtrip tests."""
    model = FairModel(name="roundtrip_model", n_simulations=1000)

    model.input_data(
        'TEF',
        distribution='poisson',
        params={'lambda': 1.0}
    )

    model.input_data(
        'TC',
        distribution='beta',
        params={'mean': 0.3},
        confidence='high'
    )

    model.input_data(
        'CS',
        distribution='beta',
        params={'mean': 0.4}
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

    return model


def test_to_json_returns_valid_json_string():
    """to_json should return a valid JSON string."""
    model = _build_structured_model()
    json_data = model.to_json()

    parsed = json.loads(json_data)

    assert isinstance(json_data, str)
    assert isinstance(parsed, dict)
    assert parsed['name'] == 'roundtrip_model'


def test_structured_input_survives_json_roundtrip():
    """Structured inputs should survive to_json/read_json without error."""
    model = _build_structured_model()

    json_data = model.to_json()
    restored = FairModel.read_json(json_data)

    params = restored.export_params()

    assert 'Threat Capability' in params
    assert params['Threat Capability']['distribution'] == 'beta'
    assert params['Threat Capability']['params']['mean'] == 0.3
    assert params['Threat Capability']['params']['k'] == 40
    assert params['Threat Capability']['confidence'] == 'high'


def test_roundtrip_model_can_calculate():
    """A restored model should still calculate successfully."""
    model = _build_structured_model()

    json_data = model.to_json()
    restored = FairModel.read_json(json_data)

    restored.calculate_all()
    results = restored.export_results()

    assert 'Risk' in results.columns
    assert len(results) == 1000


def test_roundtrip_preserves_defaults_for_non_confidence_inputs():
    """Defaults derived without confidence should still be present after reload."""
    model = _build_structured_model()

    json_data = model.to_json()
    restored = FairModel.read_json(json_data)
    params = restored.export_params()

    assert params['Control Strength']['distribution'] == 'beta'
    assert params['Control Strength']['params']['mean'] == 0.4
    assert params['Control Strength']['params']['k'] == 15
    assert params['Control Strength']['confidence'] is None

    assert params['Primary Loss']['distribution'] == 'lognormal'
    assert params['Primary Loss']['params']['sigma'] == 0.6
    assert params['Primary Loss']['confidence'] is None

    assert params['Threat Event Frequency']['distribution'] == 'poisson'
    assert params['Threat Event Frequency']['params']['range'] == 0.4
    assert params['Threat Event Frequency']['confidence'] is None


def test_bulk_import_accepts_structured_inputs():
    """bulk_import_data should support the new structured schema."""
    model = FairModel(name="bulk_structured", n_simulations=1000)

    model.bulk_import_data({
        'Threat Event Frequency': {
            'distribution': 'poisson',
            'params': {'lambda': 1.5}
        },
        'Threat Capability': {
            'distribution': 'beta',
            'params': {'mean': 0.35},
            'confidence': 'moderate'
        },
        'Control Strength': {
            'distribution': 'beta',
            'params': {'mean': 0.45}
        },
        'Primary Loss': {
            'distribution': 'lognormal',
            'params': {'mean': 120000}
        },
        'Secondary Loss Event Frequency': {
            'distribution': 'beta',
            'params': {'mean': 0.08}
        },
        'Secondary Loss Event Magnitude': {
            'distribution': 'pert',
            'params': {'low': 15000, 'mode': 30000, 'high': 125000}
        }
    })

    model.calculate_all()
    results = model.export_results()
    params = model.export_params()

    assert 'Risk' in results.columns
    assert len(results) == 1000
    assert params['Threat Capability']['params']['k'] == 15
    assert params['Threat Capability']['confidence'] == 'moderate'


def test_bulk_import_accepts_legacy_inputs():
    """bulk_import_data should still support legacy flat input dictionaries."""
    model = FairModel(name="bulk_legacy", n_simulations=1000)

    model.bulk_import_data({
        'TEF': {'low': 0, 'mode': 1, 'high': 5},
        'TC': {'low': 0.2, 'mode': 0.3, 'high': 0.5},
        'CS': {'low': 0.2, 'mode': 0.3, 'high': 0.5},
        'PL': {'low': 3700, 'mode': 76750, 'high': 252500},
        'SLEF': {'low': 0.05, 'mode': 0.06, 'high': 0.1},
        'SLEM': {'low': 100000, 'mode': 250000, 'high': 1800000},
    })

    model.calculate_all()
    results = model.export_results()

    assert 'Risk' in results.columns
    assert len(results) == 1000