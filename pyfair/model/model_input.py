"""This module contains an input object for sanitizing / checking data."""

import scipy.stats
import scipy.optimize
import copy

import numpy as np
import pandas as pd

from ..utility.fair_exception import FairException
from ..utility.beta_pert import FairBetaPert
from ..utility.confidence_mapping import (
    get_confidence_defaults,
    get_default_for_confidence,
    get_default_for_distribution,
)

class FairDataInput(object):
    """A captive class for checking and routing data inputs.

    This class now supports two input styles:

    1. Legacy keyword-based input:
       - constant=...
       - mean=..., stdev=...
       - low=..., mode=..., high=..., gamma=...

    2. Structured distribution input:
       - distribution='pert'|'normal'|'lognormal'|'beta'|'poisson'|'constant'
       - params={...}
       - confidence='very_low'|'low'|'moderate'|'high'|'very_high'

    Notes
    -----
    The structured API is additive and backward compatible. Existing
    legacy models should continue to work unchanged.
    """

    def __init__(self):
        # These targets must be less than or equal to one.
        self._le_1_targets = [
            'Probability of Action',
            'Vulnerability',
            'Control Strength',
            'Threat Capability'
        ]

        # These keywords must stay in [0, 1] for le_1 targets.
        self._le_1_keywords = [
            'constant', 'high', 'mode', 'low', 'mean',
            'alpha', 'beta', 'lambda'
        ]

        # Legacy keyword-to-generator mapping.
        self._parameter_map = {
            'constant': self._gen_constant,
            'high': self._gen_pert,
            'mode': self._gen_pert,
            'low': self._gen_pert,
            'gamma': self._gen_pert,
            'mean': self._gen_normal,
            'stdev': self._gen_normal,
        }

        # Legacy required keywords by generator function.
        self._required_keywords = {
            self._gen_constant: ['constant'],
            self._gen_pert: ['low', 'mode', 'high'],
            self._gen_normal: ['mean', 'stdev'],
        }
        
        # Distribution-to-generator mapping for structured inputs.
        self._generator_map = {
            'constant': self._gen_constant,
            'normal': self._gen_normal,
            'pert': self._gen_pert,
            'lognormal': self._gen_lognormal,
            'poisson': self._gen_poisson,
            'beta': self._gen_beta,
        }
            
        # Storage of resolved inputs for reporting/model inspection.
        self._supplied_values = {}

        # Storage of original inputs for clean JSON serialization.
        self._original_supplied_values = {}

    def get_supplied_values(self):
        """Return the dictionary of supplied model inputs."""
        return copy.deepcopy(self._supplied_values)
        
    def get_original_supplied_values(self):
        """Return the originally supplied model inputs."""
        return copy.deepcopy(self._original_supplied_values)

    def _is_structured_input(self, **kwargs):
        """Return True if the input uses the new structured API."""
        return (
            'distribution' in kwargs
            or 'params' in kwargs
            or 'confidence' in kwargs
            or 'input_mode' in kwargs
        )

    def _normalize_input(self, **kwargs):
        """Normalize legacy and structured inputs into one internal schema.

        Returns
        -------
        dict
            Normalized dictionary with shape:
            {
                'distribution': str,
                'params': dict,
                'confidence': str | None
            }
        """
        # Structured API path.
        if self._is_structured_input(**kwargs):
            distribution = kwargs.get('distribution')
            params = kwargs.get('params', {})
            confidence = kwargs.get('confidence', None)
            input_mode = kwargs.get('input_mode', None)

            if distribution is None:
                raise FairException('"distribution" is required when using structured input.')

            if type(params) is not dict:
                raise FairException('"params" must be a dictionary.')

            return {
                'distribution': distribution.lower(),
                'params': dict(params),
                'confidence': confidence,
                'input_mode': input_mode
            }

        # Legacy API path.
        func = self._determine_func(**kwargs)

        if func == self._gen_constant:
            return {
                'distribution': 'constant',
                'params': {'constant': kwargs['constant']},
                'confidence': None
            }

        if func == self._gen_normal:
            return {
                'distribution': 'normal',
                'params': {'mean': kwargs['mean'], 'stdev': kwargs['stdev']},
                'confidence': None
            }

        if func == self._gen_pert:
            params = {
                'low': kwargs['low'],
                'mode': kwargs['mode'],
                'high': kwargs['high']
            }
            if 'gamma' in kwargs:
                params['gamma'] = kwargs['gamma']

            return {
                'distribution': 'pert',
                'params': params,
                'confidence': None
            }

        raise FairException('Unable to normalize input.')

    def _resolve_special_input_modes(self, target, normalized):
        """Resolve special structured input modes into standard internal params."""
        distribution = normalized['distribution']
        input_mode = normalized.get('input_mode')
        params = dict(normalized['params'])
        confidence = normalized.get('confidence')

        if input_mode is None:
            return normalized

        if distribution != 'beta':
            raise FairException(
                f'"input_mode"="{input_mode}" is currently only supported for "beta".'
            )

        if input_mode == 'confidence_interval':
            if confidence is not None:
                raise FairException(
                    '"confidence" may not be supplied at the top level when using '
                    '"input_mode"="confidence_interval". '
                    'Put the interval confidence inside params["confidence"].'
                )

            resolved_params = self._resolve_beta_confidence_interval(target, params)

            return {
                'distribution': distribution,
                'params': resolved_params,
                'confidence': None,
                'input_mode': input_mode
            }

        raise FairException(f'Unsupported input_mode "{input_mode}".')
        
    def _resolve_beta_confidence_interval(self, target, params):
        """Resolve beta confidence-interval input into mean/k."""
        required = ['low', 'high', 'confidence']
        for key in required:
            if key not in params:
                raise FairException(
                    f'"beta" with "input_mode"="confidence_interval" requires "{key}".'
                )

        forbidden = ['mean', 'k', 'alpha', 'beta']
        conflict_keys = [key for key in forbidden if key in params]
        if conflict_keys:
            raise FairException(
                '"beta" confidence_interval input may not be combined with '
                f'{conflict_keys}.'
            )

        low = float(params['low'])
        high = float(params['high'])
        confidence = float(params['confidence'])

        if not (0.0 <= low < high <= 1.0):
            raise FairException(
                '"beta" confidence_interval requires 0 <= low < high <= 1.'
            )

        if not (0.0 < confidence < 1.0):
            raise FairException(
                '"beta" confidence_interval requires 0 < confidence < 1.'
            )

        mean, k = self._fit_beta_mean_k_from_confidence_interval(
            low=low,
            high=high,
            confidence=confidence
        )

        return {
            'mean': mean,
            'k': k
        }

    def _fit_beta_mean_k_from_confidence_interval(self, low, high, confidence):
        """Fit beta mean/k from a central confidence interval on [0, 1]."""
        p_low = (1.0 - confidence) / 2.0
        p_high = 1.0 - p_low

        # Rough initial guess
        mean0 = (low + high) / 2.0

        # Approximate interval width via a normal proxy for initialization only
        z = scipy.stats.norm.ppf(p_high)
        sigma0 = max((high - low) / (2.0 * z), 1e-4)

        variance0 = sigma0 ** 2
        max_variance = mean0 * (1.0 - mean0)

        if variance0 >= max_variance:
            variance0 = max_variance * 0.8

        k0 = max((mean0 * (1.0 - mean0) / variance0) - 1.0, 2.0)

        alpha0 = max(mean0 * k0, 1e-3)
        beta0 = max((1.0 - mean0) * k0, 1e-3)

        def objective(log_params):
            alpha = np.exp(log_params[0])
            beta = np.exp(log_params[1])

            err_low = scipy.stats.beta.cdf(low, alpha, beta) - p_low
            err_high = scipy.stats.beta.cdf(high, alpha, beta) - p_high

            return (err_low ** 2) + (err_high ** 2)

        result = scipy.optimize.minimize(
            objective,
            x0=np.log([alpha0, beta0]),
            method='L-BFGS-B'
        )

        if not result.success:
            raise FairException(
                'Unable to fit beta distribution from confidence interval.'
            )

        alpha = float(np.exp(result.x[0]))
        beta = float(np.exp(result.x[1]))

        mean = alpha / (alpha + beta)
        k = alpha + beta

        if not np.isfinite(mean) or not np.isfinite(k) or k <= 0:
            raise FairException(
                'Invalid beta parameters were produced from confidence interval.'
            )

        return mean, k

    def _apply_defaults(self, normalized):
        """Apply confidence-based or plain default shape parameters.

        Rules
        -----
        1. If confidence is given, the matching shape/range parameter
           must not already be explicitly supplied.
        2. If confidence is not given and shape/range parameter is absent,
           use the default value.
        3. If the shape/range parameter is explicitly supplied and
           confidence is absent, keep the explicit value.
        """
        distribution = normalized['distribution']
        params = dict(normalized['params'])
        confidence = normalized['confidence']

        explicit_param_by_distribution = {
            'beta': 'k',
            'lognormal': 'sigma',
            'poisson': 'range',
            'pert': 'gamma',
        }

        explicit_param = explicit_param_by_distribution.get(distribution)

        # Nothing to apply for distributions without a shape override.
        if explicit_param is None:
            return {
                'distribution': distribution,
                'params': params,
                'confidence': confidence
            }

        # If confidence is present, explicit shape/range parameter is not allowed.
        if confidence is not None and explicit_param in params:
            err = (
                f'Input for distribution "{distribution}" may not contain both '
                f'"confidence" and explicit "{explicit_param}".'
            )
            raise FairException(err)

        # Apply confidence-derived value.
        if confidence is not None:
            # Apply confidence-derived shape/range defaults.
            params.update(get_default_for_confidence(confidence, distribution))

        # Otherwise apply ordinary default if missing.
        elif explicit_param not in params:
            try:
                params.update(get_default_for_distribution(distribution))
            except FairException:
                # Distributions without a default shape/range parameter
                # simply pass through unchanged.
                pass

        return {
            'distribution': distribution,
            'params': params,
            'confidence': confidence
        }

    def _check_le_1(self, target, **params):
        """Raise an error if bounded FAIR factors exceed [0, 1]."""
        for key, value in params.items():
            applicable_keyword = key in self._le_1_keywords
            applicable_target = target in self._le_1_targets
            if applicable_keyword and applicable_target:
                if 0.0 <= value <= 1.0:
                    pass
                else:
                    raise FairException(
                        '"{}" must have "{}" value between zero and one.'.format(target, key)
                    )

    def _check_non_negative(self, **params):
        """Ensure relevant numeric parameters are not negative."""
        non_negative_keys = [
            'mean', 'constant', 'low', 'mode', 'high',
            'stdev', 'gamma', 'sigma', 'k', 'lambda',
            'alpha', 'beta', 'range'
        ]

        for keyword, value in params.items():
            if keyword in non_negative_keys and value < 0:
                raise FairException('"{}" is less than zero.'.format(keyword))

    def _check_pert(self, **params):
        """Ensure PERT parameters form a valid ordered triplet."""
        conditions = {
            'mode >= low': params['mode'] >= params['low'],
            'high >= mode': params['high'] >= params['mode'],
        }

        for condition_name, condition_value in conditions.items():
            if condition_value is False:
                err = 'Param "{}" fails PERT requirement "{}".'.format(params, condition_name)
                raise FairException(err)

    def _validate_structured_input(self, target, normalized):
        """Validate structured input after normalization/default application."""
        distribution = normalized['distribution']
        params = normalized['params']

        # Check bounded FAIR factors before sampling.
        if target in self._le_1_targets:
            self._check_le_1(target, **params)

        # General non-negative check.
        self._check_non_negative(**params)

        # Distribution-specific validation.
        if distribution == 'constant':
            self._validate_constant_params(params)

        elif distribution == 'normal':
            self._validate_normal_params(params)

        elif distribution == 'pert':
            self._validate_pert_params(params)

        elif distribution == 'lognormal':
            self._validate_lognormal_params(params)

        elif distribution == 'poisson':
            self._validate_poisson_params(params)

        elif distribution == 'beta':
            self._validate_beta_params(target, params)

        else:
            raise FairException(f'"{distribution}" is not a recognized distribution.')

    def _check_required(self, distribution, params, required):
        """Check whether all required keys are present."""
        for required_key in required:
            if required_key not in params:
                raise FairException(
                    f'"{distribution}" distribution is missing required parameter "{required_key}".'
                )
    def _validate_constant_params(self, params):
        """Validate constant distribution parameters."""
        self._check_required('constant', params, ['constant'])

    def _validate_normal_params(self, params):
        """Validate normal distribution parameters."""
        self._check_required('normal', params, ['mean', 'stdev'])

    def _validate_pert_params(self, params):
        """Validate PERT distribution parameters."""
        self._check_required('pert', params, ['low', 'mode', 'high'])
        self._check_pert(**params)

    def _validate_lognormal_params(self, params):
        """Validate lognormal distribution parameters."""
        self._check_required('lognormal', params, ['mean', 'sigma'])

    def _validate_poisson_params(self, params):
        """Validate poisson distribution parameters."""
        self._check_required('poisson', params, ['lambda', 'range'])

    def _validate_beta_params(self, target, params):
        """Validate beta distribution parameters.

        Supported parameterizations:
        1. alpha/beta
        2. mean/k
        """
        has_alpha_beta = 'alpha' in params and 'beta' in params
        has_mean_k = 'mean' in params and 'k' in params

        if not has_alpha_beta and not has_mean_k:
            raise FairException(
                '"beta" distribution requires either ("alpha", "beta") or ("mean", "k").'
            )

        # If mean is used for a bounded FAIR factor, it must be in [0, 1].
        if 'mean' in params and target in self._le_1_targets:
            self._check_le_1(target, mean=params['mean'])

    def _prepare_input(self, target, **kwargs):
        """Normalize, enrich, and validate an input definition.

        This method centralizes the preprocessing pipeline for all inputs:
        1. normalize legacy or structured input into one internal format
        2. apply defaults or confidence-based shape/range parameters
        3. validate the resulting structured input

        Returns
        -------
        dict
            Fully prepared normalized input record.
        """
        normalized = self._normalize_input(**kwargs)
        normalized = self._resolve_special_input_modes(target, normalized)
        normalized = self._apply_defaults(normalized)
        self._validate_structured_input(target, normalized)
        return normalized

    def generate(self, target, count, **kwargs):
        """Generate simulated values and store both original and resolved metadata."""
        # Preserve the original normalized input state BEFORE defaults are applied.
        original = self._normalize_input(**kwargs)

        # Prepare the resolved internal input used for sampling.
        normalized = self._prepare_input(target, **kwargs)
        result = self._generate_single(count, normalized)

        # Apply output clipping depending on FAIR factor bounds.
        if target in self._le_1_targets:
            result = np.clip(result, 0.0, 1.0)
        else:
            result = np.clip(result, 0.0, np.inf)

        # Store resolved metadata for reporting/model inspection.
        self._supplied_values[target] = {
            'distribution': normalized['distribution'],
            'params': dict(normalized['params']),
            'confidence': normalized['confidence'],
            'input_mode': normalized.get('input_mode')
        }

        # Store original metadata for clean JSON serialization.
        self._original_supplied_values[target] = {
            'distribution': original['distribution'],
            'params': dict(original['params']),
            'confidence': original['confidence'],
            'input_mode': original.get('input_mode')
        }

        return result

    def _generate_single(self, count, normalized):
        """Generate simulated values from a fully prepared input record."""
        distribution = normalized['distribution']
        params = normalized['params']

        try:
            generator = self._generator_map[distribution]
        except KeyError as exc:
            raise FairException(
                f'Unsupported distribution "{distribution}".'
            ) from exc

        return generator(count, **params)

    def generate_multi(self, prefixed_target, count, kwargs_dict):
        """Generate aggregate risk data for multiple targets.

        This method remains backward compatible and still accepts the old
        nested-dictionary style for now.
        """
        final_target = prefixed_target.lstrip('multi_')
        df_dict = {target: pd.DataFrame() for target in kwargs_dict.keys()}

        for target, column_dict in kwargs_dict.items():
            for column, params in column_dict.items():
                data = self.generate(target, count, **params)
                s = pd.Series(data)
                df_dict[target][column] = s

        df1, df2 = df_dict.values()
        combined_df = df1 * df2
        summed = combined_df.sum(axis=1)

        new_target = 'multi_' + final_target
        self._supplied_values[new_target] = kwargs_dict
        self._original_supplied_values[new_target] = kwargs_dict
        return summed

    def supply_raw(self, target, array):
        """Supply raw data directly to the model."""
        clean_array = pd.to_numeric(array)

        if type(array) == pd.Series:
            s = pd.Series(clean_array.values)
        else:
            s = pd.Series(clean_array)

        if s.isnull().any():
            raise FairException('Supplied data contains null values')

        if target in self._le_1_targets:
            if s.max() > 1 or s.min() < 0:
                raise FairException(f'{target} data greater or less than one')

        raw_record = {'raw': s.values.tolist()}
        self._supplied_values[target] = raw_record
        self._original_supplied_values[target] = raw_record
        return s.values

    def _determine_func(self, **kwargs):
        """Legacy helper to infer generator function from old-style keywords."""
        for key in kwargs.keys():
            if key not in self._parameter_map.keys():
                raise FairException('"{}"" is not a recognized keyword'.format(key))

        functions = list(set([
            self._parameter_map[key]
            for key in kwargs.keys()
        ]))

        if len(functions) > 1:
            raise FairException('"{}" mixes incompatible keywords.'.format(str(kwargs.keys())))
        else:
            function = functions[0]
            return function

    def _gen_constant(self, count, **params):
        """Generate a constant-valued array."""
        return np.full(count, params['constant'])

    def _gen_normal(self, count, **params):
        """Generate a normally distributed sample."""
        normal = scipy.stats.norm(loc=params['mean'], scale=params['stdev'])
        return normal.rvs(count)

    def _gen_pert(self, count, **params):
        """Generate a Beta-PERT distributed sample."""
        pert = FairBetaPert(**params)
        return pert.random_variates(count)

    def _gen_lognormal(self, count, **params):
        """Generate a lognormal sample using arithmetic mean and sigma.

        Notes
        -----
        This implementation interprets:
        - mean  = arithmetic mean on the original scale
        - sigma = standard deviation of the underlying normal distribution
        """
        mean = params['mean']
        sigma = params['sigma']

        if mean <= 0:
            raise FairException('"mean" for lognormal must be greater than zero.')

        mu = np.log(mean) - 0.5 * (sigma ** 2)
        return np.random.lognormal(mean=mu, sigma=sigma, size=count)

    def _gen_poisson(self, count, **params):
        """Generate a Poisson sample with uncertain lambda.

        Notes
        -----
        The base lambda is sampled from a symmetric interval:
            lambda_sample ~ Uniform(lambda * (1-range), lambda * (1+range))

        The final Poisson variates are then drawn using that sampled lambda.
        """
        lam = params['lambda']
        rel_range = params['range']

        low = lam * (1.0 - rel_range)
        high = lam * (1.0 + rel_range)

        # Keep lambda non-negative even if the range is large.
        low = max(0.0, low)

        lam_sample = np.random.uniform(low, high, size=count)
        return np.random.poisson(lam=lam_sample, size=count)

    def _gen_beta(self, count, **params):
        """Generate a beta-distributed sample.

        Supported parameterizations:
        - alpha, beta
        - mean, k  -> alpha = mean * k, beta = (1-mean) * k
        """
        if 'alpha' in params and 'beta' in params:
            alpha = params['alpha']
            beta = params['beta']
        else:
            mean = params['mean']
            k = params['k']

            if not 0.0 <= mean <= 1.0:
                raise FairException('"mean" for beta must be between zero and one.')

            alpha = mean * k
            beta = (1.0 - mean) * k

        return np.random.beta(alpha, beta, size=count)