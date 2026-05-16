"""This module contains a class for creating composite FAIR models."""

import datetime
import json
import uuid

import pandas as pd

from ..utility import FairException

from .model import FairModel


class FairMetaModel(object):
    """A class for aggregating or comparing FAIR models.

    An instance of this class is created by taking multiple FAIR models and
    either rolling the total risk into a collection, or MetaModel, or
    comparing the component model risks against a baseline model.

    Parameters
    ----------
    name : str
        A human-readable designation for identification
    models : list of FairModels
        The sub-models that roll up into the aggregate MetaModel risk
        calculation.
    mode : str, optional
        Metamodel mode. Supported values are 'sum' and 'compare'.
        Default is 'sum'.
    baseline_model : str, optional
        Required when mode='compare'. Must match the name of a component
        model, i.e. the value returned by FairModel.get_name().
    model_uuid : str, optional
        uuid.uuid4 string (default is None, meaning one will be assigned)
    creation_date : str, optional
        Creation date (default is None, meaning one will be assigned)

    Examples
    --------
    >>> m1 = pyfair.model.FairModel.from_json('model_1.json')
    >>> m2 = pyfair.model.FairModel.from_json('model_2.json')
    >>> meta1 = pyfair.model.FairMetaModel('Name', [m1, m2])
    >>> meta1.calculate_all()
    >>> meta1.export_results()

    >>> meta2 = pyfair.model.FairMetaModel(
    ...     'Comparison', [m1, m2], mode='compare',
    ...     baseline_model=m1.get_name()
    ... )
    >>> meta2.calculate_all()
    >>> meta2.export_results()

    .. warning:: Do not supply your own UUID/creation date unless you
        want to break things.

    """

    def __init__(
        self,
        name,
        models,
        mode='sum',
        baseline_model=None,
        model_uuid=None,
        creation_date=None
    ):
        self._name = name
        self._params = {}
        self._risk_table = pd.DataFrame()
        self._mode = mode
        self._baseline_model = baseline_model
        # For every model, flatten and save params.
        for model in models:
            # If model, load
            if type(model) == FairModel:
                self._load_model(model)
            # If metamodel, load components.
            elif type(model) == type(self):
                self._load_meta_model(model)
            else:
                err = f'Input {model} is not a FairModel or FairMetaModel.'
                raise FairException(err)
        # Assign UUID
        if model_uuid and creation_date:
            self._model_uuid = model_uuid
            self._creation_date = creation_date
        else:
            self._model_uuid = str(uuid.uuid1())
            self._creation_date = str(datetime.datetime.now())

    def get_name(self):
        """Returns the model name.

        Returns
        -------
        str
            The name of the model.

        """

        return self._name

    def get_uuid(self):
        """Returns the model's unique ID.

        Returns
        -------
        str
            The UUID of the model.

        """

        return self._model_uuid

    @staticmethod
    def read_json(json_data):
        """Static method to create a metamodel from a JSON string

        Parameters
        ----------
        json_data : str
            a UTF-8 encoded JSON string containing model data

        Returns
        -------
        pyfair.model.FairMetaModel
            A meta model instance using the JSON parameters supplied

        Raises
        ------
        pyfair.fair_exception.FairException
            If model JSON is supplied erroneously

        Example
        -------
        >>> with open('serialized_metamodel.json') as f:
        ...     json_text = f.read()
        ...
        >>> metamodel = pyfair.model.FairMetaModel.read_json(json_text)

        """
        data = json.loads(json_data)
        # Check type of JSON
        if data['type'] != 'FairMetaModel':
            raise FairException('Failed JSON parse attempt. This is not a FairMetaModel.')
        # Get model params
        meta_keys = [
            'name', 'model_uuid', 'type', 'creation_date',
            'mode', 'baseline_model'
        ]
        model_params = {
            key: value
            for key, value
            in data.items()
            if key not in meta_keys
        }
        # Instantiate models (there is no need to cover)
        # metamodels here because metamodel json only
        # has models in it, not metamodels.
        models = [
            FairModel.read_json(json.dumps(value))
            for value
            in model_params.values()
        ]
        # Create metamodel
        meta_model = FairMetaModel(
            name=data['name'],
            models=models,
            mode=data.get('mode', 'sum'),
            baseline_model=data.get('baseline_model'),
            model_uuid=data['model_uuid'],
            creation_date=data['creation_date']
        )
        return meta_model

    def _load_model(self, model):
        """Loads an individual model into the metamodel"""
        self._record_params(model)
        self._calculate_model(model)

    def _load_meta_model(self, meta_model):
        """Loads a metamodel into the metamodel"""
        params = meta_model.export_params()
        params = {
            key: value
            for key, value
            in params.items()
            if key not in [
                'name', 'model_uuid', 'type', 'creation_date',
                'mode', 'baseline_model'
            ]
        }
        # Iterate through params
        for model_params in params.values():
            # If model, load model from json and load into meta
            model_json = json.dumps(model_params)
            model = FairModel.read_json(model_json)
            self._load_model(model)

    def _record_params(self, model):
        """Record the params used for later serialization"""
        model_params = json.loads(model.to_json())
        model_name = model_params['name']
        self._params[model_name] = model_params

    def _calculate_model(self, model):
        """Calculate a component model"""
        # For each model, calculate and put output results in dataframe.
        model_json = json.loads(model.to_json())
        name = model_json['name']
        model.calculate_all()
        results = model.export_results()
        self._risk_table[name] = results['Risk']

    def _validate_mode_configuration(self):
        """Validate metamodel mode and baseline configuration."""
        valid_modes = ['sum', 'compare']
        if self._mode not in valid_modes:
            raise FairException(
                f"Unsupported metamodel mode '{self._mode}'. "
                f"Expected one of {valid_modes}."
            )

        if self._mode == 'compare':
            if not self._baseline_model:
                raise FairException(
                    "baseline_model must be supplied when mode='compare'."
                )

            component_names = list(self._params.keys())
            if self._baseline_model not in component_names:
                raise FairException(
                    f"baseline_model '{self._baseline_model}' is not a "
                    "component model of this FairMetaModel."
                )

    def _drop_derived_columns(self):
        """Remove columns produced by sum/compare calculations."""
        derived_columns = [
            c for c in self._risk_table.columns
            if c == 'Risk' or str(c).startswith('Delta::')
        ]
        if derived_columns:
            self._risk_table = self._risk_table.drop(columns=derived_columns)

    def export_params(self):
        """Returns the parameters used to generate the metamodel

        Returns
        -------
        dict
            The metamodel and component model parameters.

        """

        return self._params

    def _calculate_sum(self):
        """Aggregate component model risk into a single Risk column."""
        self._drop_derived_columns()

        sum_vector = self._risk_table.sum(axis=1)
        self._risk_table['Risk'] = sum_vector
        # Check for NaN values in sum_vector
        if pd.isnull(sum_vector).any():
            raise FairException('np.NaN values in summed Risk column.'
                                ' Likely cause: n_simulations mismatch across models.')
        return self

    def _calculate_compare(self):
        """Calculate delta columns against the configured baseline model."""
        self._drop_derived_columns()

        baseline = self._risk_table[self._baseline_model]

        for column in self._risk_table.columns.tolist():
            if column == self._baseline_model:
                continue
            delta_name = f'Delta::{column}'
            self._risk_table[delta_name] = self._risk_table[column] - baseline

        return self

    def calculate_all(self):
        """Calculate all the component models and the metamodel itself

        Returns
        -------
        pyfair.model.FairMetaModel
            Reference to current instance.

        """
        self._validate_mode_configuration()

        if self._mode == 'sum':
            return self._calculate_sum()

        if self._mode == 'compare':
            return self._calculate_compare()

        raise FairException(f"Unhandled metamodel mode '{self._mode}'.")

    def export_results(self):
        """Returns the metamodel calculations

        Returns
        -------
        pd.DataFrame
            A dataframe with columns of risk outputs for the various
            models. In mode='sum', includes total sum in column 'Risk'.
            In mode='compare', includes delta columns prefixed with
            'Delta::'.

        """

        return self._risk_table

    def to_json(self):
        """Returns serialized, JSON-formatted data suitable for storage

        Returns
        -------
        str
            JSON-formatted data and meta data that contains everything
            (parameters including random seed) needed to reproduce the
            model.

        """
        data = {**self._params}
        data['name'] = str(self._name)
        data['model_uuid'] = self._model_uuid
        data['creation_date'] = self._creation_date
        data['type'] = str(self.__class__.__name__)
        data['mode'] = self._mode
        data['baseline_model'] = self._baseline_model
        json_data = json.dumps(
            data,
            indent=4,
        )
        return json_data

    def calculation_completed(self):
        """Checks if the calculation is complete.

        Returns
        -------
        bool
            Returns True if the metamodel calculation has already been
            conducted. Otherwise False.

        """
        if self._mode == 'sum':
            return 'Risk' in self._risk_table.columns

        if self._mode == 'compare':
            return any(
                str(c).startswith('Delta::')
                for c in self._risk_table.columns
            )

        return False
