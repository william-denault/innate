import logging
import re
import numpy as np
 

_logger = logging.getLogger('Innate')



def generate_specific_function(expression, coefficients, variable_names):
    """
    Creates a specific function that evaluates an expression with varying variables.

    Parameters
    ----------
    expression : str
        The mathematical expression to evaluate.
    coefficients : dict
        Dictionary containing fixed values for coefficients.
    variable_names : list
        List of variable names that will vary in the expression.

    Returns
    -------
    function
        A NumPy universal function that can handle arrays of varying variables.

    Examples
    --------
    >>> expression = 'a + b / (variable1_range/10000.0) + c * np.log10(variable2_range/10000)'
    >>> coefficients = {'a': 1, 'b': 4, 'c': 7}
    >>> variable_names = ['variable1_range', 'variable2_range']
    >>> specific_function = generate_specific_function(expression, coefficients, variable_names)
    >>> specific_function([10000, 20000, 30000], [10000, 20000, 30000])
    array([5.0, 5.107209969647869, 5.67318211637097], dtype=object)
    """
    def specific_function(*variable_values):
        local_vars = coefficients.copy()  # Copy the dictionary of fixed variables
        # Update variable names with values dynamically from the input
        local_vars.update(dict(zip(variable_names, variable_values)))
        local_vars['np'] = np  # Ensure np is available for np.log10 and other operations
        return eval(expression, {}, local_vars)
    # The number of inputs is now the number of variable names provided
    return np.frompyfunc(specific_function, len(variable_names), 1)


def extract_coef_names(expression):
    """
    Extracts coefficient names from a mathematical expression.

    Parameters
    ----------
    expression : str
        The mathematical expression containing coefficients.

    Returns
    -------
    list
        Sorted list of unique coefficient names.

    Examples
    --------
    >>> expression = 'a + b / (variable1_range/10000.0) + c * np.log10(variable2_range/10000)'
    >>> coef_names = extract_coef_names(expression)
    >>> coef_names
    ['a', 'b', 'c']
    """
    pattern = r'\b[a-zA-Z]\b'
    matches = set(re.findall(pattern, expression))
    return sorted(matches)


def create_coef_dict(coef_names, coef_values):
    """
    Creates a dictionary mapping coefficient names to their corresponding values.

    Parameters
    ----------
    coef_names : list
        List of coefficient names.
    coef_values : list
        List of coefficient values.

    Returns
    -------
    dict
        Dictionary of coefficients and their values.

    Raises
    ------
    TypeError
        If the length of `coef_names` and `coef_values` are not equal.

    Examples
    --------
    >>> coef_names = ['a', 'b', 'c']
    >>> coef_values = [1, 2, 3]
    >>> coefficients = create_coef_dict(coef_names, coef_values)
    >>> coefficients
    {'a': 1, 'b': 2, 'c': 3}
    """
    if len(coef_names) == len(coef_values):
        out = dict(zip(coef_names, coef_values))
    else:
        raise TypeError("length of coefficients names different from the length of coefficients values")
    return out


def extract_variables_names(expression, suffix='_range'):
    """
    Extracts variable names from a mathematical expression that end with a specified suffix.

    Parameters
    ----------
    expression : str
        The mathematical expression containing variable names.
    suffix : str, optional
        The suffix that variable names end with. Default is '_range'.

    Returns
    -------
    list
        Sorted list of unique variable names with the specified suffix.

    Examples
    --------
    >>> expression = 'a + b / (variable1_range/10000.0) + c * np.log10(variable2_range/10000)'
    >>> variable_names = extract_variables_names(expression)
    >>> variable_names
    ['variable1_range', 'variable2_range']
    """
    pattern = rf'\b\w+{re.escape(suffix)}\b'
    matches = set(re.findall(pattern, expression))
    return sorted(list(matches))


def parse_string_equation(data_label, str_eqn, coeffs_eqn, variable_names):

    if (str_eqn is not None) or (coeffs_eqn is not None):
        coef_names = extract_coef_names(str_eqn)
        coeffs_dict = create_coef_dict(coef_names, coeffs_eqn)
        eqn = generate_specific_function(str_eqn, coeffs_dict, variable_names)

    else:
        message = f'Data set "{data_label}" is missing:'
        if str_eqn is None:
            message += f'\nParametrisation formula ("eqn" key in dataset configuration).'
        if coeffs_eqn is None:
            message += f'\nParametrisation coefficients ("eqn_coeffs" key in dataset configuration).'
        _logger.warning(message)

        eqn, coeffs_dict = None, None

    return eqn, coeffs_dict


class Regressor:

    def __init__(self, grid, technique_list, data_cfg=None):

        self.eqn = None
        self.coeffs = None
        self.techniques = []

        # Constrain to regresion techniques
        algorithms = list(set(_setup_cfg['parameter_labels']['reg'].keys()) & set(technique_list))

        # Regular grid Interpolation
        if 'eqn' in algorithms:
            self.techniques.append('eqn')

            # Reconstruct the string equation into a python function
            self.eqn, self.coeffs = parse_string_equation(grid.label,
                                                          data_cfg.get('eqn', None),
                                                          data_cfg.get('eqn_coeffs', None),
                                                          data_cfg.get('axes', None))

        return




