import numpy as np
import yaml
from pint import Quantity

from src import controllers, dynamics, integrators, primitives, system, ureg

# Following classes use generic "constructor" function (best suited for dataclasses)
constructable_classes = {
    "ConstantController": controllers.ConstantController,
    "LQRController": controllers.LQRController,
    "BasicDynamics": dynamics.BasicDynamics,
    "RK4Integrator": integrators.RK4Integrator,
    "State": primitives.State,
    "Actuator": system.Actuator,
    "Cart": system.Cart,
    "Pendulum": system.Pendulum,
    "System": system.System,
    # Add other classes as needed
}

def add_yaml_constructors() -> None:
    """Add PyYAML constructors to parse custom classes and pint Quantities from YAML files."""
    # Custom constructors for core classes
    for class_name in constructable_classes:
        yaml_tag = f"!{class_name}"
        yaml.SafeLoader.add_constructor(yaml_tag, constructor)

    # Custom constructors for other types
    yaml.SafeLoader.add_constructor("!unit", pint_constructor)
    yaml.SafeLoader.add_constructor("!array", array_constructor)
    yaml.SafeLoader.add_constructor("!range", range_constructor)

def constructor(loader: yaml.Loader, node: yaml.MappingNode) -> object:
    """Custom constructor from YAML inputs for every class enumerated above.

        Args:
            loader (yaml.SafeLoader): Loader used to generate object from YAML inputs.
            node (yaml.MappingNode): YAML input parameters describing the object.

        Returns:
            any: Object of any type that uses this constructor function (enumerated above).

    """  # fmt: skip

    class_type = constructable_classes[node.tag.replace("!", "")]
    value = loader.construct_mapping(node, deep=True)
    return class_type(**value)

def pint_constructor(loader: yaml.Loader, node: yaml.Node) -> Quantity:
    if isinstance(node.value, str):
        value = loader.construct_scalar(node)
        return ureg(value)
    if isinstance(node.value, list):
        value = loader.construct_sequence(node, deep=True)
        return value[0] * ureg(value[1])
    msg = f"Invalid format for pint Quantity: {node.value}"
    raise ValueError(msg)

def array_constructor(loader: yaml.Loader, node: yaml.SequenceNode) -> np.ndarray:
    value = loader.construct_sequence(node, deep=True)
    return np.array(value)

def range_constructor(loader: yaml.Loader, node: yaml.SequenceNode) -> np.ndarray:
    """!range tag custom constructor to define a range from [start], stop, [step] inputs.

        Args:
            loader (yaml.SafeLoader): Loader used to generate object from YAML inputs.
            node (yaml.SequenceNode): List of inputs, interpreted based on number of inputs:
                - 1 input: node = [stop]
                - 2 inputs: node = [start, stop]
                - 3 inputs: node = [start, stop, step]

        Raises:
            ValueError: Input more than 3 elements or less than 1 element.

        Returns:
            np.ndarray: array of numbers in the range.

    """  # fmt: skip

    sequence = loader.construct_sequence(node, deep=True)
    match len(sequence):
        case 1:
            start = 0
            stop = sequence[0]
            step = 1
        case 2:
            start = sequence[0]
            stop = sequence[1]
            step = 1
        case 3:
            start = sequence[0]
            stop = sequence[1]
            step = sequence[2]
        case _:
            msg = "Invalid range format. Expected [start], stop, [step]."
            raise ValueError(msg)

    list_range = np.arange(start, stop, step)

    return list_range
