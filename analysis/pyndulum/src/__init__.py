from pint import UnitRegistry, set_application_registry

ureg = UnitRegistry()
ureg.setup_matplotlib()
set_application_registry(ureg)
