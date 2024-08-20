from enum import Enum
import types
from src.smlp_py.solvers.z3.solver import Form2_Solver
from src.smlp_py.solvers.marabou.solver import Pysmt_Solver


class Solver:
    class Version(Enum):
        FORM2 = 0
        PYSMT = 1

    _instance = None
    version = None

    def __new__(cls, *args, **kwargs):
        version = kwargs["version"]
        if isinstance(version, cls.Version):
            cls.version = version
        else:
            raise ValueError("Must be a valid version")

        if cls._instance is None and isinstance(cls.version, cls.Version):
            if cls.version == cls.Version.PYSMT:
                specs = kwargs["specs"]
                cls._instance = Pysmt_Solver(specs)
            else:
                cls._instance = Form2_Solver()
            cls._map_instance_methods()
        return cls._instance

    @classmethod
    def _map_instance_methods(cls):
        """Automatically maps all methods from the instance to the SingletonFactory class."""
        for base_class in cls._instance.__class__.__mro__:
            for name, method in base_class.__dict__.items():
                if isinstance(method, types.FunctionType):
                    if not hasattr(cls, name):
                        setattr(cls, name, cls._create_delegator(name))

    @classmethod
    def _create_delegator(cls, method_name):
        """Create a method that delegates the call to the _instance."""
        def delegator(*args, **kwargs):
            return getattr(cls._instance, method_name)(*args, **kwargs)
        return delegator




    #
    # @classmethod
    # def _map_instance_properties(cls):
    #     """Automatically maps all properties from the instance to the Solver class."""
    #     for name, attribute in cls._instance.__class__.__dict__.items():
    #         if isinstance(attribute, property):
    #             # Map property to Solver class
    #             if not hasattr(cls, name):
    #                 setattr(cls, name, cls._create_property_delegator(name))
    #
    # @classmethod
    # def _create_property_delegator(cls, property_name):
    #     """Create a property that delegates access to the _instance."""
    #     def getter(self):
    #         return getattr(self._instance, property_name)
    #
    #     def setter(self, value):
    #         setattr(self._instance, property_name, value)
    #
    #     def deleter(self):
    #         delattr(self._instance, property_name)
    #
    #     # Return a property with the mapped getter, setter, and deleter
    #     return property(getter, setter, deleter)
