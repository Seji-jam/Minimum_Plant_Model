from importlib.metadata import version as _v
from .plant.plant import Plant
from .plant.model_handler import ModelHandler   
__all__ = ["Plant", "ModelHandler", "__version__"]
__version__ = _v(__name__)