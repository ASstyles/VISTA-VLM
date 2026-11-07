from typing import Dict, Any, Callable

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str = None) -> Callable:
        def decorator(cls_or_func: Any) -> Any:
            key = name if name is not None else cls_or_func.__name__
            if key in self._registry:
                raise KeyError(f"'{key}' already registered in {self.name} registry.")
            self._registry[key] = cls_or_func
            return cls_or_func
        return decorator

    def get(self, key: str) -> Any:
        if key not in self._registry:
            raise KeyError(f"'{key}' not found in {self.name} registry. Available: {list(self._registry.keys())}")
        return self._registry[key]

    def list_keys(self):
        return list(self._registry.keys())

MODEL_REGISTRY = Registry("Models")
DATASET_REGISTRY = Registry("Datasets")
BASELINE_REGISTRY = Registry("Baselines")
