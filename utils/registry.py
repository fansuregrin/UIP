from typing import Dict, Any


class Registry:
    def __init__(self, name):
        self.module_dict: Dict[str, Any] = dict()
        self.name = name

    def register(self, name: str):
        def _register(module: type):
            self.module_dict[name] = module
            return module
        return _register
    
    def get(self, name: str):
        assert name in self.module_dict, f'invalid name: [{name}]'
        return self.module_dict[name]