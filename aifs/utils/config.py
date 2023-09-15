from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class DotConfig(dict):
    def __init__(self, *args, **kwargs):
        for a in args:
            self.update(a)
        self.update(kwargs)

    def __getattr__(self, name):
        if name in self:
            x = self[name]
            if isinstance(x, dict):
                return DotConfig(x)
            return x
        raise AttributeError(name)
