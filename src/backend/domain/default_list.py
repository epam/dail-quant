class DefaultList(list):
    def __init__(self, list_, def_value):
        super().__init__(list_)
        self.__def_value = def_value

    def __getitem__(self, key):
        if isinstance(key, slice):
            return DefaultList(super().__getitem__(key), self.__def_value)
        elif isinstance(key, int):
            try:
                return super().__getitem__(key)
            except IndexError:
                return self.__def_value
