class ElementNetwork:
    def __init__(self):
        self.pn = None

    def __setitem__(self, idx, value):
        self.pn[idx] = value
        setattr(self, idx.split(".")[1], value)
