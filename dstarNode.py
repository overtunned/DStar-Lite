import numpy as np

class Node:
    def __init__(self, key, v1, v2):
        self.key = key
        self.v1 = v1
        self.v2 = v2

    def __eq__(self, other):
        return np.sum(np.abs(self.key - other.key)) == 0

    def __ne__(self, other):
        return self.key != other.key

    def __lt__(self, other):
        return (self.v1, self.v2) < (other.v1, other.v2)

    def __le__(self, other):
        return (self.v1, self.v2) <= (other.v1, other.v2)

    def __gt__(self, other):
        return (self.v1, self.v2) > (other.v1, other.v2)

    def __ge__(self, other):
        return (self.v1, self.v2) >= (other.v1, other.v2)