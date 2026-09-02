from __future__ import annotations

from abc import ABC, abstractmethod


class Mesh(ABC):
    """Base class for meshes, mirroring MATLAB Mesh."""

    @property
    def ncells(self) -> int:
        return self.get_ncells()

    @abstractmethod
    def index(self, n, sigma):
        raise NotImplementedError

    @abstractmethod
    def plot(self, var=None):
        raise NotImplementedError

    @abstractmethod
    def get_ncells(self) -> int:
        raise NotImplementedError