from __future__ import annotations

from abc import ABC, abstractmethod

from .SigmaZetaMesh import SigmaZetaMesh


class SigmaZetaMeshGenerator(ABC):
    """Base class to produce SigmaZetaMesh objects."""

    @abstractmethod
    def get_mesh(self) -> SigmaZetaMesh:
        raise NotImplementedError
