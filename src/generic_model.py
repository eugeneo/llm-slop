"""
This file contains the abstract class for the generic model
"""

from abc import ABC, abstractmethod
from typing import List

from dataclasses import dataclass


@dataclass
class Message:
    """
    Single message.
    """

    content: str
    role: str = "user"


class Model(ABC):
    """
    Abstract model.
    """

    @abstractmethod
    def chat(self, messages: List[Message]) -> str:
        """
        Sends messages to the model and returns the response.
        """
