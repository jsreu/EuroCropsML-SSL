"""Taken from original Presto code.

https://github.com/nasaharvest/presto/blob/main/presto/dataops/pipelines/ee_pipeline.py
"""

import tempfile

tempdir = tempfile.gettempdir()


class EEPipeline:
    """Used for obtaining, organizing and using data from EarthEngine."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__
        if self.name == "EEPipeline":
            print("Warning: EEPipeline is an abstract class")
        self._ee_task_list = None
        self._client = None
