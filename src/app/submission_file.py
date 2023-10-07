from typing import TYPE_CHECKING

import pandas as pd

from src.app.constants import (
    DATA_PATH,
    ID_COLUMN,
    TARGET_COLUMN,
)

if TYPE_CHECKING:
    import numpy as np


class SubmissionFile:
    predictions: "np.array"
    submission_dataframe: "pd.DataFrame"

    def __init__(self, predictions: "np.array") -> None:
        self.predictions = predictions
        self._create_submission_file()

    def _create_submission_file(
        self, test_data_path: str = DATA_PATH + "test.csv"
    ) -> None:
        self.submission_dataframe = pd.DataFrame(
            {
                ID_COLUMN: pd.read_csv(test_data_path)[ID_COLUMN],
                TARGET_COLUMN: self.predictions,
            }
        )

    def save_submission_file(
        self, saving_path: str = DATA_PATH + "outputs/twitter-disaster-predictions.csv"
    ) -> None:
        self.submission_dataframe.to_csv(saving_path, index=False)
