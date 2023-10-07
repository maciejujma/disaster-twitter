from typing import TYPE_CHECKING

from src.app.constants import (
    TEXT_COLUMN,
    COLUMNS_TO_REMOVE,
    COLUMNS_TO_RENAME,
)

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import AutoTokenizer


class DatasetNLP:
    dataset: "Dataset"

    def tokenize_data(
        self, tokenizer: "AutoTokenizer", truncation: bool = True
    ) -> None:
        """Tokenizing dataset

        Args:
            tokenizer (AutoTokenizer): Pretrained tokenizer.
            truncation (bool, optional): Option to truncate the sentence. Defaults to True.
        """

        def tokenize_function(example: dict) -> dict:
            """Method tokenizing string sentence.

            Args:
                example (dict): Dictionary with item containing string sentence.

            Returns:
                dict: Dictionary with item containing tokenized sentence.
            """
            return tokenizer(example[TEXT_COLUMN], truncation=truncation)

        self.dataset = self.dataset.map(tokenize_function, batched=True)

    def remove_columns(self) -> None:
        """Method removing columns from dataset."""
        for key, value in COLUMNS_TO_REMOVE.items():
            if key == "global":
                self.dataset = self.dataset.remove_columns(value)
            else:
                self.dataset[key] = self.dataset[key].remove_columns(value)

    def rename_columns(self) -> None:
        """Method renaming columns from dataset."""
        for key, name_dict in COLUMNS_TO_RENAME.items():
            if key not in self.dataset:
                continue
            for old_name, new_name in name_dict.items():
                self.dataset[key] = self.dataset[key].rename_column(old_name, new_name)
