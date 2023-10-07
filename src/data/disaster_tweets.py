# built-in
import re

# 3rd party
import numpy as np
import pandas as pd
from typing import Tuple, List

# hugging face
from datasets import Dataset, DatasetDict
from transformers import pipeline

from src.data.dataset_nlp import DatasetNLP
from src.app.constants import (
    DATA_PATH,
    SEED_VALUE,
    TRAIN_COLUMN,
    VALIDATION_COLUMN,
    TEST_COLUMN,
    TEXT_COLUMN,
    TEXT_LENGTH_COLUMN,
    TRAIN_VAL_SPLIT,
    MASK_TOKEN,
    NEW_SENTENCE_NUMBER,
    SEQUENCE_KEY,
    TARGET_COLUMN,
    LENGHT_LIMIT,
    NEGATIVE_TARGET_VALUE,
    POSITIVE_TARGET_VALUE,
)


class DisasterTweets(DatasetNLP):
    def __init__(self) -> None:
        pass

    def _load_data(self) -> Tuple[pd.DataFrame]:
        """Method loading training and test data.

        Returns:
            Tuple[pd.DataFrame]: Loaded train and test sets.
        """
        train = pd.read_csv(DATA_PATH + "train.csv")
        test = pd.read_csv(DATA_PATH + "test.csv")

        return train, test

    def create_dataset(
        self, validation: bool = False, train_val_ratio: float = TRAIN_VAL_SPLIT
    ) -> None:
        """Method creating Hugging Face dataset objects.

        Args:
            validation (bool, optional): Option to create validation dataset. Defaults to False.
        """
        train, test = self._load_data()

        self.dataset = DatasetDict()
        self.dataset[TRAIN_COLUMN] = Dataset.from_pandas(train)

        if validation:
            self.dataset = self.dataset[TRAIN_COLUMN].train_test_split(
                train_size=train_val_ratio, seed=SEED_VALUE
            )
            self.dataset[VALIDATION_COLUMN] = self.dataset.pop(TEST_COLUMN)
        self.dataset[TEST_COLUMN] = Dataset.from_pandas(test)

    @staticmethod
    def _lowercase_text(batch_examples: List[str]) -> List[str]:
        """Lowering input text

        Args:
            batch_examples (List[str]): strings (tweets) in list.

        Returns:
            List[str]: Lowered strings (tweets) in list.
        """
        return {
            TEXT_COLUMN: [example.lower() for example in batch_examples[TEXT_COLUMN]]
        }

    @staticmethod
    def _remove_urls(batch_examples: List[str]) -> List[str]:
        """Removing URLs from the strings.

        Args:
            batch_examples (List[str]): strings (tweets) in list.

        Returns:
            List[str]: Strings (tweets) without URLs in list.
        """
        url_pattern = r"https?://\S+|www\.\S+"
        return {
            TEXT_COLUMN: [
                re.sub(url_pattern, "", example)
                for example in batch_examples[TEXT_COLUMN]
            ]
        }

    @staticmethod
    def _remove_mentions(batch_examples: List[str]) -> List[str]:
        """Removing twitter's 'mentions' from the strings.

        Args:
            batch_examples (List[str]): strings (tweets) in list.

        Returns:
            List[str]: Strings (tweets) without 'mentions' in list.
        """
        mention_pattern = r"@\w+"
        return {
            TEXT_COLUMN: [
                re.sub(mention_pattern, "", example)
                for example in batch_examples[TEXT_COLUMN]
            ]
        }

    @staticmethod
    def _remove_special_characters(batch_examples: List[str]) -> List[str]:
        """Removing special characters from the strings.

        Args:
            batch_examples (List[str]): strings (tweets) in list.

        Returns:
            List[str]: Strings (tweets) without special characters in list.
        """
        pattern = r"[^\w\s]"
        return {
            TEXT_COLUMN: [
                re.sub(pattern, "", example) for example in batch_examples[TEXT_COLUMN]
            ]
        }

    @staticmethod
    def _compute_text_length(batch_examples: List[str]) -> List[int]:
        """Computing lenght of the strings.

        Args:
            batch_examples (List[str]): strings (tweets) in list.

        Returns:
            List[int]: List of the lenghts of the strings.
        """
        return {
            TEXT_LENGTH_COLUMN: [
                len(example) for example in batch_examples[TEXT_COLUMN]
            ]
        }

    def process_data(self) -> None:
        """Running processing methods and some additional data processing."""
        # running processing methods
        self.dataset = self.dataset.map(self._lowercase_text, batched=True)
        self.dataset = self.dataset.map(self._remove_urls, batched=True)
        self.dataset = self.dataset.map(self._remove_mentions, batched=True)
        self.dataset = self.dataset.map(self._remove_special_characters, batched=True)
        self.dataset = self.dataset.map(self._compute_text_length, batched=True)

        # removing rows with empty "text" (should be in the method, just for the purpose of using lambda)
        self.dataset[TRAIN_COLUMN] = self.dataset[TRAIN_COLUMN].filter(
            lambda x: (x[TEXT_COLUMN] is not None) or (x[TEXT_COLUMN] != "")
        )

        # sorting data to use dynamic padding
        self.dataset[TRAIN_COLUMN] = self.dataset[TRAIN_COLUMN].sort(TEXT_LENGTH_COLUMN)

    @staticmethod
    def _fill_masked_words(sentence: str, unmasker: "pipeline") -> list[str]:
        """Filling masked words in the sentence.

        Args:
            sentence (str): Sentence with the masked words.
            unmasker (pipeline): Model filling the words.

        Returns:
            list[str]: List of sentences with the filled masked words.
        """
        # splitting the sentence
        splitted_sentence: List[str] = sentence.split()

        # randomly replacing one word with the <mask> token
        random_index: int = np.random.randint(0, len(splitted_sentence))
        splitted_sentence[random_index] = MASK_TOKEN

        # reconnecting words into the sentence
        full_sentence = " ".join(splitted_sentence)

        # creating list of new sentences with the masked word replaced by the unmasker
        synthetic_sentences: List[str] = [
            sentence[SEQUENCE_KEY]
            for sentence in unmasker(full_sentence, top_k=NEW_SENTENCE_NUMBER)
        ]

        return synthetic_sentences

    def _create_synthetic_data(
        self, dataset: "Dataset", target_value: int, unmasker: "pipeline"
    ):
        dataset = dataset[TEXT_COLUMN].values

        # for each sentence creating additional synthetic sentences
        synthetic_dataset = [
            self._fill_masked_words(sentence, unmasker) for sentence in dataset
        ]

        # unnesting and creating dataframe
        synthetic_dataset = [item for sublist in synthetic_dataset for item in sublist]
        synthetic_dataset = pd.DataFrame(
            {"text": synthetic_dataset, "target": target_value}
        )

        return synthetic_dataset

    def adding_synthetic_data(self) -> None:
        """Adding generated data to the dataset."""
        unmasker: "pipeline" = pipeline("fill-mask")

        # taking the long negative/positive sentences
        dataset_neg: "Dataset" = self.dataset[TRAIN_COLUMN].filter(
            lambda x: (x[TEXT_LENGTH_COLUMN] > LENGHT_LIMIT)
            and (x[TARGET_COLUMN] == NEGATIVE_TARGET_VALUE)
        )
        dataset_pos: "Dataset" = self.dataset[TRAIN_COLUMN].filter(
            lambda x: (x[TEXT_LENGTH_COLUMN] > LENGHT_LIMIT)
            and (x[TARGET_COLUMN] == POSITIVE_TARGET_VALUE)
        )

        # modifying datasets to pandas
        train_dataset = self.dataset[TRAIN_COLUMN].to_pandas()
        dataset_pos = dataset_pos.to_pandas()
        dataset_neg = dataset_neg.to_pandas()

        # creating synthetic datasets
        synthetic_pos = self._create_synthetic_data(
            dataset=dataset_pos, target_value=POSITIVE_TARGET_VALUE, unmasker=unmasker
        )
        synthetic_neg = self._create_synthetic_data(
            dataset=dataset_neg, target_value=NEGATIVE_TARGET_VALUE, unmasker=unmasker
        )

        # concatenating datasets
        train_dataset = pd.concat([train_dataset, synthetic_neg])
        train_dataset = pd.concat([train_dataset, synthetic_pos])
        self.dataset[TRAIN_COLUMN] = Dataset.from_pandas(train_dataset)
        self.dataset.set_format("pytorch")
