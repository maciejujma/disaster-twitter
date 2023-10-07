# paths
DATA_PATH: str = "data/"

# random
SEED_VALUE: int = 23

# model
PRETRAINED_MODEL_NAME: str = "distilbert-base-uncased"

# training
USE_VALIDATION: bool = False
SHUFFLE_DATA: bool = False
TRAIN_VAL_SPLIT: float = 0.9
BATCH_SIZE: int = 16
NUM_LABELS: int = 2
NUM_EPOCHS: int = 1

# optimizer
LEARNING_RATE: float = 5e-5

# scheduler
SCHEDULER_WARMUP_STEPS: int = 0
SCHEDULER_TYPE: str = "linear"

# columns
NEGATIVE_TARGET_VALUE: int = 0
POSITIVE_TARGET_VALUE: int = 1
LABELS_COLUMN: str = "labels"
TRAIN_COLUMN: str = "train"
ID_COLUMN: str = "id"
VALIDATION_COLUMN: str = "validation"
TEST_COLUMN: str = "test"
TEXT_COLUMN: str = "text"
TEXT_LENGTH_COLUMN: str = "text_length"
TARGET_COLUMN: str = "target"
COLUMNS_TO_REMOVE: dict = {
    "global": ["text", "keyword", "location", "id", "text_length"],
    "train": ["__index_level_0__"],
}
COLUMNS_TO_RENAME: dict = {
    "train": {"target": "labels"},
    "validation": {"target": "labels"},
}

# synthetic data
NEW_SENTENCE_NUMBER: int = 2
LENGHT_LIMIT: int = 130
MASK_TOKEN: str = "<mask>"
SEQUENCE_KEY: str = "sequence"
