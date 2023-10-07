import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_scheduler,
)

from src.app.constants import (
    PRETRAINED_MODEL_NAME,
    LABELS_COLUMN,
    TRAIN_COLUMN,
    TEST_COLUMN,
    VALIDATION_COLUMN,
    USE_VALIDATION,
    BATCH_SIZE,
    SHUFFLE_DATA,
    NUM_LABELS,
    LEARNING_RATE,
    NUM_EPOCHS,
    SCHEDULER_WARMUP_STEPS,
    SCHEDULER_TYPE,
)
from src.app.submission_file import SubmissionFile
from src.app.training import loop_over_dataloader
from src.data.disaster_tweets import DisasterTweets

if USE_VALIDATION:
    from sklearn.metrics import f1_score

# loading tokenizer, data collator and classification model
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS
)

# loading and preparing data
disaster_tweets = DisasterTweets()
disaster_tweets.create_dataset()
disaster_tweets.process_data()
disaster_tweets.adding_synthetic_data()
disaster_tweets.tokenize_data(tokenizer=tokenizer)
disaster_tweets.remove_columns()
disaster_tweets.rename_columns()
disaster_tweets.dataset.set_format("torch")

# creating pytorch data loaders
train_dataloader = DataLoader(
    disaster_tweets.dataset[TRAIN_COLUMN],
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE_DATA,
    collate_fn=data_collator,
)

if USE_VALIDATION:
    validation_dataloader = DataLoader(
        disaster_tweets.dataset[VALIDATION_COLUMN],
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_DATA,
        collate_fn=data_collator,
    )

test_dataloader = DataLoader(
    disaster_tweets.dataset[TEST_COLUMN],
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)

# setting up optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# setting up number of training steps
num_training_steps = NUM_EPOCHS * len(train_dataloader)

# configuring learning rate scheduler
lr_scheduler = get_scheduler(
    SCHEDULER_TYPE,
    optimizer=optimizer,
    num_warmup_steps=SCHEDULER_WARMUP_STEPS,
    num_training_steps=num_training_steps,
)

# putting model on GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# training loop
loop_over_dataloader(
    dataloader=train_dataloader,
    model=model,
    device=device,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    iterations=NUM_EPOCHS,
)

# run validation if required
if USE_VALIDATION:
    all_predictions = loop_over_dataloader(
        dataloader=validation_dataloader,
        model=model,
        device=device,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        training=False,
    )
    labels = disaster_tweets.dataset[VALIDATION_COLUMN][LABELS_COLUMN].numpy()
    all_predictions = np.concatenate(all_predictions, axis=0)
    print(f"F1 score: {f1_score(labels, all_predictions)}")


# test predictions
test_predictions = loop_over_dataloader(
    dataloader=test_dataloader,
    model=model,
    device=device,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    training=False,
)
test_predictions = np.concatenate(test_predictions, axis=0)

# saving predictions in submission file
submission_file = SubmissionFile(predictions=test_predictions)
submission_file.save_submission_file()
