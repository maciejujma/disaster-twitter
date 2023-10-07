from tqdm import tqdm
from typing import List, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def loop_over_dataloader(
    dataloader: "DataLoader",
    model,
    device,
    optimizer,
    lr_scheduler,
    training: bool = True,
    iterations: int = 1,
) -> List[int] | None:
    """Function running model over iterable dataloader's data.

    Args:
        dataloader (DataLoader): Iterable wrapper on pytorch dataset object.
        model (transformers.models): Machine learning model
        device (torch.device): Device context-manager
        optimizer (transformers.optimization): Optimizer
        lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        training (bool, optional): Training job. Defaults to True.
        iterations (int, optional): Number of learning iterations. Defaults to 1.

    Returns:
        List[int] | None: Depending on "training" argument returns list of predictions or None.
    """
    if training:
        model.train()
        progress_bar = tqdm(range(iterations * len(dataloader)))
    else:
        model.eval()
        iterations = 1
        all_predictions = []

    for epoch in range(iterations):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if training:
                outputs = model(**batch)
            else:
                with torch.no_grad():
                    outputs = model(**batch)

            if training:
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            else:
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.append(predictions.detach().cpu().numpy())
        if training:
            print(loss)
    if training:
        return True
    else:
        return all_predictions
