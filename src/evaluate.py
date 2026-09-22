import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

TASKS = {"tissue_type": "type", "zoom_level": "zoom", "focus_quality": "focus"}


@torch.no_grad()
def evaluate(model, dataset, processor, max_new_tokens=200):
    """Generate on the validation set and report accuracy / weighted F1 per task."""
    model.eval()
    texts = []
    for batch in tqdm(dataset.val_loader, desc="Evaluating"):
        batch = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        texts += processor.batch_decode(
            outputs[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
        )

    predictions = [dataset.postprocess(t) for t in texts]
    metrics = {}
    for task, column in TASKS.items():
        y_true = list(dataset.val_dataset[column])
        y_pred = [p[task] for p in predictions]
        metrics[f"{task}_accuracy"] = accuracy_score(y_true, y_pred)
        metrics[f"{task}_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["overall_accuracy"] = sum(metrics[f"{t}_accuracy"] for t in TASKS) / len(TASKS)
    metrics["overall_f1"] = sum(metrics[f"{t}_f1"] for t in TASKS) / len(TASKS)

    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
    return metrics
