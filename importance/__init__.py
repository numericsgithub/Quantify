"""Universal importance analysis for arbitrary PyTorch / Brevitas / Quantify models.

    from importance import analyze, view

    result = analyze(model, dataloader)
    result.save("runs/imp_mymodel")
    view("runs/imp_mymodel")

See docs/llm/importance_analysis.md for the metrics and on-disk format.
"""
from importance.api import analyze, view, load
from importance.storage import Result

__all__ = ["analyze", "view", "load", "Result"]
