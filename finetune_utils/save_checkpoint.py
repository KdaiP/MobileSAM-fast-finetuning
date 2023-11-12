import torch
import logging
from pathlib import Path

def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: Path):
    """
    Save the current training checkpoint.

    Args:
        state: A dictionary containing the model's state and optimizer's state.
        is_best: A boolean flag to determine if the current checkpoint is the best based on validation loss.
        checkpoint_dir: The directory path where checkpoints are saved.
    """
    last_path = checkpoint_dir / 'last.pth'
    best_path = checkpoint_dir / 'best.pth'
    try:
        torch.save(state, last_path)
        logging.info(f"Checkpoint saved successfully at {last_path}")

        if is_best:
            torch.save(state['model'], best_path)
            logging.info(f"New best checkpoint saved successfully at {best_path}")
    except OSError as e:
        # Log an error message if saving fails
        logging.error(f"Saving checkpoint failed: {e}", exc_info=True)
        raise