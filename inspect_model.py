from pathlib import Path
from dataclasses import dataclass

import jax

from infer import load_newest_checkpoint

import jax.numpy as jnp
import numpy as np

from typing import Tuple

def plot_histogram(histogram: Tuple[np.ndarray, np.ndarray],
                   char: str = " ",
                   width: int = 15
) -> str:
    """
    Generates a string representation of a histogram with linear color gradient.

    Returns:
        A string representing the histogram.
    """
    counts, bin_edges = histogram
    max_count = np.max(counts)  # Use numpy's max for efficiency
    output_lines = []

    for i, count in enumerate(counts):
        normalized_count = count / max_count if max_count > 0 else 0  # Avoid division by zero
        
        bar = (char * int(normalized_count * width))
        bin_start = f"{bin_edges[i]:.4f}"
        bin_end = f"{bin_edges[i+1]:.4f}"
        output_lines.append(f"[{bin_start}, {bin_end}]\t{bar} {count}")

    return "\n".join(output_lines)


@dataclass
class NodeStats:
    min: float
    max: float
    l1_norm: float
    all_finite: bool
    histogram: np.array
    unit_histogram: np.array

    def __str__(self) -> str:
        """Custom string representation for pretty printing."""
        finite_str = "Yes" if self.all_finite else "No"
        output = (
            f"Min: {self.min:.4f}\n"
            f"Max: {self.max:.4f}\n"
            f"L1 Norm: {self.l1_norm:.4f}\n"
            f"All Finite: {finite_str}\n"
            f"Histogram:\n{plot_histogram(self.histogram, '█')}\n"
            f"Unit histogram:\n{plot_histogram(self.unit_histogram, '█')}"
        )
        return output


if __name__ == "__main__":
    jax.config.update('jax_default_prng_impl', 'unsafe_rbg')

    current_directory = Path(__file__).resolve().parent
    checkpoint_path = current_directory / "audio_to_midi_checkpoints"
    model, state = load_newest_checkpoint(
        checkpoint_path,
        model_replication=False,  # Disable model sharding as it is not supported by coremlutils
    )

    # Get statistics for the model weights
    def reduce_min_max(acc, leaf):
        cur_min, cur_max = acc
        return (min(np.min(leaf), cur_min), max(np.max(leaf), cur_max))
    global_min, global_max = jax.tree_util.tree_reduce(reduce_min_max, model, (100, -100))

    def collect_stats(leaf) -> NodeStats:
        return NodeStats(
            min=np.min(leaf),
            max=np.max(leaf),
            l1_norm=np.max(np.abs(leaf)),
            all_finite=np.isfinite(leaf).all(),
            histogram=np.histogram(leaf),
            unit_histogram=np.histogram(leaf, range=(global_min, global_max)),
        )
    stats = jax.tree_util.tree_map(collect_stats, model)
    stats = jax.tree_util.tree_leaves_with_path(stats)

    warnings = []
    for path, node_stats in stats:
        print(f"Path {path}")
        print(node_stats)
        print("")

        if not node_stats.all_finite:
            warnings.append((path, node_stats, "Not all finite"))

    if len(warnings) > 0:
        print("")
        print("")
        print("WARNINGS:")
        for path, node_stats, msg in warnings:
            print(f"{msg} at {path}:")
            print(node_stats)
            print("")
    else:
        print("All is looking fine ^^")
