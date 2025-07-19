from dataclasses import dataclass, field

import numpy as np
import spark_config as sc

from semantic_inference.misc import Logger
from semantic_inference.models.openset_segmenter import Results


class ComponentVisualizer:
    """Visualize features by three components."""

    def __init__(self, config):
        self.config = config
        if len(self.config.components) != 3:
            Logger.error(f"Invalid components specified: {self.config.components}!")
            self.config.components = [0, 1, 2]

        self._indices = np.array(self.config.components)

    def call(self, results: Results) -> np.ndarray:
        colors = results.features[:, self._indices].numpy()
        c_min = np.min(colors, axis=0)
        c_max = np.max(colors, axis=0)
        c_range = c_max - c_min
        c_range[c_range <= 1.0e-3] = 1.0
        colors = 255 * ((colors - c_min) / c_range)
        colors = np.vstack(([0, 0, 0], colors))
        colors = colors.astype(np.uint8)
        return colors[results.instances]


@sc.register_config("feature_visualizer", "component", ComponentVisualizer)
@dataclass
class ComponentVisualizerConfig(sc.Config):
    """Configuration for component visualizer."""

    components: list[int] = field(default_factory=lambda: [0, 1, 2])
