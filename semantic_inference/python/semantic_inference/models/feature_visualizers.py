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
        colors = self.config.scale * (colors + self.config.offset)
        colors = 255 * np.clip(colors, 0.0, 1.0)
        colors = np.vstack(([0, 0, 0], colors))
        colors = colors.astype(np.uint8)
        return colors[results.instances]


@sc.register_config("feature_visualizer", "component", ComponentVisualizer)
@dataclass
class ComponentVisualizerConfig(sc.Config):
    """Configuration for component visualizer."""

    components: list[int] = field(default_factory=lambda: [-1, -2, -3])
    offset: float = 0.5
    scale: float = 0.5
