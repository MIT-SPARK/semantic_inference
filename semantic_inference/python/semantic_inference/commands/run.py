import logging

import click
import spark_config as sc

from semantic_inference.image_io import bag_image_store, parse_image_msg


@click.group(name="run")
def cli():
    """Subcommands for running models."""
    pass


@cli.command()
@click.argument("bag_path", type=click.Path(exists=True))
@click.option("--topic", "-t", multiple=True, type=str)
@click.option("--config", "-c", multiple=True, type=str)
@click.option("--config-file", "-f", multiple=True, type=click.Path(exists=True))
@click.option("--rgb-topics", multiple=True, type=str)
@click.option("--max-images", "-m", type=int, default=None)
def bag(bag_path, topic, config, config_file, rgb_topics, max_images):
    sc.discover_plugins("semantic_inference")
    model_config = sc.load_yaml(config, config_file)
    model = sc.construct("closed_set_model", model_config)

    with bag_image_store(bag_path, topic, with_output=True) as bags:
        bag_in, bag_out = bags

        for idx, ret in enumerate(bag_in):
            if max_images is not None and idx >= max_images:
                break

            topic, msg, t = ret
            img = parse_image_msg(msg)
            if img is None:
                logging.error(f"Invalid msg on topic '{topic}' at {t}")
                continue

            if topic not in rgb_topics:
                img = img[:, :, ::-1]

            labels = model.segment(img)
            bag_out.write(msg.header, t, topic, labels)
