from utils import config, data, factory, devices
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument(
        "-c", "--config",
        dest="config_path",
        type=str,
        help="Path to the configuration file",
        required=True
    )
    parser.add_argument(
        "-l", "--load-checkpoint",
        dest="load_checkpoint",
        default=None,
        type=str,
        help="Name of the checkpoint to load. Overwrites model.load_checkpoint in configuration file"
    )
    args = parser.parse_args()
    return args


def process_config(args, cfg):
    if args.load_checkpoint is not None:
        cfg.model.load_checkpoint = args.load_checkpoint


def main(args):
    cfg = config.get_config(args.config_path)
    process_config(args, cfg)
    data.process_config(cfg)

    devices.device_setup(cfg)

    print("Loading data...")
    data_loader = data.get_test_loader(cfg)

    model_module_name = cfg.model.module_name
    model_class_name = cfg.model.class_name
    model_class = factory.get_class(model_module_name, model_class_name)

    print("Building model...")
    model = model_class(cfg)
    model.build_model()
    model.load(cfg.model.load_checkpoint)

    print("Evaluating...")
    result = model.evaluate(
        data_loader,
        workers=cfg.trainer.get("workers", 1),
    )
    for metric, x in zip(["loss"] + cfg.model.metrics, result):
        print(f"{metric}: {x}")


if __name__ == '__main__':
    args = get_args()
    main(args)
