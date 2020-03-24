from utils import config, data, factory, devices
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Run model training.")
    parser.add_argument(
        "-c", "--config",
        dest="config_path",
        type=str,
        help="Path to the configuration file",
        required=True
    )
    parser.add_argument(
        "-s", "--save-checkpoint",
        dest="save_checkpoint",
        default=None,
        type=str,
        help="Where to save the model checkpoint. Overwrites model.save_checkpoint in configuration file"
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
    if args.save_checkpoint is not None:
        cfg.model.save_checkpoint = args.save_checkpoint
    if args.load_checkpoint is not None:
        cfg.model.load_checkpoint = args.load_checkpoint

def main(args):
    cfg = config.get_config(args.config_path)
    process_config(args, cfg)
    data.process_config(cfg)

    devices.device_setup(cfg)

    print("Loading data...")
    data_loaders = data.get_train_valid_loaders(cfg)

    classes = {}
    for k in ["model", "trainer"]:
        module_name = cfg[k].module_name
        class_name = cfg[k].class_name
        classes[k] = factory.get_class(module_name, class_name)

    print("Building model...")
    model = classes["model"](cfg)
    model.build_model()

    print(model.summary())

    if "load_checkpoint" in cfg.model:
        model.load(cfg.model.load_checkpoint)

    trainer = classes["trainer"](cfg, model, data_loaders)
    try:
        print("Training...")
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if "save_checkpoint" in cfg.model:
            model.save(cfg.model.save_checkpoint)


if __name__ == '__main__':
    args = get_args()
    main(args)
