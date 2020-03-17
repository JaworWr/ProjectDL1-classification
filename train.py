from utils import config, data, factory, devices
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Run model training.")
    parser.add_argument(
        dest="config_path",
        type=str,
        help="Path to the configuration file"
    )
    args = parser.parse_args()
    return vars(args)


def main(config_path: str):
    cfg = config.get_config(config_path)

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

    print("Training...")
    trainer = classes["trainer"](cfg, model, data_loaders)
    trainer.train()


if __name__ == '__main__':
    args = get_args()
    main(**args)
