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
        workers=cfg.training.get("workers", 1),
        use_multiprocessing=cfg.training.get("use_multiprocessing", False)
    )
    for metric, x in zip(["loss"] + cfg.training.metrics, result):
        print(f"{metric}:\t {x}")


if __name__ == '__main__':
    args = get_args()
    main(**args)
