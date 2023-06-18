import utils
import torch
from absl import app
from absl import logging

from pprint import pprint

from absl.logging import PythonFormatter
from run import load_model
import pdb


class CustomPythonFormatter(PythonFormatter):
    def format(self, record):
        grey = "\x1b[38;21m"
        reset = "\x1b[0m"
        # note - colors not appearing on osx
        return f"{grey}[{record.levelname}]{reset} {record.getMessage()}"


def setup_logging():
    logging.set_verbosity(logging.INFO)
    logging.get_absl_handler().setFormatter(CustomPythonFormatter())


def get_model(optimize, height, square, device):
    # logging.log(logging.INFO, "Loading model...")
    model_path, model_type = ("weights/dpt_beit_large_512.pt", "dpt_beit_large_512")
    # model_path, model_type = ("weights/dpt_levit_224.pt", "dpt_levit_224")
    # model_path, model_type = ("weights/dpt_swin2_tiny_256.pt", "dpt_swin2_tiny_256")

    model, transform, net_w, net_h = load_model(
        device, model_path, model_type, optimize, height, square
    )
    logging.log(logging.INFO, "Done loading model...")
    return model, transform, net_w, net_h


def walk_params(model):
    for idx, param in enumerate(model.named_parameters()):
        logging.log(logging.INFO, " %3d %-70s %s", idx, param[0], list(param[1].shape))


def main(argv):
    setup_logging()

    optimize = False
    height = None
    square = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, net_w, net_h = get_model(optimize, height, square, device)
    logging.log(logging.INFO, "Pretrained model")
    walk_params(model.pretrained)
    logging.log(logging.INFO, "Scratch model")
    walk_params(model.scratch)

    image_name = "input/living_room.png"
    original_image_rgb = utils.read_image(image_name)  # in [0, 1]
    logging.log(logging.INFO, f"Original image shape {original_image_rgb.shape}")

    logging.log(logging.INFO, f"Done")


if __name__ == "__main__":
    app.run(main)
