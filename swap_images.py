import argparse
import torch

from icm.util import instantiate_from_config
from omegaconf import OmegaConf
from tqdm import tqdm

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.eval()
    return model

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # if args.checkpoint:
    #     path = args.checkpoint.split('checkpoints')[0]
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config.model, args.checkpoint)
    model.eval()
    model.cuda()
    model.freeze()
    model = model.cpu()
    torch.save(model, args.save_path)
    print(f"Model saved to {args.save_path}")
    print("Done")