from tkinter import Tk
from tkinter.filedialog import askopenfilename

import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch

import model
import utility
from data import common
from option import args

matplotlib.use("TkAgg")


torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
device = torch.device("cpu" if args.cpu else "cuda")


def main():
    global _model, device

    _model = model.Model(args, checkpoint)
    print(
        "Total params: %.2fM"
        % (sum(p.numel() for p in _model.parameters()) / 1000000.0)
    )

    Tk().withdraw()
    filename = askopenfilename()

    img = cv2.imread(filename)

    torch.set_grad_enabled(False)
    _model.eval()

    def prepare(img):
        def _prepare(tensor):
            return tensor.to(device)

        return _prepare(img)

    lr = img.copy()
    # lr, = common.set_channel(lr, n_channels=args.n_colors)
    (lr,) = common.np2Tensor(lr, rgb_range=args.rgb_range)
    lr = prepare(lr.unsqueeze(0))
    sr = _model(lr, 0)
    sr = utility.quantize(sr, args.rgb_range).squeeze(0)

    normalized = sr * 255 / args.rgb_range
    ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()

    plt.imshow(cv2.cvtColor(ndarr, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    main()

# python demo.py --model CSNLN --scale 2 --n_feats 128 --depth 12 --pre_train ../models/CSNLN_x2.pt --test_only --chop

# python demo.py --model RCAN --scale 2 --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../models/RCAN_BIX2.pt --test_only --chop
