import torch

import data
import loss
import model
import utility
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def main():
    global model
    if args.data_test == ["video"]:
        from videotester import VideoTester

        model = model.Model(args, checkpoint)
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            print(
                "Total params: %.2fM"
                % (sum(p.numel() for p in _model.parameters()) / 1000000.0)
            )
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()


if __name__ == "__main__":
    main()

# python main.py --chop --batch_size 1 --model CSNLN --scale 2 --patch_size 96 --save CSNLN_x2 --n_feats 128 --depth 12 --data_train DIV2K --save_models
