import argparse
import torch
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import numpy as np
from utils.train_eval_util import set_val_loader, set_ood_loader_ImageNet
from utils.detection_util import get_and_print_results
from utils.plot_util import plot_distribution
import trainers.locoop
import datasets.imagenet


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.lambda_value:
        cfg.lambda_value = args.lambda_value

    if args.topk:
        cfg.topk = args.topk


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.LOCOOP = CN()
    cfg.TRAINER.LOCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.LOCOOP.CSC = False  # class-specific context
    cfg.TRAINER.LOCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.LOCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.LOCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    import clip_w_local
    cfg = setup_cfg(args)
    _, preprocess = clip_w_local.load(cfg.MODEL.BACKBONE.NAME)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if args.in_dataset in ['imagenet']:
        out_datasets = ['iNaturalist', 'SUN', 'places365', 'Texture']

    trainer = build_trainer(cfg)

    trainer.load_model(args.model_dir, epoch=args.load_epoch)

    id_data_loader = set_val_loader(args, preprocess)

    in_score_mcm, in_score_gl = trainer.test_ood(id_data_loader, args.T)

    auroc_list_mcm, aupr_list_mcm, fpr_list_mcm = [], [], []
    auroc_list_gl, aupr_list_gl, fpr_list_gl = [], [], []

    for out_dataset in out_datasets:
        print(f"Evaluting OOD dataset {out_dataset}")
        ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess)
        out_score_mcm, out_score_gl = trainer.test_ood(ood_loader, args.T)

        print("MCM score")
        get_and_print_results(args, in_score_mcm, out_score_mcm,
                              auroc_list_mcm, aupr_list_mcm, fpr_list_mcm)

        print("GL-MCM score")
        get_and_print_results(args, in_score_gl, out_score_gl,
                              auroc_list_gl, aupr_list_gl, fpr_list_gl)

        plot_distribution(args, in_score_mcm, out_score_mcm, out_dataset, score='MCM')
        plot_distribution(args, in_score_gl, out_score_gl, out_dataset, score='GLMCM')

    print("MCM avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list_mcm), np.mean(auroc_list_mcm), np.mean(aupr_list_mcm)))
    print("GL-MCM avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list_gl), np.mean(auroc_list_gl), np.mean(aupr_list_gl)))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument('--in_dataset', default='imagenet', type=str,
                        choices=['imagenet'], help='in-distribution dataset')
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    # augment for LoCoOp
    parser.add_argument('--lambda_value', type=float, default=1,
                        help='temperature parameter')
    parser.add_argument('--topk', type=int, default=200,
                        help='topk')
    # augment for MCM and GL-MCM
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size')
    parser.add_argument('--T', type=float, default=1,
                        help='temperature parameter')
    args = parser.parse_args()
    main(args)
