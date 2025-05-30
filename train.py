import os
from shutil import rmtree
import sys

import argparse
import tqdm
import matplotlib as mpl
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import DataLoader

from utils import *
from models import *
from dataset import *
from pre_process import load_data

def main(args):
    """Train model and evaluate on test set."""
    print(args) 

    # Load data from the train, val, and test folders
    train_meta, train_x_cropped, train_y = load_data(os.path.join(args.data_dir, "train"))
    val_meta, val_x_cropped, val_y = load_data(os.path.join(args.data_dir, "val"))
    test_meta, test_x_cropped, test_y = load_data(os.path.join(args.data_dir, "test"))

    # Set random seeds for reproducibility
    set_seed(args.seed)

    # Set device for PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Get train and val data
    # Create custom datasets for training and evaluation
    train_data = BreastMRIFusionDataset(meta_data=train_meta, x_cropped_data=train_x_cropped, y_data=train_y,augment=args.augment)
    val_data = BreastMRIFusionDataset(meta_data=val_meta, x_cropped_data=val_x_cropped, y_data=val_y, augment=args.augment)
    test_data = BreastMRIFusionDataset(meta_data=test_meta, x_cropped_data=test_x_cropped, y_data=test_y, augment=args.augment)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, num_workers=0)
    val_loader   = DataLoader(val_data, shuffle=False, batch_size=args.batch_size, num_workers=0)
    test_loader  = DataLoader(test_data, shuffle=False, batch_size=args.batch_size, num_workers=0)

    # Compute class weights
    if args.use_class_weights:
        n_train = len([f for f in os.listdir(os.path.join(args.data_dir, "train")) if "_y" in f])
        y_train = np.array([np.load(os.path.join(args.data_dir, "train", str(i+1) + "_y.npy")).item() for i in tqdm.tqdm(range(n_train), desc="Getting class weights")])
        class_weights = torch.Tensor(compute_class_weight('balanced', np.unique(y_train), y_train)).to(device)
    else:
        class_weights = torch.Tensor([1, 1]).to(device)

    # Define model
    if args.model == "image-only":
        model = ResNet50(pre_trained=args.pretrained, frozen=False).to(device)
        fusion, meta_only = False, False
    
    elif args.model == "non-image-only":
        model = ShallowFFNN(meta_features=train_data.meta_features).to(device)
        fusion, meta_only = False, True
    
    elif args.model == "feature-fusion":
        model = FeatureFusion(meta_features=train_data.meta_features, pre_trained=args.pretrained, frozen=False).to(device)
        fusion, meta_only = True, False

    elif args.model == "learned-feature-fusion":
        if args.train_mode == "default":
            model = LearnedFeatureFusion(meta_features=train_data.meta_features, mode=args.fusion_mode, pre_trained=args.pretrained, frozen=False).to(device)
        elif args.train_mode == "multiloss" or args.train_mode == "multiopt":
            model = LearnedFeatureFusionVariant(meta_features=train_data.meta_features, mode=args.fusion_mode, pre_trained=args.pretrained, frozen=False).to(device)
        else:
            sys.exit("Invalid train_mode specified")
        fusion, meta_only = True, False

    else:
        sys.exit("Invalid model specified.")

    print(model)
    print("# params:", np.sum([p.numel() for p in model.parameters() if p.requires_grad]))
    print("Class weights:", class_weights)
    print("Positive class weight:", class_weights[1] / class_weights[0])

    # Choose proper train and evaluation functions based on optimization approach
    if args.train_mode == "default":
        train_fxn = train
        eval_fxn = evaluate
    elif args.train_mode == "multiloss":
        train_fxn = train_multiloss
        eval_fxn = evaluate_multiloss
    elif args.train_mode == "multiopt":
        train_fxn = train_multiopt
        eval_fxn = evaluate

    # Train
    model, history = train_fxn(model=model,
                               train_loader=train_loader,
                               val_loader=val_loader,
                               max_epochs=args.max_epochs,
                               optim=torch.optim.Adam(model.parameters(), lr=1e-4),
                               class_weights=class_weights,
                               early_stopping={"metric": "val_auc_roc", "mode": "max", "patience": args.patience},
                               device=device,
                               label_smoothing=args.label_smoothing,
                               fusion=fusion,
                               meta_only=meta_only)

    # Set model output save directory
    MODEL_NAME = f"{args.model}"
    if args.model == "learned-feature-fusion":
        MODEL_NAME += f"-{args.fusion_mode}"
    if not args.use_class_weights:
        MODEL_NAME += "_no-CW"
    if args.augment:
        MODEL_NAME += "_aug"
    if args.max_epochs != 100:
        MODEL_NAME += f"_ep{args.max_epochs}"
    if args.patience != 5:
        MODEL_NAME += f"_pat{args.patience}"
    if args.label_smoothing != 0:
        MODEL_NAME += f"_ls{args.label_smoothing}"
    if args.train_mode != "default":
        MODEL_NAME += f"_{args.train_mode}"
    if args.pretrained != False:
        MODEL_NAME += f"_pretrained"
    if args.seed != 0:
        MODEL_NAME += f"_seed{args.seed}"
    save_dir = os.path.join(args.out_dir, MODEL_NAME)

    # Create output directories
    if os.path.isdir(save_dir):
        rmtree(save_dir)
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir, "train_history"))
    os.mkdir(os.path.join(save_dir, "summary_plots"))

    # Save best model weights and full training history
    history.to_csv(os.path.join(save_dir, "train_history.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(save_dir, MODEL_NAME + ".pt"))

    # Evaluate on test set and save summary outplot/plots
    set_seed(0)  # reset random seeds in case TTA used
    mpl.use("Agg")  # use Agg matplotlib backend
    pred_df, loss_fig, acc_fig, auc_roc_fig, cm_fig, roc_fig, summary = eval_fxn(
        model=model,
        data_loader=test_loader,
        history_df=history,
        device=device,
        fusion=fusion,
        meta_only=meta_only
    )
    summary += "\n" + repr(args) + "\n\n"
    summary += f"Class weights: {class_weights}\n"
    summary += f"Positive class weight: {class_weights[1] / class_weights[0]}\n"

    # Save test set predictions to csv
    pred_df.to_csv(os.path.join(save_dir, "preds.csv"), index=False)

    # Save summary output text
    f = open(os.path.join(save_dir, "summary.txt"), "w")
    f.write(summary)
    f.close()

    # Save other figures
    loss_fig.savefig(os.path.join(save_dir, "train_history", "train_loss.pdf"), bbox_inches="tight")
    acc_fig.savefig(os.path.join(save_dir, "train_history", "train_acc.pdf"), bbox_inches="tight")
    auc_roc_fig.savefig(os.path.join(save_dir, "train_history", "train_auc_roc.pdf"), bbox_inches="tight")

    cm_fig.savefig(os.path.join(save_dir, "summary_plots", "confusion_matrix.pdf"), bbox_inches="tight")
    roc_fig.savefig(os.path.join(save_dir, "summary_plots", "roc_curve.pdf"), bbox_inches="tight")

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset", type=str,
                        help="path to processed data directory (output of preprocess.py)")
    parser.add_argument("--out_dir", default="Results", type=str,
                        help="path to directory where results and model weights will be saved")        
    parser.add_argument("--model", default="image-only", type=str,
                        help="must be one of ['image-only', 'shallow-only', 'feature-fusion']")
    parser.add_argument("--train_mode", default="default", type=str,
                        help="approach to optimizing fusion model (one of ['default', 'multiloss', 'multiopt']")
    parser.add_argument("--fusion_mode", default="concat", help="fusion type for LearnedFeatureFusion or ProbabilityFusion (one of ['concat', 'multiply', 'add'])")
    parser.add_argument("--max_epochs", default=100, type=int, help="maximum number of epochs to train")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size for training, validation, and testing (will be lowered if TTA used)")
    parser.add_argument("--patience", default=5, type=int, help="early stopping 'patience' during training")
    parser.add_argument("--use_class_weights", default=False, action="store_true", help="whether or not to use class weights applied to loss during training")
    parser.add_argument("--augment", default=True, action="store_true", help="whether or not to use augmentation during training")
    parser.add_argument("--pretrained", default=False, action="store_true", help="whether or not to use ImageNet weight initialization for ResNet backbone")
    parser.add_argument("--label_smoothing", default=0, type=float, help="ratio of label smoothing to use during training")
    parser.add_argument("--seed", default=0, type=int, help="set random seed")

    args = parser.parse_args()

    # Ensure "--model" argument is valid
    assert (args.model in ['image-only', 'non-image-only', 'feature-fusion', 'learned-feature-fusion']), "--model must be one of ['image-only', 'non-image-only', 'feature-fusion', 'learned-feature-fusion']"

    # Ensure "--train_mode" argument is valid
    assert (args.train_mode) in ['default', 'multiloss', 'multiopt'], "--train_mode must be one of ['default', 'multiloss', 'multiopt']"

    main(args)