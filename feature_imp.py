import os
import time
import sys
from copy import deepcopy

import tqdm
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy import io

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib as mpl
from pre_process import load_data


from dataset import FeatureImpDataset
from utils import to_one_hot, predict_prob
from models import ShallowFFNN, FeatureFusion, LearnedFeatureFusion

def permutation_importance(pt_model, data, data_loader, feature_mask, device, fusion, iters=30):
    # Get original predicted probabilities on test set
    y_prob = predict_prob(pt_model, data_loader, device, fusion=fusion)

    # Get original (base) AUC on test set
    base_err = roc_auc_score(data.y_test, y_prob)
    
    np.random.seed(0)
    importances = []
    for idx in feature_mask:
        imp_vals = []
        for _ in range(iters):
            if isinstance(idx, list):
                new_X = deepcopy(data.orig_meta_test)
               
                new_X[:, idx] = np.random.permutation(new_X[:, idx])  # permute
 
            else:
                # Permute across dim 0
                new_X = deepcopy(data.orig_meta_test)
                new_X[:, idx] = np.random.permutation(data.orig_meta_test[:, idx])

            # Change underlying meta_test to permuted version
            data.meta_test = new_X

            # Get predicted probabilities on permuted test set
            y_prob = predict_prob(pt_model, data_loader, device, fusion=fusion)

            # Recalculate AUC on permuted test set
            new_err = roc_auc_score(data.y_test, y_prob)
                
            # Calculate feature importance (% change in err)
            imp_vals.append((-(new_err - base_err) / base_err) * 100)
        importances.append(imp_vals)
        
    return importances


def main(args):
    mpl.use("Agg")
        
    # Set device for PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Set data file path
    test_dir = os.path.join(args.data_dir, "test")

    # Prepare test data loaders
    test_meta, test_x_cropped, test_y = load_data(test_dir)#[:sample_size]
    test_data = FeatureImpDataset(test_meta=test_meta, test_x_cropped=test_x_cropped, test_y=test_y)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)

    # Define model
    if args.model == "non-image-only":
        fusion = False
        model = ShallowFFNN(meta_features=test_data.meta_features).to(device)
        model_name = "shallow-only_1"
    elif args.model == "feature-fusion":
        fusion = True
        model = FeatureFusion(meta_features=test_data.meta_features, pre_trained=False, frozen=False).to(device)
        model_name = "feature-fusion_2"
    elif args.model == "learned-feature-fusion":
        fusion = True
        model = LearnedFeatureFusion(meta_features=test_data.meta_features, mode=args.fusion_mode, pre_trained=False, frozen=False).to(device)
        model_name = "hidden-feature-fusion-concat_3"
    else:
        sys.exit("Invalid model specified.")

    # Load model weights
    model.load_state_dict(torch.load(os.path.join(args.out_dir, args.model_name, args.model_name + ".pt"), map_location=device))
    model.eval()

    # Load meta feature names
    feature_names = []
    with open('dataset/meta_header.txt', 'r') as file:
        for line in file:
            feature_names.append(line.strip())  # strip() removes leading/trailing whitespace

    # Create "mask" to group dummy variables that represent a single feature and rename features accordingly
    feature_mask = list(range(6)) + [list(range(6, 14))] + [list(range(14, 19))] + [list(range(19, 29))] + [list(range(29, 34))]
    new_feature_names = feature_names[:6]
    new_feature_names.extend(['mass_shape', 'mass_margins', 'calc_type', 'calc_dist'])

    # Calculate permutation feature importances on test set
    test_importances = permutation_importance(pt_model=model,
                                              data=test_data,
                                              data_loader=test_loader,
                                              feature_mask=feature_mask,
                                              device=device,
                                              fusion=fusion,
                                              iters=args.iters)

    # Rank features by decreasing median importance (across 'iters' runs)
    ranking = np.argsort([np.median(x) for x in test_importances])[::-1]

    # Create box-and-whisker plot of (sorted) feature importance values by feature
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.boxplot([test_importances[i] for i in ranking], vert=False)
    ax.set_yticklabels([new_feature_names[i] for i in ranking])
    ax.set_ylabel("Shallow Feature")
    ax.set_xlabel("Percent Decrease in Test AUC")

    # Create data frame of all feature importance values to save
    imp_df = pd.DataFrame({"Importance": [imp for feature in test_importances for imp in feature],
                        "Feature": [feature for feature in new_feature_names for _ in range(args.iters)],
                        "Iteration": list(range(1, args.iters+1)) * len(new_feature_names)})

    # Save box-and-whisker plot and feature importance values with appropriate names
    fig.savefig(os.path.join(os.path.join(args.out_dir, args.model_name),'feature_imp'), bbox_inches="tight")
    imp_df.to_csv(os.path.join(os.path.join(args.out_dir, args.model_name),'feature_imp'), index=False)

if __name__ == "__main__":
    # Parse indication argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset", type=str,
                        help="path to processed data directory (output of preprocess.py)")
    parser.add_argument("--out_dir", default="Results", type=str,
                        help="path to directory where results and model weights are saved")
    parser.add_argument("--model", default="non-image-only", type=str, help="must be one of ['non-image-only', 'feature-fusion', 'learned-feature-fusion']")
    parser.add_argument("--model_name", type=str, help="name of model (e.g., name of saved weights file <model_name>.pt)")
    parser.add_argument("--fusion_mode", default="concat", help="fusion operation for LearnedFeatureFusion or Probability Fusion (one of ['concat', 'multiply', 'add'])")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size for testing")
    parser.add_argument("--iters", default=1, type=int, help="number of times to permute each feature")
    
    args = parser.parse_args()

    # Ensure "--model" argument is valid
    assert (args.model in ['non-image-only', 'feature-fusion', 'learned-feature-fusion']), "--model must be one of ['non-image-only', 'feature-fusion', 'learned-feature-fusion']"

    main(args)