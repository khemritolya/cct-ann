import argparse
import warnings
from copy import deepcopy
from pathlib import Path

import torch

from pipeline.callbacks import logger
from pipeline.pipeline import train_loop, weight_fn
from utils.parseconfigs import callback_dict, preprocess, stopping_criterion_dict
from utils.postprocessing import get_filename, save_results

warnings.simplefilter("ignore", FutureWarning)

CONFIG_PATH = Path("configs")
OUTPUT_PATH = Path("checkpts")


def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--config", type=str, default="mc1")
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--bootstrap-seed", type=int, default=-1)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--no-logger", action="store_true")
    parser.add_argument("--se", action="store_true")
    parser.add_argument("--exp-bootstrap", action="store_true")
    parser.add_argument("--no-config-save", action="store_true")
    return parser


def main(parser, provided_args=None, return_locals=False, identity_only=False):
    # --------------------------- Preprocess ---------------------------
    config_lst, dgp_objects, model_objects = preprocess(
        parser, configpath=CONFIG_PATH, provided_args=provided_args
    )
    (
        args,
        config,
        dgp_args,
        model_config,
        train_step_kwargs,
        optimizer_config,
    ) = config_lst

    dgp, data, corr_mat = dgp_objects
    npvec, torchvec, dataset, loader = data
    model, optimizer_constructor, optimizer = model_objects
    torch.manual_seed(config["seed"])

    # --------------------------- Inefficient estimate ---------------------------
    cache_df = train_loop(
        model,
        optimizer,
        loader,
        inverse_design_instrument=torchvec["inverse_design_instrument"],
        max_epoch=config["train_max_epoch"],
        min_epochs=config["train_min_epoch"],
        stopping_kwargs=dict(
            param_tol=config["train_stopping_param_tol"],
            grad_tol=config["train_stopping_grad_tol"],
        ),
        history=config["train_stopping_history_length"],
        print_freq=config["train_callback_freq"],
        callback=callback_dict[config["callback"]],
        stopping_criterion=stopping_criterion_dict[config["stopping_criterion"]],
        train_step_kwargs=train_step_kwargs,
        name=f"{config['model_name']}_inefficient",
    )

    inefficient_parameter_estimate = model.get_parameter_of_interest(
        torchvec["endogenous"]
    )
    inefficient_model = deepcopy(model.state_dict())
    # ---- Diagnostic: plot estimated vs true structural function ----
    import numpy as np
    import matplotlib.pyplot as plt
 
    _endo = npvec["endogenous"]  # shape [n, d]; col 0 is y2 (feature of interest)
 
    # Grid: y2 runs over its 5th–95th percentile; all other features fixed at mean
    _y2_grid = np.linspace(
        np.percentile(_endo[:, 0], 5),
        np.percentile(_endo[:, 0], 95),
        200,
    )
    _other_means = _endo[:, 1:].mean(axis=0)  # mean of every other feature
 
    # Build input matrix: [y2_grid, mean_y3, mean_x2, mean_x_highdim...]
    _grid_endo = np.column_stack([
        _y2_grid,
        np.tile(_other_means, (len(_y2_grid), 1)),
    ])
 
    # Estimated function and its derivative w.r.t. y2 (index 0) via autograd
    _grid_tensor = torch.tensor(_grid_endo, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        _y_hat = model(_grid_tensor).numpy().flatten()
    # Derivative: d(model)/d(y2) at each grid point
    _grid_tensor_grad = torch.tensor(_grid_endo, dtype=torch.float32, requires_grad=False)
    _probe = torch.zeros(len(_y2_grid), 1, requires_grad=True)
    _grid_with_probe = torch.cat([_grid_tensor[:, [0]] + _probe, _grid_tensor[:, 1:]], dim=1)
    model(_grid_with_probe).sum().backward()
    _dy_hat = _probe.grad.numpy().flatten()
    model.train()
 
    # True structural function: y1 = y2 + h01(y3_mean) + h02(x2_mean) + complex_func(x_hd_mean)
    # True derivative w.r.t. y2 is exactly 1 everywhere (linear in y2 in mc2).
    _h01 = dgp.h01  # sigmoid
    _h02 = dgp.h02  # log(1+x)
    _y3_mean = _other_means[0]
    _x2_mean = _other_means[1]
    _true_base = np.cos(_y2_grid) + _h01(_y3_mean) + _h02(_x2_mean)
    if hasattr(dgp, "complex_func") and dgp.high_dim_relevant and _endo.shape[1] > 3:
        _xhd_mean = _other_means[2:]
        _true_base += dgp.complex_func(_xhd_mean[None, :]).item()
    _true_deriv = -np.sin(_y2_grid)  # d/d(y2) of cos(y2)
 
    # Level-shift true function to share midpoint with model estimate
    _mid = len(_y2_grid) // 2
    _true_shifted = _true_base - _true_base[_mid] + _y_hat[_mid]
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
 
    ax1.plot(_y2_grid, _true_shifted, label="True $h(y_2, \\bar{\\cdot})$ (level-shifted)", lw=2)
    ax1.plot(_y2_grid, _y_hat, label="Estimated $\\hat{h}(y_2, \\bar{\\cdot})$", lw=2, ls="--")
    ax1.set_xlabel("$y_2$ (feature of interest)")
    ax1.set_ylabel("$y_1$")
    ax1.set_title("Structural function")
    ax1.legend()
 
    ax2.plot(_y2_grid, _true_deriv, label="True $\\partial h / \\partial y_2 = -\\sin(y_2)$", lw=2)
    ax2.plot(_y2_grid, _dy_hat, label="Estimated $\\partial \\hat{h} / \\partial y_2$", lw=2, ls="--")
    ax2.axhline(inefficient_parameter_estimate, color="grey", ls=":", lw=1.5,
                label=f"Avg derivative = {inefficient_parameter_estimate:.3f}")
    ax2.set_xlabel("$y_2$ (feature of interest)")
    ax2.set_ylabel("Derivative")
    ax2.set_title("Derivative w.r.t. $y_2$")
    ax2.legend()
 
    fig.suptitle(
        f"Dataset: {config.get('dataset', '?')} — other features fixed at sample mean (inefficient estimate)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("diagnostic_structural_slice.png", dpi=120)
    plt.show()
    print("Diagnostic plot saved to diagnostic_structural_slice.png")
    
    _test_y2 = np.array([0.0, np.pi/2, np.pi])
    _test_endo = np.column_stack([_test_y2, np.tile(_other_means, (3, 1))])
    print("True y1 values at y2=0, pi/2, pi:", 
      _test_y2 + dgp.h01(_other_means[0]) + dgp.h02(_other_means[1]))
    
    breakpoint()
    # ---- end diagnostic ----
 

    if not args.no_tqdm:
        print(f"initial estimate = {inefficient_parameter_estimate}")

    inefficient_derivative = None
    inefficient_prediction = None
    corrected_identity_weighting = None

    if hasattr(model, "_forward_filter_residuals"):
        corrected_identity_weighting = model._forward_filter_residuals(
            torchvec["endogenous"],
            torchvec["response"],
            torchvec["inverse_design_instrument"],
            torchvec["transformed_instrument"],
        )

    if hasattr(model, "get_parameter_of_interest_with_correction"):
        inefficient_prediction = model(torchvec["endogenous"]).detach()
        inefficient_derivative = model.get_derivatives(torchvec["endogenous"]).detach()

    if identity_only and return_locals:
        results = {
            "identity_weighting": inefficient_parameter_estimate,
            "initial_loss": cache_df["loss"].iloc[-1],
            "corrected_identity_weighting": corrected_identity_weighting,
        }
        return locals()

    # --------------------------- Efficiency weighting ---------------------------
    weights = weight_fn(
        prediction=model(torchvec["endogenous"]),
        truth=torchvec["response"],
        basis=torchvec["instrument"],
        n_neighbors=5,
    )
    torchvec["weights"] = weights
    dataset_with_weights, loader_with_weights = dgp.package_dataset(torchvec)

    # Refresh the optimizer
    optimizer = optimizer_constructor(model.parameters(), **optimizer_config)

    # Train again
    cache_df_eff = train_loop(
        model,
        optimizer,
        loader_with_weights,
        inverse_design_instrument=torchvec["inverse_design_instrument"],
        max_epoch=config["train_max_epoch"],
        min_epochs=config["train_min_epoch"],
        stopping_kwargs=dict(
            param_tol=config["train_stopping_param_tol"],
            grad_tol=config["train_stopping_grad_tol"],
        ),
        history=config["train_stopping_history_length"],
        print_freq=config["train_callback_freq"],
        callback=callback_dict[config["callback"]],
        stopping_criterion=stopping_criterion_dict[config["stopping_criterion"]],
        has_weights=True,
        train_step_kwargs=train_step_kwargs,
        name=f"{config['model_name']}_efficient",
    )
    efficient_parameter_estimate = model.get_parameter_of_interest(
        torchvec["endogenous"]
    )
    efficient_model = deepcopy(model.state_dict())

    param_with_correction = None
    param_se = None
    if hasattr(model, "get_parameter_of_interest_with_correction"):
        param_with_correction = model.get_parameter_of_interest_with_correction(
            endogenous=torchvec["endogenous"],
            response=torchvec["response"],
            inefficient_derivative=inefficient_derivative,
            inefficient_prediction=inefficient_prediction,
            weights=weights,
            basis=torchvec["transformed_instrument"],
            inverse_design=torchvec["inverse_design_instrument"],
            return_standard_error=args.se,
        )
        if not args.no_tqdm:
            print(f"final estimate = {param_with_correction}")
        if args.se:
            param_with_correction, param_se = param_with_correction

    else:
        if not args.no_tqdm:
            print(f"final estimate = {efficient_parameter_estimate}")

    # --------------------------- Post-processing ---------------------------
    results = {
        "optimal_weighting_uncorrected": efficient_parameter_estimate,
        "identity_weighting": inefficient_parameter_estimate,
        "initial_loss": cache_df["loss"].iloc[-1],
        "final_loss": cache_df_eff["loss"].iloc[-1],
        "corrected_identity_weighting": corrected_identity_weighting,
    }

    if hasattr(model, "get_parameter_of_interest_with_correction"):
        results["optimal_weighting"] = param_with_correction
        # if args.se:
        #     results["efficient_parameter_se"] = param_se
    else:
        results["optimal_weighting"] = results["optimal_weighting_uncorrected"]

    if not args.no_save:
        save_results_kwargs = {}
        if args.save_weights:
            save_results_kwargs["efficient_model"] = efficient_model
            save_results_kwargs["inefficient_model"] = inefficient_model

        save_results(
            OUTPUT_PATH,
            get_filename(config, args.name),
            config_lst,
            results,
            logger if not args.no_logger else None,
            no_config_save=args.no_config_save,
            **save_results_kwargs,
        )

    if return_locals:
        return locals()

    return results


if __name__ == "__main__":
    parser = generate_parser()
    main(parser)
