import os
import random
import warnings
import urllib
import zipfile
warnings.filterwarnings("ignore")
import numpy as np
import wandb
import torch
import constants as cst
import hydra
from config.config import Config
from run import run_wandb, run, sweep_init
from preprocessing.lobster import LOBSTERDataBuilder
from preprocessing.btc import BTCDataBuilder
from preprocessing.combined import CombinedDataBuilder
from constants import DatasetType

@hydra.main(config_path="config", config_name="config")
def hydra_app(config: Config):
    set_reproducibility(config.experiment.seed)
    print("Using device: ", cst.DEVICE)
    if cst.DEVICE == "cpu":
        accelerator = "cpu"
    else:
        accelerator = "gpu"
    if config.dataset.type == DatasetType.FI_2010:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 144
    elif config.dataset.type == DatasetType.BTC:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 40
    elif config.dataset.type == DatasetType.LOBSTER:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 46
    elif config.dataset.type == DatasetType.COMBINED:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 40  # Adjust based on dataset features
    
    if config.dataset.type.value == "LOBSTER" and not config.experiment.is_data_preprocessed:
        data_builder = LOBSTERDataBuilder(
            stocks=config.dataset.training_stocks,
            data_dir=cst.DATA_DIR,
            date_trading_days=config.dataset.dates,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.dataset.sampling_type,
            sampling_time=config.dataset.sampling_time,
            sampling_quantity=config.dataset.sampling_quantity,
        )
        data_builder.prepare_save_datasets()
        
    elif config.dataset.type.value == "FI_2010" and not config.experiment.is_data_preprocessed:
        try:
            dir = cst.DATA_DIR + "/FI_2010/"
            for filename in os.listdir(dir):
                if filename.endswith(".zip"):
                    filename = dir + filename
                    with zipfile.ZipFile(filename, 'r') as zip_ref:
                        zip_ref.extractall(dir)
            print("Data extracted.")
        except Exception as e:
            raise(f"Error downloading or extracting data: {e}")
        
    elif config.dataset.type == cst.DatasetType.BTC and not config.experiment.is_data_preprocessed:
        data_builder = BTCDataBuilder(
            data_dir=cst.DATA_DIR,
            date_trading_days=config.dataset.dates,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.dataset.sampling_type,
            sampling_time=config.dataset.sampling_time,
            sampling_quantity=config.dataset.sampling_quantity,
        )
        data_builder.prepare_save_datasets()
    
    elif config.dataset.type == cst.DatasetType.COMBINED and not config.experiment.is_data_preprocessed:
        data_builder = CombinedDataBuilder(
            data_dir=cst.DATA_DIR,
            date_trading_days=config.dataset.dates,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.dataset.sampling_type,
            sampling_time=config.dataset.sampling_time,
            sampling_quantity=config.dataset.sampling_quantity,
        )
        data_builder.prepare_save_datasets()
        
    elif config.dataset.type == cst.DatasetType.SBI and not config.experiment.is_data_preprocessed:
        # For SBI dataset, copy from Colab content directory to data/SBI directory
        sbi_data_dir = cst.DATA_DIR + "/SBI"
        if not os.path.exists(sbi_data_dir):
            os.makedirs(sbi_data_dir)
        
        # Source file in Colab content directory
        source_file = "content/combined_output_week_20.csv"
        target_file = os.path.join(sbi_data_dir, "sbi_data.csv")
        
        try:
            # Copy the file from Colab content to our data directory
            import shutil
            shutil.copy2(source_file, target_file)
            print(f"SBI dataset copied from {source_file} to {target_file}")
        except FileNotFoundError:
            print(f"Warning: Could not find {source_file}")
            print("Please ensure the SBI dataset file is available at content/combined_output_week_20.csv")
            print(f"Or manually place sbi_data.csv in the {sbi_data_dir} directory")
        except Exception as e:
            print(f"Error copying SBI dataset: {e}")
            print(f"Please manually copy the file to {target_file}")

    if config.experiment.is_wandb:
        if config.experiment.is_sweep:
            sweep_config = sweep_init(config)
            sweep_id = wandb.sweep(sweep_config, project=cst.PROJECT_NAME, entity="")
            wandb.agent(sweep_id, run_wandb(config, accelerator), count=sweep_config["run_cap"])
        else:
            start_wandb = run_wandb(config, accelerator)
            start_wandb()
    else:
        run(config, accelerator)

def set_reproducibility(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_torch():
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    set_torch()
    hydra_app()