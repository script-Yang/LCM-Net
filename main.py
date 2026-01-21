import argparse
from utils.my_utils import seed_torch
from utils.my_core import train_model
from timeit import default_timer as timer
import numpy as np
import os

parser = argparse.ArgumentParser(
    description='Configurations for Survival Analysis on TCGA Data.')

from dataset.dataset_survival import Generic_MIL_Survival_Dataset
dataset_name = 'CESC'

parser.add_argument('--model_type', type=str, default='lcmnet', 
choices=['snn', 'clam_sb', 'clam_mb','deepset','attmil','transmil',"gla", "lcmnet"]
,help='Type of model (Default: motcat)')

parser.add_argument('--results_dir',     type=str, default=f'./results/{dataset_name}',
                    help='Results directory (Default: ./results)')

parser.add_argument('--mode', default='coattn', type=str, choices=['omic', 'path', 'coattn'],
                    help='Specifies which modalities to use / collate function in dataloader.')

parser.add_argument('--split_dir',       type=str, default='/vip_media/sicheng/DataShare/tmi_re/ours/splits/5foldcv',
                    help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca)')

parser.add_argument('--data_root_dir',   type=str, default=f'/vip_media/sicheng/DataShare/tmi_re/UNI_results/UNI_{dataset_name}/pt_files',
                    help='Data directory to WSI features')
parser.add_argument('--seed', 			 type=int, default=1,
                    help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', 			     type=int, default=5,
                    help='Number of folds (default: 5)')
parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--n_classes',      type=float, default=4)
parser.add_argument('--batch_size',      type=int, default=1,
                    help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--weighted_sample', action='store_true',
                    default=True, help='Enable weighted sampling')

parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--lr',type=float, default=2e-4, help='Learning rate (default: 0.0002)')
parser.add_argument('--reg', type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--lambda_reg',type=float, default=1e-4,help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--gc',type=int, default=32, help='Gradient Accumulation Step.')


parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch.')
parser.add_argument('--max_epochs', type=int, default=20,help='Maximum number of epochs to train (default: 20)')

parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')

parser.add_argument('--load_model', action='store_true', default=False, help='whether to load model')

args = parser.parse_args()
# args = get_custom_exp_code(args)

dataset = Generic_MIL_Survival_Dataset(csv_path=f"/vip_media/sicheng/DataShare/tmi_re/Gene_data_parquet/{dataset_name}_data.parquet",
                                        mode=args.mode,
                                        data_dir=args.data_root_dir,
                                        shuffle=False,
                                        seed=args.seed,
                                        print_info=True,
                                        patient_strat=False,
                                        n_bins=4,
                                        label_col = 'survival_days',
                                        ignore=[])

omic_size = 10110
def main(args):
    os.makedirs(args.results_dir,exist_ok=True)
    os.makedirs(os.path.join(args.results_dir,args.model_type),exist_ok=True)
    start = 0
    end = args.k
    folds = np.arange(start, end)
    for i in folds:
        start_t = timer()
        seed_torch(args.seed)
        split_path = '{}/{}/splits_{}.csv'.format(args.split_dir, dataset_name, i)
        print(f"Using split path: {split_path}")
        train_dataset, val_dataset = dataset.return_splits(from_id=False,
        csv_path='{}/{}/splits_{}.csv'.format(args.split_dir, dataset_name, i))
        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        datasets = (train_dataset, val_dataset)

        if 'omic' or 'coattn' in args.mode:
            args.omic_input_dim = train_dataset.genomic_features.shape[1]
            args.omic_sizes = [omic_size,omic_size,omic_size,omic_size,omic_size]

        summary_results, print_results = train_model(datasets, i, args) 
        
        end_t = timer()
        print('Fold %d Time: %f seconds' % (i, end_t - start_t))
        print('ok')


if __name__ == "__main__":
    #torch.multiprocessing.set_start_method('spawn')
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))