from argparse import Namespace
from utils.my_utils import NLLSurvLoss, get_optim, get_split_loader, Monitor_CIndex, l1_reg_all
from utils.trainer import train_loop_survival
from utils.trainer import validate_survival
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os

def train_model(datasets: tuple, cur: int, args: Namespace):
    train_split, val_split = datasets
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))    
    loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    reg_fn = l1_reg_all
    writer = None

    if args.model_type =='snn':
        from models.model_snn import SNN
        model_dict = {'omic_input_dim': args.omic_input_dim, 'model_size_omic': "small", 'n_classes': args.n_classes}
        model = SNN(**model_dict)
    elif args.model_type == "clam_sb":
        from models.model_clam import CLAM_SB
        model_dict = {}
        model = CLAM_SB(**model_dict)
    elif args.model_type == 'clam_mb':
        from models.model_clam import CLAM_MB
        model_dict = {}
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'deepset':
        from models.model_deepset import MIL_Sum_FC_surv as deepset
        model_dict = {}
        model = deepset(**model_dict)
    elif args.model_type == 'attmil':
        from models.model_attmil import MIL_Attention_FC_surv as attmil
        model_dict = {}
        model = attmil(**model_dict)
    elif args.model_type == 'transmil':
        from models.model_transmil import TransMIL
        model_dict = {}
        model = TransMIL(**model_dict)
    elif args.model_type == 'gla':
        from models.LCM_net_utils.GLA.model_GLA import GLA
        model = GLA(
            gene_dim=60660,                   
            llama_path="/vip_media/sicheng/DataShare/Llama-2-7b-hf",
        ).cuda()
    elif args.model_type == 'lcmnet':
        from models.LCM_net_utils.MEI.model_MEI import MEI
        model = MEI(
            dim=256,
            Ng=4,     
            Np=4,    
            Nf=2,     
        )  

    model = model.to(device)
    optimizer = get_optim(model, args)

    train_loader = get_split_loader(train_split, training=True, testing = False, 
    weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = False, mode=args.mode, batch_size=args.batch_size)

    # if args.early_stopping:
    #     early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=20, verbose = True)
    # else:
    early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    latest_c_index = 0.
    max_c_index = 0.
    epoch_max_c_index = 0
    best_val_dict = {}

    print("running with {} {}".format(args.model_type, args.mode))

    for epoch in range(args.start_epoch, args.max_epochs):
        train_loop_survival(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc, args)
        val_latest, c_index_val, stop = validate_survival(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, args)
    
        if c_index_val > max_c_index:
            max_c_index = c_index_val
            epoch_max_c_index = epoch
            save_name = 's_{}_checkpoint'.format(cur)
            if args.load_model and os.path.isfile(os.path.join(
                args.results_dir, args.model_type, save_name+".pt".format(cur))):
                save_name+='_load'

            torch.save(model.state_dict(), os.path.join(
                args.results_dir, args.model_type, save_name+".pt".format(cur)))
            best_val_dict = val_latest

    print_results = {'result': (max_c_index, epoch_max_c_index)}
    print("================= summary of fold {} ====================".format(cur))
    print("result: {:.4f}".format(max_c_index))
    with open(os.path.join(args.results_dir, args.model_type, 'log.txt'), 'a') as f:
        f.write('result: {:.4f}, epoch: {}\n'.format(max_c_index, epoch_max_c_index))
    return best_val_dict, print_results