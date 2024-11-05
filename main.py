import pickle
import torch.optim as optim
from losses import DiceBCELoss
from utils import *
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

# Define Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, default='0')
parser.add_argument("--model_name", type=str, default='Brits')
parser.add_argument("--dataset", type=str, default='physionet')
parser.add_argument("--hours", type=int, default='48')
parser.add_argument("--epoch", type=int, default=300)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--weight_decay", type=float, default=0.00001)

parser.add_argument("--imputation_weight", type=float, default=0.3)
parser.add_argument("--classification_weight", type=float, default=1)
parser.add_argument("--consistency_weight", type=float, default=0.1)

parser.add_argument("--hiddens", type=int, default=108)
parser.add_argument("--channels", type=int, default=64)
parser.add_argument("--removal_percent", type=int, default=10)
parser.add_argument("--task", type=str, default='I')  # I: Imputation, C: Classification
parser.add_argument("--out_size", type=int, default=1)

parser.add_argument("--increase_factor", type=float, default=0.5)
parser.add_argument("--step_channels", type=int, default=512)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--model_path", type=str, default='./log')
parser.add_argument("--pre_model", type=str, default='.')

args, unknown = parser.parse_known_args()

# GPU Configuration
gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = config(args)
# Loading the kfold dataset
kfold_data = pickle.load(open(args.data_path, 'rb'))
kfold_label = pickle.load(open(args.label_path, 'rb'))

model_detail = '{}_remove_{}'.format(args.model_name, args.removal_percent)

date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))
rootdir = '%s/%s/%s/task_%s/%s' % (args.model_path,
                                    args.dataset,
                                    model_detail,
                                    args.task,
                                    date_str)
kfold_best = {}
for fold in range(5):
    # For logging purpose, create several directories
    dir = rootdir + '/fold_%d/' % (fold)

    if not os.path.exists(dir):
        os.makedirs(dir)
        os.makedirs(dir + 'tflog/')
        os.makedirs(dir + 'model_state/')

    # Text Logging
    f = open(dir + 'log.txt', 'a')
    writelog(f, '---------------')
    writelog(f, 'Model: %s' % model_detail)
    writelog(f, 'Hidden Units: %d' % args.hiddens)
    writelog(f, 'Times: %d' % args.times)
    writelog(f, 'Increase_factor: %s' % args.increase_factor)
    writelog(f, 'Step channels: %s' % args.step_channels)
    writelog(f, 'Task: %s' % args.task)
    writelog(f, 'Pre model: %s' % args.pre_model)
    writelog(f, '---------------')
    writelog(f, 'Dataset: %s' % args.dataset)
    writelog(f, 'Hours: %s' % args.hours)
    writelog(f, 'Removal: %s' % args.removal_percent)
    writelog(f, '---------------')
    writelog(f, 'Fold: %d' % fold)
    writelog(f, 'Learning Rate: %.5f' % args.lr)
    writelog(f, 'Batch Size: %d' % args.batchsize)
    writelog(f, 'Weight decay: %.5f' % args.weight_decay)
    writelog(f, 'Imputation Weight: %.3f' % args.imputation_weight)
    writelog(f, 'Consitency Loss Imputation Weight: %.3f' % args.consistency_weight)
    writelog(f, 'Classification Weight: %.3f' % args.classification_weight)
    writelog(f, '---------------')
    writelog(f, 'TRAINING LOG')

    # Process Defined Fold
    writelog(f, '---------------')
    writelog(f, 'FOLD ' + str(fold))

    # Tensorboard Logging
    tfw_train = SummaryWriter(log_dir=dir + 'tflog/train_')
    tfw_valid = SummaryWriter(log_dir=dir + 'tflog/valid_')
    tfw_test = SummaryWriter(log_dir=dir + 'tflog/test_')
    
    tfw = {'train': tfw_train,
           'valid': tfw_valid,
           'test': tfw_test}

    # Get dataset
    train_data = kfold_data[fold][0]
    train_label = kfold_label[fold][0]

    valid_data = kfold_data[fold][1]
    valid_label = kfold_label[fold][1]

    test_data = kfold_data[fold][2]
    test_label = kfold_label[fold][2]

    print('Unbalanced ratio of train_data: ', sum(train_label) / len(train_label))
    print('Unbalanced ratio of valid_data: ', sum(valid_label) / len(valid_label))
    print('Unbalanced ratio of test_data: ', sum(test_label) / len(test_label))

    # Normalization
    writelog(f, 'Normalization')
    train_data, mean_set, std_set, intervals = normalize(data=train_data, mean=[], std=[], compute_intervals=True)
    valid_data, m, s = normalize(valid_data, mean_set, std_set)
    test_data, m, s = normalize(test_data, mean_set, std_set)

    train_missing_rates = np.isnan(train_data).sum(axis=(0, 1)) / (train_data.shape[0] * train_data.shape[1])
    valid_missing_rates = np.isnan(valid_data).sum(axis=(0, 1)) / (valid_data.shape[0] * valid_data.shape[1])
    test_missing_rates = np.isnan(test_data).sum(axis=(0, 1)) / (test_data.shape[0] * test_data.shape[1])
    # Define Loaders        
    train_loader, train_replacement_probabilities = non_uniform_sample_loader_bidirectional(data=train_data, label=train_label, batch_size=args.batchsize, removal_percent=args.removal_percent, increase_factor=args.increase_factor)
    valid_loader, _ = non_uniform_sample_loader_bidirectional(valid_data, valid_label, args.batchsize, args.removal_percent, pre_replacement_probabilities=train_replacement_probabilities)
    test_loader, _ = non_uniform_sample_loader_bidirectional(test_data, test_label, args.batchsize, args.removal_percent, pre_replacement_probabilities=train_replacement_probabilities)


    dataloaders = {'train': train_loader,
                    'valid': valid_loader,
                    'test': test_loader}
    # Remove Data
    del train_data, train_label, valid_data, valid_label, test_data, test_label

    # Define Model
    criterion = DiceBCELoss().to(args.device)

    if args.model_name == 'Brits':
        from models import brits as net
    elif args.model_name == 'GRUD':
        from models import gru_d as net
    elif args.model_name == 'BVRIN':
        from models import bvrin as net
    elif args.model_name == 'MRNN':
        from models import m_rnn as net
    elif args.model_name == 'Brits_gru':
        from models import brits_gru as net        
    elif args.model_name == 'CSAI':
        from models import bcsai as net  

    if args.task == 'I':
        model = net(args=args, medians_df=intervals).to(args.device)
    elif args.task == 'C':
        model = net(args=args, medians_df=intervals, get_y=True).to(args.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writelog(f, 'Total params is {}'.format(total_params))

    # Define Optimizer
    optimizer = optim.Adam([
        {'params': list(model.parameters()), 'lr': args.lr, 'weight_decay': args.weight_decay}])

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * args.epoch, 0.75 * args.epoch], gamma=0.1)

    train = {
        'epoch': 0,
        'mae': 9999,
        'auc': 0,
    }

    valid = {
        'epoch': 0,
        'mae': 9999,
        'auc': 0,
    }
    
    test = {
        'epoch': 0,
        'mae': 9999,
        'auc': 0,
    }

    test_bets = {
        'epoch': 0,
        'mae': 0,
        'mre': 0,
        'acc': 0,
        'auc': 0,
        'prec_macro': 0,
        'recall_macro': 0,
        'f1_macro': 0,
        'bal_acc': 0,
    }

    test_bets_train = {
        'epoch': 0,
        'mae': 0,
        'mre': 0,
        'acc': 0,
        'auc': 0,
        'prec_macro': 0,
        'recall_macro': 0,
        'f1_macro': 0,
        'bal_acc': 0,
    }

    test_bets_valid = {
        'epoch': 0,
        'mae': 0,
        'mre': 0,
        'acc': 0,
        'auc': 0,
        'prec_macro': 0,
        'recall_macro': 0,
        'f1_macro': 0,
        'bal_acc': 0,
    }

    train_info = {'Loss':[], 'Loss_imputation':[], 'Loss_classification':[], 'Mae':[], 'Mre':[], 'Auc':[], 'prec_macro':[], 'recall_macro':[], 'f1_macro':[], 'bal_acc':[]}
    valid_info = {'Loss':[], 'Loss_imputation':[], 'Loss_classification':[], 'Mae':[], 'Mre':[], 'Auc':[], 'prec_macro':[], 'recall_macro':[], 'f1_macro':[], 'bal_acc':[]}
    test_info = {'Loss':[], 'Loss_imputation':[], 'Loss_classification':[], 'Mae':[], 'Mre':[], 'Auc':[], 'prec_macro':[], 'recall_macro':[], 'f1_macro':[], 'bal_acc':[]}

    # Training & Validation Loop
    for epoch in range(args.epoch):
        writelog(f, '------ Epoch ' + str(epoch))
        writelog(f, '-- Training')
        train_res = training(phase='train', model=model, optimizer=optimizer, criterion=criterion, args=args, data=dataloaders['train'], f=f, task=args.task, tfw=tfw, epoch=epoch)
        train_info['Loss'].append(train_res['loss'])
        train_info['Loss_imputation'].append(train_res['loss_imputation'])
        train_info['Mae'].append(train_res['mae'])
        train_info['Mre'].append(train_res['mre'])

        writelog(f, '-- Validation')
        valid_res = evaluate(phase='valid', model=model, criterion=criterion, args=args, data=dataloaders['valid'], f=f, task=args.task, tfw=tfw, epoch=epoch)
        valid_info['Loss'].append(valid_res['loss'])
        valid_info['Loss_imputation'].append(valid_res['loss_imputation'])
        valid_info['Mae'].append(valid_res['mae'])
        valid_info['Mre'].append(valid_res['mre'])

        writelog(f, '-- Testing')
        test_res = evaluate(phase='test', model=model, criterion=criterion, args=args, data=dataloaders['test'], f=f, task=args.task, tfw=tfw, epoch=epoch)
        test_info['Loss'].append(test_res['loss'])
        test_info['Loss_imputation'].append(test_res['loss_imputation'])
        test_info['Mae'].append(test_res['mae'])
        test_info['Mre'].append(test_res['mre'])
        
        # scheduler.step()
        
        if args.task == 'I':
            if train_res['mae'] < train['mae']:
                # torch.save(model, '%s/model/model_%d_best_train_%s.pt' % (dir, fold, args.task))
                torch.save(model.state_dict(), '%s/model_state/model_%d_best_train_%s_state_dict.pth' % (dir, fold, args.task))
                train['mae'] = train_res['mae']
                train['epoch'] = epoch
                test_bets_train['epoch'] = epoch
                test_bets_train['mae'] = test_res['mae']
                test_bets_train['mre'] = test_res['mre']
                get_polarfig(args, train_replacement_probabilities, train_missing_rates, train_res['feature_mae'], dir, fold, 'train', args.attributes)

            if valid_res['mae'] < valid['mae']:
                # torch.save(model, '%s/model/model_%d_best_valid_%s.pt' % (dir, fold, args.task))
                torch.save(model.state_dict(), '%s/model_state/model_%d_best_valid_%s_state_dict.pth' % (dir, fold, args.task))
                writelog(f, 'Best validation MAE is found! Training MAE : %f' % train_res['mae'])
                writelog(f, 'Best validation MAE is found! Validation MAE : %f' % valid_res['mae'])
                writelog(f, 'Best validation MAE is found! Testing MAE : %f' % test_res['mae'])
                writelog(f, 'Models at Epoch %d are saved!' % epoch)
                valid['mae'] = valid_res['mae']
                valid['epoch'] = epoch
                test_bets_valid['epoch'] = epoch
                test_bets_valid['mae'] = test_res['mae']
                test_bets_valid['mre'] = test_res['mre']
                get_polarfig(args, train_replacement_probabilities, valid_missing_rates, valid_res['feature_mae'], dir, fold, 'valid', args.attributes)

            if test_res['mae'] < test['mae']:
                # torch.save(model, '%s/model/model_%d_best_test_%s.pt' % (dir, fold, args.task))
                torch.save(model.state_dict(), '%s/model_state/model_%d_best_test_%s_state_dict.pth' % (dir, fold, args.task))
                test['mae'] = test_res['mae']
                test['epoch'] = epoch
                test_bets['epoch'] = epoch
                test_bets['mae'] = test_res['mae']
                test_bets['mre'] = test_res['mre']        
                get_polarfig(args, train_replacement_probabilities, test_missing_rates, test_res['feature_mae'], dir, fold, 'test', args.attributes)

        else:
            train_info['Loss_classification'].append(train_res['loss_classification'])
            train_info['Auc'].append(train_res['auc'])
            train_info['prec_macro'].append(train_res['prec_macro'])
            train_info['recall_macro'].append(train_res['recall_macro'])
            train_info['f1_macro'].append(train_res['f1_macro'])
            train_info['bal_acc'].append(train_res['bal_acc'])

            valid_info['Loss_classification'].append(valid_res['loss_classification'])
            valid_info['Auc'].append(valid_res['auc'])
            valid_info['prec_macro'].append(valid_res['prec_macro'])
            valid_info['recall_macro'].append(valid_res['recall_macro'])
            valid_info['f1_macro'].append(valid_res['f1_macro'])
            valid_info['bal_acc'].append(valid_res['bal_acc'])

            test_info['Loss_classification'].append(test_res['loss_classification'])
            test_info['Auc'].append(test_res['auc'])
            test_info['prec_macro'].append(test_res['prec_macro'])
            test_info['recall_macro'].append(test_res['recall_macro'])
            test_info['f1_macro'].append(test_res['f1_macro'])
            test_info['bal_acc'].append(test_res['bal_acc'])

            if train_res['auc'] > train['auc']:
                # torch.save(model, '%s/model/model_%d_best_train_%s.pt' % (dir, fold, args.task))
                torch.save(model.state_dict(), '%s/model_state/model_%d_best_train_%s_state_dict.pth' % (dir, fold, args.task))
                train['auc'] = train_res['auc']
                train['epoch'] = epoch
                test_bets_train['epoch'] = epoch
                test_bets_train['mae'] = test_res['mae']
                test_bets_train['mre'] = test_res['mre']
                test_bets_train['acc'] = test_res['accuracy']
                test_bets_train['auc'] = test_res['auc']
                test_bets_train['prec_macro'] = test_res['prec_macro']
                test_bets_train['recall_macro'] = test_res['recall_macro']
                test_bets_train['f1_macro'] = test_res['f1_macro']
                test_bets_train['bal_acc'] = test_res['bal_acc']
                get_polarfig(args, train_replacement_probabilities, train_missing_rates, train_res['feature_mae'], dir, fold, 'train', args.attributes)

            if valid_res['auc'] > valid['auc']:
                # torch.save(model, '%s/model/model_%d_best_valid_%s.pt' % (dir, fold, args.task))
                torch.save(model.state_dict(), '%s/model_state/model_%d_best_valid_%s_state_dict.pth' % (dir, fold, args.task))
                writelog(f, 'Best validation AUC is found! Training AUC : %f' % train_res['auc'])
                writelog(f, 'Best validation AUC is found! Validation AUC : %f' % valid_res['auc'])
                writelog(f, 'Best validation AUC is found! Testing AUC : %f' % test_res['auc'])
                writelog(f, 'Training MAE : %f' % train_res['mae'])
                writelog(f, 'Validation MAE : %f' % valid_res['mae'])
                writelog(f, 'Testing MAE : %f' % test_res['mae'])
                writelog(f, 'Models at Epoch %d are saved!' % epoch)
                valid['auc'] = valid_res['auc']
                valid['epoch'] = epoch
                test_bets_valid['epoch'] = epoch
                test_bets_valid['mae'] = test_res['mae']
                test_bets_valid['mre'] = test_res['mre']
                test_bets_valid['acc'] = test_res['accuracy']
                test_bets_valid['auc'] = test_res['auc']
                test_bets_valid['prec_macro'] = test_res['prec_macro']
                test_bets_valid['recall_macro'] = test_res['recall_macro']
                test_bets_valid['f1_macro'] = test_res['f1_macro']
                test_bets_valid['bal_acc'] = test_res['bal_acc']
                get_polarfig(args, train_replacement_probabilities, valid_missing_rates, valid_res['feature_mae'], dir, fold, 'valid', args.attributes)

            if test_res['auc'] > test['auc']:
                # torch.save(model, '%s/model/model_%d_best_test_%s.pt' % (dir, fold, args.task))
                torch.save(model.state_dict(), '%s/model_state/model_%d_best_test_%s_state_dict.pth' % (dir, fold, args.task))
                test['auc'] = test_res['auc']
                test['epoch'] = epoch
                test_bets['epoch'] = epoch
                test_bets['mae'] = test_res['mae']
                test_bets['mre'] = test_res['mre']
                test_bets['acc'] = test_res['accuracy']
                test_bets['auc'] = test_res['auc']
                test_bets['prec_macro'] = test_res['prec_macro']
                test_bets['recall_macro'] = test_res['recall_macro']
                test_bets['f1_macro'] = test_res['f1_macro']
                test_bets['bal_acc'] = test_res['bal_acc']
                get_polarfig(args, train_replacement_probabilities, test_missing_rates, test_res['feature_mae'], dir, fold, 'test', args.attributes)

    writelog(f, '-- Best Test for task' + args.task + ' of FOLD' + str(fold))
    for b in test_bets:
        writelog(f, '%s:%f' % (b, test_bets[b]))
    
    writelog(f, '-- Best Test train for task' + args.task + ' of FOLD' + str(fold))
    for b in test_bets_train:
        writelog(f, '%s:%f' % (b, test_bets_train[b]))

    writelog(f, '-- Best Test valid for task' + args.task + ' of FOLD' + str(fold))
    for b in test_bets_valid:
        writelog(f, '%s:%f' % (b, test_bets_valid[b]))

    kfold_best[str(fold)+'test_bets'] = test_bets
    kfold_best[str(fold)+'test_bets_train'] = test_bets_train
    kfold_best[str(fold)+'test_bets_valid'] = test_bets_valid

    writelog(f, 'END OF FOLD')
    f.close()

    #save all training recording
    training_record = {'train': train_info,
                    'valid': valid_info,
                    'test': test_info}
    pickle.dump(training_record, open('%s/training_recording_%d.pkl' % (dir, fold), 'wb'), -1)

    if args.task == 'I':
        fig, axes = plt.subplots(3, 1, figsize=(20, 20))
        axes[0].plot(train_info['Loss'], label='Training')
        axes[0].plot(valid_info['Loss'], label='Validation')
        axes[0].plot(test_info['Loss'], label='Testing')
        axes[0].legend()
        axes[0].title.set_text('Overall osses over epoches')

        axes[1].plot(train_info['Loss_imputation'], label='Training')
        axes[1].plot(valid_info['Loss_imputation'], label='Validation')
        axes[1].plot(test_info['Loss_imputation'], label='Testing')
        axes[1].legend()
        axes[1].title.set_text('Losses of imputation over epoches')

        axes[2].plot(train_info['Mae'], label='Training')
        axes[2].plot(valid_info['Mae'], label='Validation')
        axes[2].plot(test_info['Mae'], label='Testing')
        axes[2].legend()
        axes[2].title.set_text('MAEs over epoches')

        plt.savefig(dir+"/fold_{}_{}_performances.png".format(fold, args.task), dpi=500)
        
    else:
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))            
        axes[0, 0].plot(train_info['Loss'], label='Training')
        axes[0, 0].plot(valid_info['Loss'], label='Validation')
        axes[0, 0].plot(test_info['Loss'], label='Testing')
        axes[0, 0].legend()
        axes[0, 0].title.set_text('Overall osses over epoches')

        axes[0, 1].plot(train_info['Loss_imputation'], label='Training')
        axes[0, 1].plot(valid_info['Loss_imputation'], label='Validation')
        axes[0, 1].plot(test_info['Loss_imputation'], label='Testing')
        axes[0, 1].legend()
        axes[0, 1].title.set_text('Losses of imputation over epoches')

        axes[1, 0].plot(train_info['Loss_classification'], label='Training')
        axes[1, 0].plot(valid_info['Loss_classification'], label='Validation')
        axes[1, 0].plot(test_info['Loss_classification'], label='Testing')
        axes[1, 0].legend()
        axes[1, 0].title.set_text('Losses of classification over epoches')

        axes[1, 1].plot(train_info['Mae'], label='Training')
        axes[1, 1].plot(valid_info['Mae'], label='Validation')
        axes[1, 1].plot(test_info['Mae'], label='Testing')
        axes[1, 1].legend()
        axes[1, 1].title.set_text('MAEs over epoches')

        plt.savefig(dir+"/fold_{}_{}_imputation_performances.png".format(fold, args.task), dpi=500)

        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes[0, 0].plot(train_info['Auc'], label='auc')
        axes[0, 0].plot(train_info['prec_macro'], label='prec_macro')
        axes[0, 0].plot(train_info['recall_macro'], label='recall_macro')
        axes[0, 0].plot(train_info['f1_macro'], label='f1_macro')
        axes[0, 0].plot(train_info['bal_acc'], label='bal_acc')
        axes[0, 0].legend()
        axes[0, 0].title.set_text('Training performance over epoches')
        
        axes[0, 1].plot(valid_info['Auc'], label='auc')
        axes[0, 1].plot(valid_info['prec_macro'], label='prec_macro')
        axes[0, 1].plot(valid_info['recall_macro'], label='recall_macro')
        axes[0, 1].plot(valid_info['f1_macro'], label='f1_macro')
        axes[0, 1].plot(valid_info['bal_acc'], label='bal_acc')
        axes[0, 1].legend()
        axes[0, 1].title.set_text('Validation performance over epoches')

        axes[1, 0].plot(test_info['Auc'], label='auc')
        axes[1, 0].plot(test_info['prec_macro'], label='prec_macro')
        axes[1, 0].plot(test_info['recall_macro'], label='recall_macro')
        axes[1, 0].plot(test_info['f1_macro'], label='f1_macro')
        axes[1, 0].plot(test_info['bal_acc'], label='bal_acc')
        axes[1, 0].legend()
        axes[1, 0].title.set_text('Test performance over epoches')

        axes[1, 1].plot(train_info['Auc'], label='auc of training')
        axes[1, 1].plot(valid_info['Auc'], label='auc of validation')
        axes[1, 1].plot(test_info['Auc'], label='auc of test')
        axes[1, 1].legend()
        axes[1, 1].title.set_text('AUCs over epoches')

        plt.savefig(dir+"/fold_{}_{}_classification_performances.png".format(fold, args.task), dpi=500)

pickle.dump(kfold_best, open(rootdir + '/kfold_best.pkl', 'wb'), -1)
