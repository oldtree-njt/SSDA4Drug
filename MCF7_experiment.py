import argparse
import time
import warnings
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import utils
import trainer
import model as m
import matplotlib
import random
import itertools

# Set a random number seed to facilitate experiment reproduction
seed = 42
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
# slower and more reproducible
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):
    epochs = args.epochs
    lr = args.lr
    dropout = args.dropout
    batch_size = args.batch_size

    preditor_hdims = args.predictor_h_dims.split(",")
    preditor_hdims = list(map(int, preditor_hdims))

    encoder_hdims = args.encoder_h_dims.split(",")
    encoder_hdims = list(map(int, encoder_hdims))

    para = "method_" + str(args.method) + "_gene_" + str(args.gene) + "_DAEdim_" + str(
        args.encoder_h_dims) + "_Predim_" + str(args.predictor_h_dims) + "_dropout_" + str(args.dropout) + "_lr_" + str(
        args.lr) + "_bs_" + str(args.batch_size) # (para)
    print("para:", para)
    if args.gene == "":
        gene = "all"
    elif args.gene == "_tp4k":
        gene = "tp4k"

    ####################################################################################################################
    # data loading
    ####################################################################################################################
    # Loading Source Data
    x_source = pd.read_csv("./experiment/MCF7/source_data" + "/source_scaled" + args.gene + ".csv",
                           index_col=0)
    # train
    x_train_source = pd.read_csv("./experiment/MCF7/source_data" + '/' + gene + "/X_train_source.csv",
                                 index_col=0)
    y_train_source = pd.read_csv("./experiment/MCF7/source_data" + '/' + gene + "/Y_train_source.csv",
                                 index_col=0)

    if args.sampling_method == "weight":
        y_source = pd.read_csv("./experiment/MCF7/source_data/source_meta_data.csv", index_col=0)
        from collections import Counter
        class_sample_count = np.array([Counter(y_source['response'])[0] / len(y_source['response']),
                                       Counter(y_source['response'])[1] / len(y_source['response'])])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train_source["response"].values])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.reshape(-1)
        from torch.utils.data.sampler import WeightedRandomSampler
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                        replacement=True)
    elif args.sampling_method == "smote":

        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline
        from imblearn.under_sampling import RandomUnderSampler
        over = SMOTE(sampling_strategy=0.5)
        under = RandomUnderSampler(sampling_strategy=0.5)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        x_train_source = x_train_source.T
        x_train_source, y_train_source = pipeline.fit_resample(x_train_source, y_train_source)
        x_train_source = x_train_source.T
        sampler = None

    source_train = utils.create_dataset(x=x_train_source, y=y_train_source, batch_size=batch_size, shuffle=False,
                                        sampler=sampler)

    # valid
    x_val_source = pd.read_csv("./experiment/MCF7/source_data" + '/' + gene + "/X_val_source.csv",
                               index_col=0)
    y_val_source = pd.read_csv("./experiment/MCF7/source_data" + '/' + gene + "/Y_val_source.csv",
                               index_col=0)
    source_valid = utils.create_dataset(x=x_val_source, y=y_val_source, batch_size=batch_size, shuffle=False)

    dataloader_source = {'train': source_train, 'val': source_valid}

    # Loading Unlabeled Target Data
    # train
    x_train_target = pd.read_csv(
        "./experiment/MCF7" + '/' + args.shot_method + '/' + gene + "/unlabeled_target_data" + '/' + "/X_train_target.csv",
        index_col=0)
    y_train_target = pd.read_csv(
        "./experiment/MCF7" + '/' + args.shot_method + '/' + gene + "/unlabeled_target_data" + '/' + "/Y_train_target.csv",
        index_col=0)
    target_train = utils.create_dataset(x=x_train_target, y=y_train_target, batch_size=batch_size, shuffle=True)

    # valid
    x_val_target = pd.read_csv(
        "./experiment/MCF7" + '/' + args.shot_method + '/' + gene + "/unlabeled_target_data" + '/' + "/X_val_target.csv",
        index_col=0)
    y_val_target = pd.read_csv(
        "./experiment/MCF7" + '/' + args.shot_method + '/' + gene + "/unlabeled_target_data" + '/' + "/Y_val_target.csv",
        index_col=0)
    target_valid = utils.create_dataset(x=x_val_target, y=y_val_target, batch_size=batch_size, shuffle=False)

    dataloader_unlabeled_target = {'train': target_train, 'val': target_valid}

    # Loading labeled Target Data
    # train
    x_train_target = pd.read_csv(
        "./experiment/MCF7" + '/' + args.shot_method + '/' + gene + "/labled_target_data" + '/' + "/X_train_target.csv",
        index_col=0)
    y_train_target = pd.read_csv(
        "./experiment/MCF7" + '/' + args.shot_method + '/' + gene + "/labled_target_data" + '/' + "/Y_train_target.csv",
        index_col=0)
    target_train = utils.create_dataset(x=x_train_target, y=y_train_target, batch_size=batch_size, shuffle=True)

    # valid
    x_val_target = pd.read_csv(
        "./experiment/MCF7" + '/' + args.shot_method + '/' + gene + "/labled_target_data" + '/' + "/X_val_target.csv",
        index_col=0)
    y_val_target = pd.read_csv(
        "./experiment/MCF7" + '/' + args.shot_method + '/' + gene + "/labled_target_data" + '/' + "/Y_val_target.csv",
        index_col=0)
    target_valid = utils.create_dataset(x=x_val_target, y=y_val_target, batch_size=batch_size, shuffle=False)

    dataloader_labeled_target = {'train': target_train, 'val': target_valid}

    # test
    X = pd.read_csv("./experiment/MCF7/target_data/target_scaled.csv", index_col=0)
    Y = pd.read_csv("./experiment/MCF7/target_data/target_meta_data.csv", index_col=0)
    test_data = utils.create_dataset(x=X, y=Y, batch_size=batch_size, shuffle=False)

    ####################################################################################################################
    # data loading finished
    ####################################################################################################################
    bulk_tasks = {'expression': 12863, 'pathway': 128}
    sc_tasks = {'expression': 12863, 'pathway': 128}
    dim_model_out = 2

    for path in [args.sc_all + args.drug, args.result + '/' + args.method]:
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")

    # target model path
    sc_all_path = args.sc_all + para
    result_path = args.result
    # Select the Training device
    if (args.device == "gpu"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    print("device:", device)


    encoder = m.DAE(input_dim=sc_tasks['pathway'], fc_dim=256,
                    AE_input_dim=bulk_tasks['expression'], AE_h_dims=encoder_hdims,
                    pretrained_weights=None,drop=dropout)
    encoder.to(device)

    # predictor
    predictor = m.Predictor(input_dim=sc_tasks["pathway"],
                            output_dim=32,
                            drop_out=dropout)
    predictor.to(device)

    Predictor_adentropy = m.Predictor_adentropy(num_class=dim_model_out, inc=32)
    Predictor_adentropy.to(device)

    fgm = m.FGM([encoder,predictor,Predictor_adentropy])

    loss_c = nn.CrossEntropyLoss()
    loss_e = nn.MSELoss()
    optimizer_d = torch.optim.Adagrad(
        itertools.chain(encoder.parameters(), predictor.parameters(), Predictor_adentropy.parameters()), lr=lr)

    encoder_f, predictor_f, Predictor_adentropy_f = trainer.train_semi_dae(fgm, encoder, predictor, Predictor_adentropy,
                                                                           dataloader_source,
                                                                           dataloader_unlabeled_target,
                                                                           dataloader_labeled_target,
                                                                           args.method,
                                                                           optimizer_d,
                                                                           loss_c,
                                                                           loss_e, epochs, start_epoch=0,
                                                                           save_path=sc_all_path + ".pkl",
                                                                           device=device, auc_path=result_path)
    torch.save({
        'encoder_state_dict': encoder_f.state_dict(),
        'predictor_state_dict': predictor_f.state_dict(),
        'Predictor_adentropy_state_dict': Predictor_adentropy_f.state_dict()
    }, sc_all_path + '.pth')

    if args.gene == "_tp4k":
        Gene = "tp4k"
    elif args.gene == "":
        Gene = "all"

    print("Sampling Method:", args.sampling_method)
    print("Gene:", Gene)
    print("Method:", args.method)
    print("Shot-Method:", args.shot_method)
    print("Drug:", args.drug)
    loss_class = nn.CrossEntropyLoss()
    print("Transfer " + args.method + " finished")
    test_model = m.Test_Double_Model(predictor=predictor_f, encoder=encoder_f, adentropy_p=Predictor_adentropy_f)
    test_predicted_t48_file = "./experiment/MCF7/MCF7.tsv"
    trainer.test_experiment(test_model, test_data, loss_class, device, test_predicted_t48_file, Y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--path', type=str, default='data/SCAD',
                        help='Path of the dataset used for model training')
    parser.add_argument('--drug', type=str, default='bortezomib',
                        help='choose drug,Can be PLX4720_451Lu,Gefitinib,Erlotinib,PLX4720,Vorinostat, Cetuximab,AR-42,and Etoposide.')
    parser.add_argument('--gene', type=str, default="",
                        help='choose data type. Can be all or _tp4k. Default: _tp4k')
    parser.add_argument('--sampling_method', type=str, default="weight",
                        help='choose sampling type. Can be weight or smote. Default: weight')
    parser.add_argument('--shot_method', type=str, default="3-shot",
                        help='choose shot type. Can be n-shot. Default: 3-shot')
    # save
    parser.add_argument('--result', type=str, default='experiment/MCF7/',
                        help='Path of the training result report files')
    parser.add_argument('--sc_all', type=str, default='experiment/MCF7/',
                        help='Path of the model in the sc level')

    # train
    parser.add_argument('--device', type=str, default="gpu",
                        help='Device to train the model. Can be cpu or gpu. Default: gpu')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of model training. Default: 1e-2')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epoches training. Default: 500')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of batch size when training. Default: 200')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout of neural network. Default: 0.3')
    parser.add_argument('--fix_source', type=int, default=0,
                        help='Fix the parameters in the bulk model. 0: do not freeze, 1: freeze. Default: 0')
    # model
    parser.add_argument('--method', type=str, default="adv",
                        help='choose model type. Can be DANN or CDAN. Default: DANN')
    parser.add_argument('--encoder_h_dims', type=str, default="512,256", help='Shape of the encoder. Each number represent the number of neuron in a layer. \
                                Layers are seperated by a comma. Default: 512,256')
    parser.add_argument('--bottleneck', type=int, default=256,
                        help='Size of the bottleneck layer of the model. Default: 128')
    parser.add_argument('--predictor_h_dims', type=str, default="64,32", help='Shape of the predictor. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 128')
    parser.add_argument('--load_source_model', type=int, default=0,
                        help='Load a trained bulk level or not. 0: do not load, 1: load. Default: 0')
    warnings.filterwarnings("ignore")

    args, unknown = parser.parse_known_args()
    matplotlib.use('Agg')

    main(args)