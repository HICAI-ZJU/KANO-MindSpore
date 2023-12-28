import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  
from argparse import Namespace
from logging import Logger
import chemprop_ms.train.run_pretrain as training
from chemprop_ms.parsing import parse_train_args, modify_train_args
from chemprop_ms.torchlight import initialize_exp
#from rich import pretty
#pretty.install()


def pretrain(args: Namespace, logger: Logger = None):
    training.pre_training(args, logger)



if __name__ == '__main__':
    args = parse_train_args()
    args.data_path = './data/zinc15_250K.csv'
    args.gpu = 0
    args.epochs = 50
    args.batch_size = 1024
    modify_train_args(args)
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))
    pretrain(args, logger)
