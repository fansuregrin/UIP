from argparse import ArgumentParser
from models import _models


class BaseArgParser:
    def __init__(self):
        self.setup()
        self.modify()
    
    def setup(self):
        self.parser = ArgumentParser()
        self.parser.add_argument('--model_name', type=str, required=True, help='name of the model')
        self.parser.add_argument('--device', type=str, default='cuda',
                                 help='the device on which a tensor is or will be allocated')
        self.parser.add_argument('--ckpt_dir', type=str, default='checkpoints', 
                                 help='the directory saves the state of the network, optimizer, learning rate scheduler, etc')
        self.parser.add_argument('--quite', action='store_true', help='do not print message to stdout')
        self.parser.add_argument('--name', type=str, default='experiment', help='name of training process')
        self.parser.add_argument('--net_cfg', type=str, required=True, help='path to network config file')
        self.parser.add_argument('--ds_cfg', type=str, required=True, help='path to dataset config file')
        self.parser.add_argument('--batch_size', type=int, default=8, help='the batch size')
        self.parser.add_argument('--shuffle', action='store_true', help='shuffle the data at each epoch')
        self.parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
        self.parser.add_argument('--drop_last', action='store_true', help='drop the last incomplete batch, if the dataset size is not divisible by the batch size')
        self.parser.add_argument('--train_ds', type=str, default='train')
        self.parser.add_argument('--val_ds', type=str, default='val')
        self.parser.add_argument('--test_ds', type=str, default='test')

    def modify(self):
        args, _ = self.parser.parse_known_args()
        model = _models.get(args.model_name)
        self.parser = model.modify_args(self.parser, args.mode)

    def parse_args(self):
        return vars(self.parser.parse_args())


class TrainArgParser(BaseArgParser):
    def __init__(self):
        super().__init__()

    def setup(self):
        super().setup()
        self.parser.add_argument('--mode', type=str, default='train', choices=['train'])
        self.parser.add_argument('--lr_scheduler_cfg', type=str, default='configs/lr_scheduler/none.yaml')
        self.parser.add_argument('--start_epoch', type=int, default=0, help='which epoch to start from')
        self.parser.add_argument('--start_iteration', type=int, default=0, help='which iteration to start from')
        self.parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
        self.parser.add_argument('--seed', type=int, default=2023, help='lucky random seed')
        self.parser.add_argument('--load_prefix', type=str, default='weights', help='the prefix string of the filename of the weights to be loaded')
        self.parser.add_argument('--save_prefix', type=str, default='', help='the prefix string of the filename that needs to save the weights')
        self.parser.add_argument('--optimizer', type=str, default='adam', help='which optimizer to use')
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate for optimizer')
        self.parser.add_argument('--val_interval', type=int, default=100, help='how many iterations to validate the model')
        self.parser.add_argument('--ckpt_interval', type=int, default=5, help='how many epochs to save checkpoint')


class TestArgParser(BaseArgParser):
    def __init__(self):
        super().__init__()

    def setup(self):
        super().setup()
        self.parser.add_argument('--mode', type=str, default='test', choices=['test'])
        self.parser.add_argument('--test_name', type=str, help='name for test dataset')
        self.parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='path to checkpoint dir')
        self.parser.add_argument("--result_dir", type=str, default="results")
        self.parser.add_argument("--epochs", type=int, nargs='+', default=[99,], help="which epoch to load")
        self.parser.add_argument("--load_prefix", type=str, default='weights', help="the prefix string of the filename of the weights to be loaded")
