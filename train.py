from argparser import TrainArgParser
from models import create_model


if __name__ == '__main__':
    parser = TrainArgParser()
    model = create_model(parser.parse_args())
    model.train()