from argparser import TestArgParser
from models import create_model


if __name__ == '__main__':
    parser = TestArgParser()
    model = create_model(parser.parse_args())
    model.test()