from markup_service import init, get_markup
import argparse
from time import time


if __name__ == '__main__':
    # Configuration

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default.config')
    parser.add_argument('--stage', type=str, default='predict_classifier')
    args = parser.parse_args()

    start_time = time()
    try:

        # init

        START_STAGE = 0
        if args.stage == "train_word2vec":
            START_STAGE = 0
        elif args.stage == "clusterize_vocab":
            START_STAGE = 1
        elif args.stage == "train_classifier":
            START_STAGE = 2
        elif args.stage == "test_classifier":
            START_STAGE = 3
        elif args.stage == "predict_classifier":
            START_STAGE = 4
        else:
            raise Exception('Unknown start stage', 'Check start stage value')

        init(args.config, START_STAGE)

        # run

        get_markup()

        print('Total time: {} mins'.format(round((time() - start_time) / 60, 2)))

    except Exception as e:
        print('Total time: {} mins'.format(round((time() - start_time) / 60, 2)))
        raise e