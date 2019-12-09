from markup_service import init, get_markup
import argparse
from time import time


if __name__ == '__main__':
    # Configuration
    start_time = time()
    try:

        # init

        filepath = 'data/justice/33m-russian-courts-cases-by-suvorov/arb_sud.txt'
        with open(filepath) as fp:
            line = fp.readline()
            cnt = 1
            while line:
                print("Line {}: {}".format(cnt, line.strip()))
                line = fp.readline()
                cnt += 1

                if cnt > 10:
                    break

        print('Total time: {} mins'.format(round((time() - start_time) / 60, 2)))

    except Exception as e:
        print('Total time: {} mins'.format(round((time() - start_time) / 60, 2)))
        raise e