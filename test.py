from time import time
import gzip


if __name__ == '__main__':
    # Configuration
    start_time = time()
    try:

        filepath = 'data/justice/33m-russian-courts-cases-by-suvorov/arb_sud/vysshij-arbitrazhnyj-sud-rf-40001/2013/306465818.xml.gz'
        with gzip.open(filepath, 'rt') as f:
            cnt = 1
            for line in f:
                print("Line {}: {}".format(cnt, line.strip()))
                cnt += 1

                if cnt > 10:
                    break

        print('Total time: {} mins'.format(round((time() - start_time) / 60, 2)))

    except Exception as e:
        print('Total time: {} mins'.format(round((time() - start_time) / 60, 2)))
        raise e