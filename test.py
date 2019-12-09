from time import time
import gzip
from bs4 import BeautifulSoup


if __name__ == '__main__':

    magic = '''<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
            "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd" [
            <!ENTITY nbsp ' '>
            ]>'''

    # Configuration
    start_time = time()
    try:

        filepath = 'data/justice/33m-russian-courts-cases-by-suvorov/arb_sud/vysshij-arbitrazhnyj-sud-rf-40001/2013/306465818.xml.gz'
        with gzip.open(filepath, 'rt') as f:

            file_content = f.read()

            # file_content = magic + '<root>' + file_content + '</root>'

            print(file_content)

            print("-----------------------------------")

            soup = BeautifulSoup(file_content, 'html.parser')

            for child in soup.find_all('p'):
                print(child)


        print('Total time: {} mins'.format(round((time() - start_time) / 60, 2)))

    except Exception as e:
        print('Total time: {} mins'.format(round((time() - start_time) / 60, 2)))
        raise e