from time import time
import gzip
import xml.etree.ElementTree as ET


if __name__ == '__main__':
    # Configuration
    start_time = time()
    try:

        filepath = 'data/justice/33m-russian-courts-cases-by-suvorov/arb_sud/vysshij-arbitrazhnyj-sud-rf-40001/2013/306465818.xml.gz'
        with gzip.open(filepath, 'rt') as f:

            file_content = f.read()

            print(file_content)

            print("-----------------------------------")

            root = ET.fromstring(file_content)

            # Top-level elements
            root.findall(".")

            print("-----------------------------------")

            for child in root:
                print(child.tag, child.attrib)

        print('Total time: {} mins'.format(round((time() - start_time) / 60, 2)))

    except Exception as e:
        print('Total time: {} mins'.format(round((time() - start_time) / 60, 2)))
        raise e