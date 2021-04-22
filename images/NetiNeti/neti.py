from src.neti_neti_trainer import NetiNetiTrainer
from src.neti_neti import NetiNeti
from glob import glob
import sys
import os

nnt = NetiNetiTrainer()
nn = NetiNeti(nnt)

if len(sys.argv) >= 2:

    input_dir = sys.argv[1]

    result = []
    for document in glob(os.path.join(input_dir, "*.txt")):

        with open(document, 'rb') as f:
            data = unicode(f.read(), 'utf8')
            #data = data.replace("\r\n", "\n")
            
            res = nn.find_names(data)
            result += [os.path.basename(document) + " {}".format(res)]

    print("\t".join(result))
