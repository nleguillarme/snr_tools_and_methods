import merpy
from glob import glob
import sys
import os
import pandas as pd

def get_entity_dict(ent):
	ent_dict = {
		"offsets": "LIVB {} {}".format(ent[0], ent[1]),
		"text": ent[2].replace("\n", " "),
	}
	return ent_dict

if len(sys.argv) >= 3:

	input_dir = sys.argv[1]
	output_dir = sys.argv[2]

	result = []
	for document in glob(os.path.join(input_dir, "*.txt")):

		with open(document, 'r') as f:
			data = f.read()
                     
		entities = merpy.get_entities(data, "ncbi")
		entities = [get_entity_dict(ent) for ent in entities if len(ent) == 3]
		df = pd.DataFrame(entities)
		df = df.dropna()
		df = df.loc[df.astype(str).drop_duplicates().index]
		df = df.reset_index(drop=True)
		df = df.rename("T{}".format)
		ann_filename = os.path.basename(document).split(".")[0]+".ann"
		df.to_csv(os.path.join(output_dir, ann_filename), sep="\t", header=False)
