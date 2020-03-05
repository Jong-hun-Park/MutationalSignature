from scipy.stats import entropy
from collections import defaultdict
import numpy as np


def extract_significant_signature(entropy_threshold=1.5):
    input = "../data/signatures_ludmil.csv"
    signature_dict = defaultdict(list)
    result_dict = defaultdict(list)
    mutation_type_list = []
    with open(input, "r") as f:
        header = f.readline().strip().split("\t")
        signatures = header[3:]
        for line in f:
            split = line.strip().split("\t")
            mutation_type = split[2]
            mutation_type_list.append(mutation_type)
            for i, signature in enumerate(signatures):
                signature_dict[signature].append(float(split[i+3]))

    for key, problist in signature_dict.items():
        ep = 10000
        print(signature_dict[key])
        extracted_list = []
        print(entropy(problist))
        while ep > entropy_threshold:
            max_index = np.argmax(problist)
            extracted_list.append(mutation_type_list[max_index])
            problist[max_index] = 0
            ep = entropy(problist)
            print(ep)
        print(problist)
        print(extracted_list)
        print(entropy(signature_dict[key]))
        result_dict[key] = extracted_list

    return result_dict


if __name__ == "__main__":
    signature_dict = extract_significant_signature()



