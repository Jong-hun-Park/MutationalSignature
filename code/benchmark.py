from scipy.stats import entropy
from collections import defaultdict
import numpy as np
from graph_constructor import compute_jaccard_index


def load_signature_data(infile):
    signature_dict = defaultdict(list)
    mutation_type_list = []
    with open(infile, "r") as f:
        header = f.readline().strip().split("\t")
        signatures = header[3:]
        for line in f:
            split = line.strip().split("\t")
            mutation_type = split[2]
            mutation_type_list.append(mutation_type)
            for i, signature in enumerate(signatures):
                signature_dict[signature].append(float(split[i + 3]))

    return signature_dict, mutation_type_list


def extract_significant_signature_using_entropy(entropy_threshold=1.5):
    infile = "../data/signatures_ludmil.csv"
    signature_dict, mutation_type_list = load_signature_data(infile)

    result_dict = defaultdict(list)
    for key, problist in signature_dict.items():
        ep = 10000
        extracted_list = []
        # print("First entropy {}".format(entropy(problist)))
        # print(problist)
        while ep > entropy_threshold:
            # print(problist)
            max_index = np.argmax(problist)
            max_prob = problist[max_index]
            # print(max_prob)
            extracted_list.append(mutation_type_list[max_index])
            problist[max_index] = 0
            problist = [prob/max_prob for prob in problist]
            ep = entropy(problist)
            # print(ep)
        # print(problist)
        # print(extracted_list)
        # print(entropy(signature_dict[key]))
        result_dict[key] = extracted_list

    return result_dict


def compare_with_known_signatures(signature_name, test_list, top_k=False):
    signature_dict_using_entropy = extract_significant_signature_using_entropy(entropy_threshold=3)
    true_set = signature_dict_using_entropy[signature_name]
    if top_k:
        true_set = true_set[:len(test_list)]

    test_renamed_list = []
    for mutation in test_list:
        tri = mutation[4:]
        reformated = tri[0] + "[" + mutation[:3] + "]" + tri[-1]
        test_renamed_list.append(reformated)

    print("Testing {}...".format(signature_name))
    print("Test set", test_renamed_list)
    print("Test set length", len(test_renamed_list))
    print("True set", true_set)
    print("True set length", len(true_set))
    print("Jaccard Index {}".format(compute_jaccard_index(set(true_set), set(test_renamed_list))))


if __name__ == "__main__":
    signature_name = 'Signature 6'
    testing_set = ['C>A/CCC', 'C>A/CCT', 'C>T/ACA', 'C>T/ACC', 'C>T/ACG', 'C>T/CCG', 'C>T/GCA', 'C>T/GCC', 'C>T/GCG', 'C>T/GCT', 'C>T/TCG', 'T>C/ATC', 'T>C/ATG', 'T>C/CTA', 'T>C/CTG', 'T>C/GTA', 'T>C/GTC', 'T>C/GTG', 'T>C/GTT', 'T>C/TTC', 'T>C/TTG']
    compare_with_known_signatures(signature_name, testing_set, top_k=True)