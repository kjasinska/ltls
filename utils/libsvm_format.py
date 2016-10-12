# A reprodicton of code for experiments done during the internship.

def sort_libsvm_file(original_file, sorted_file):
    with open(original_file, 'r') as forig:
        with open(sorted_file, 'w') as fsort:
            for line in forig:
                fields = line.strip().split()
                labels_orig = fields[0]
                if len(fields) < 2:
                    features_orig = None
                else:
                    features_orig = fields[1:]
                if ':' in labels_orig:
                    labels_orig = None
                    features_orig = fields

                labels = list(sorted([int(label) for label in labels_orig.strip().split(",")]))
                features = list(
                    sorted([(int(pair.strip().split(":")[0]), pair.strip().split(":")[1]) for pair in features_orig]))
                labels = [str(label) for label in labels]
                newline = ",".join(labels)
                for feature, value in features:
                    newline += " {0}:{1}".format(feature, value)
                newline += "\n"
                fsort.write(newline)
