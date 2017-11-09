import json
import sys

data_in_text_file = sys.argv[1]
data_out_json_file = sys.argv[2]

print("parsing text file")
res = []
exple = []

f_text = open(data_in_text_file)
for i, line in enumerate(f_text):
    if line.strip() == "":
        res += [(exple[:-1], json.loads(exple[-1]))]
        exple = []
    else:
        if line.startswith('"human:'):
            exple += [line.strip()[1:-1]]
        else:
            exple += [line.strip()]
    if i % 100000 == 0:
        print("read %d lines, found %d examples" % (i, len(res)))

f_text.close()

n_train = (9 * len(res)) // 10
n_valid = len(res) // 20

data = {
    "train": {"templated": res[:n_train]},
    "valid": {"templated": res[n_train : n_train + n_valid]},
    "test": {"templated": res[n_train + n_valid :]},
}

print("saving data dict")
json.dump(data, open(data_out_json_file, "w"))
