import argparse
import json
import os
import sys

stack_agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(stack_agent_dir)

from train_ttad_model import ActionDictBuilder

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="../models/ttad.pth", help="model to test")
parser.add_argument("--actions", default="actions.txt", help="output of generate_actions.py")
parser.add_argument("--only-show-failure", action="store_true", help="only print failure cases")
args = parser.parse_args()

listener = ActionDictBuilder(os.path.join(stack_agent_dir, "models/ttad.pth"))


def remove_description(d):
    """Recursively remove the 'description' keys in-place"""
    if "description" in d:
        del d["description"]
    for v in d.values():
        if type(v) == dict:
            remove_description(v)


with open(args.actions, "r") as f:
    lines = f.readlines()


good, bad, err = [], [], []
for i in range(0, len(lines), 3):
    text = json.loads(lines[i])
    d = json.loads(lines[i + 1])
    remove_description(d)

    # evaluate
    try:
        guess = listener.listen(text)
    except Exception as e:
        err.append((text, d, e))
        continue

    # bucket results
    if d == guess:
        good.append((text, d, guess))
    else:
        bad.append((text, d, guess))

# sort results by size
good.sort(key=lambda tup: len(tup[0]))
bad.sort(key=lambda tup: len(tup[0]))

# print results

if not args.only_show_failure:
    print("-------- SUCCESS --------\n")
    for text, d, guess in good:
        print(*map(json.dumps, [text, d, guess]), "", sep="\n")

print("-------- FAILURE --------\n")
for text, d, guess in bad:
    print(*map(json.dumps, [text, d, guess]), "", sep="\n")


print("-------- ERROR --------\n")
for text, d, e in err:
    print(json.dumps(text), "\n", json.dumps(d), "\n", e, "\n")


print("-------- SUMMARY --------\n")
assert len(good) + len(bad) + len(err) == len(lines) / 3
print(
    "{} success + {} failure + {} error = {} accuracy".format(
        len(good), len(bad), len(err), len(good) / (len(lines) / 3)
    )
)
