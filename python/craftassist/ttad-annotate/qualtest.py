from collections import Counter, defaultdict
import fileinput
import json
import os

right_answer_count = Counter()
wrong_answer_count = Counter()

# compile sets of allowed answers
allowed_answers = defaultdict(set)
command = None
with open(os.path.join(os.path.dirname(__file__), "data/qualtest.answers.txt"), "r") as f:
    for line in f:
        line = line.strip()
        if line == "":
            continue
        if line.startswith("{"):
            try:
                json.loads(line)
                allowed_answers[command].add(line)
            except:
                print("Bad allowed answer:", line)
                raise
        else:
            command = line


# validate answers
for line in fileinput.input():
    command, worker_id, answer = line.strip().split("\t")
    action_dict = json.loads(answer)

    if not any(action_dict == json.loads(d) for d in allowed_answers[command]):
        wrong_answer_count[worker_id] += 1
    else:
        right_answer_count[worker_id] += 1


for worker_id in right_answer_count:
    print(
        right_answer_count[worker_id],
        "/",
        right_answer_count[worker_id] + wrong_answer_count[worker_id],
        worker_id,
    )
