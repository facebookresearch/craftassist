"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import csv
import argparse
import json
from collections import defaultdict, Counter
import re

from ttad_annotate import MAX_WORDS


def process_result(full_d):
    worker_id = full_d["WorkerId"]

    d = with_prefix(full_d, "Answer.root.")
    try:
        action = d["action"]
    except KeyError:
        return None, None, None
    action_dict = {action: process_dict(with_prefix(d, "action.{}.".format(action)))}

    ##############
    # repeat dict
    ##############

    if d.get("loop") not in [None, "Other"]:
        repeat_dict = process_repeat_dict(d)

        # Some turkers annotate a repeat dict for a repeat_count of 1.
        # Don't include the repeat dict if that's the case
        if repeat_dict.get("repeat_count"):
            a, b = repeat_dict["repeat_count"]
            repeat_count_str = " ".join(
                [full_d["Input.word{}".format(x)] for x in range(a, b + 1)]
            )
            if repeat_count_str not in ("a", "an", "one", "1"):
                action_val = list(action_dict.values())[0]
                if action_val.get("schematic"):
                    action_val["schematic"]["repeat"] = repeat_dict
                elif action_val.get("action_reference_object"):
                    action_val["action_reference_object"]["repeat"] = repeat_dict
                else:
                    action_dict["repeat"] = repeat_dict

    ##################
    # post-processing
    ##################

    # Fix Build/Freebuild mismatch
    if action_dict.get("Build", {}).get("Freebuild") == "Freebuild":
        action_dict["FreeBuild"] = action_dict["Build"]
        del action_dict["Build"]
    action_dict.get("Build", {}).pop("Freebuild", None)
    action_dict.get("FreeBuild", {}).pop("Freebuild", None)

    # Fix empty words messing up spans
    words = [full_d["Input.word{}".format(x)] for x in range(MAX_WORDS)]
    action_dict, words = fix_spans_due_to_empty_words(action_dict, words)

    return worker_id, action_dict, words


def process_dict(d):
    r = {}

    # remove key prefixes
    d = remove_key_prefixes(d, ["copy.yes.", "copy.no."])

    if "location" in d:
        r["location"] = {"location_type": d["location"]}
        if r["location"]["location_type"] == "location_reference_object":
            r["location"]["location_type"] = "BlockObject"
            r["location"]["relative_direction"] = d.get(
                "location.location_reference_object.relative_direction"
            )
            if r["location"]["relative_direction"] in ("EXACT", "NEAR", "Other"):
                del r["location"]["relative_direction"]
            d["location.location_reference_object.relative_direction"] = None
        r["location"].update(process_dict(with_prefix(d, "location.")))

    for k, v in d.items():
        if (
            k == "location"
            or k.startswith("location.")
            or k == "copy"
            or (k == "relative_direction" and v in ("EXACT", "NEAR", "Other"))
        ):
            continue

        # handle span
        if re.match("[^.]+.span#[0-9]+", k):
            prefix, rest = k.split(".", 1)
            idx = int(rest.split("#")[-1])
            if prefix in r:
                a, b = r[prefix]
                r[prefix] = [min(a, idx), max(b, idx)]  # expand span to include idx
            else:
                r[prefix] = [idx, idx]

        # handle nested dict
        elif "." in k:
            prefix, rest = k.split(".", 1)
            prefix_snake = snake_case(prefix)
            r[prefix_snake] = r.get(prefix_snake, {})
            r[prefix_snake].update(process_dict(with_prefix(d, prefix + ".")))

        # handle const value
        else:
            r[k] = v

    return r


def process_repeat_dict(d):
    if d["loop"] == "ntimes":
        return {
            "repeat_key": "FOR",
            "repeat_count": process_dict(with_prefix(d, "loop.ntimes."))["repeat_for"],
        }
    if d["loop"] == "repeat_all":
        return {"repeat_key": "ALL"}
    if d["loop"] == "forever":
        return {"stop_condition": {"condition_type": "NEVER"}}
    raise NotImplementedError("Bad repeat dict option: {}".format(d["loop"]))


def with_prefix(d, prefix):
    return {
        k.split(prefix)[1]: v
        for k, v in d.items()
        if k.startswith(prefix) and v not in ("", None, "None")
    }


def snake_case(s):
    return re.sub("([a-z])([A-Z])", "\\1_\\2", s).lower()


def remove_key_prefixes(d, ps):
    d = d.copy()
    rm_keys = []
    add_items = []
    for p in ps:
        for k, v in d.items():
            if k.startswith(p):
                rm_keys.append(k)
                add_items.append((k[len(p) :], v))
    for k in rm_keys:
        del d[k]
    for k, v in add_items:
        d[k] = v
    return d


def fix_spans_due_to_empty_words(action_dict, words):
    """Return modified (action_dict, words)"""

    def reduce_span_vals_gte(d, i):
        for k, v in d.items():
            if type(v) == dict:
                reduce_span_vals_gte(v, i)
                continue
            try:
                a, b = v
                if a >= i:
                    a -= 1
                if b >= i:
                    b -= 1
                d[k] = [a, b]
            except ValueError:
                pass
            except TypeError:
                pass

    # remove trailing empty strings
    while words[-1] == "":
        del words[-1]

    # fix span
    i = 0
    while i < len(words):
        if words[i] == "":
            reduce_span_vals_gte(action_dict, i)
            del words[i]
        else:
            i += 1

    return action_dict, words


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv")
    parser.add_argument(
        "--min-votes", type=int, default=1, help="Required # of same answers, defaults to 2/3"
    )
    parser.add_argument(
        "--only-show-disagreements",
        action="store_true",
        help="Only show commands that did not meet the --min-votes requirement",
    )
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    parser.add_argument(
        "--tsv", action="store_true", help="Show each result with worker id in tsv format"
    )
    args = parser.parse_args()

    result_counts = defaultdict(Counter)  # map[command] -> Counter(dict)

    with open(args.results_csv, "r") as f:
        r = csv.DictReader(f)
        for d in r:
            command = d["Input.command"]
            try:
                worker_id, action_dict, words = process_result(d)
            except:
                continue
            if action_dict is None:
                continue
            command = " ".join(words)
            result = json.dumps(action_dict)
            result_counts[command][result] += 1

            if args.debug:
                for k, v in with_prefix(d, "Answer.").items():
                    print((k, v))

            # show each result with worker info
            if args.tsv:
                print(command, worker_id, result, "", sep="\t")

    # results by command
    if not args.tsv:
        for command, counts in sorted(result_counts.items()):
            if not any(v >= args.min_votes for v in counts.values()):
                if args.only_show_disagreements:
                    print(command)
                continue
            elif args.only_show_disagreements:
                continue

            print(command)

            for result, count in counts.items():
                if count >= args.min_votes:
                    print(result)

            print()
