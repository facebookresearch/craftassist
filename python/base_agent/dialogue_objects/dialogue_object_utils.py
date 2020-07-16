SPEAKERLOOK = {"reference_object": {"special_reference": "SPEAKER_LOOK"}}
SPEAKERPOS = {"reference_object": {"special_reference": "SPEAKER"}}
AGENTPOS = {"reference_object": {"special_reference": "AGENT"}}


def is_loc_speakerlook(d):
    # checks a location dict to see if it is SPEAKER_LOOK
    r = d.get("reference_object")
    if r and r.get("special_reference"):
        if r["special_reference"] == "SPEAKER_LOOK":
            return True
    return False


def process_spans(d, original_words, lemmatized_words):
    if type(d) is not dict:
        return
    for k, v in d.items():
        if type(v) == dict:
            process_spans(v, original_words, lemmatized_words)
        elif type(v) == list and type(v[0]) == dict:
            for a in v:
                process_spans(a, original_words, lemmatized_words)
        else:
            try:
                sentence, (L, R) = v
                if sentence != 0:
                    raise NotImplementedError("Must update process_spans for multi-string inputs")
                assert 0 <= L <= R <= (len(lemmatized_words) - 1)
            except ValueError:
                continue
            except TypeError:
                continue
            original_w = " ".join(original_words[L : (R + 1)])
            # The lemmatizer converts 'it' to -PRON-
            if original_w == "it":
                d[k] = original_w
            else:
                d[k] = " ".join(lemmatized_words[L : (R + 1)])


#####FIXME!!!
# this is bad
# and
# in addition to being bad, abstraction is leaking
def coref_resolve(memory, d, chat):
    """Walk logical form "d" and replace coref_resolve values

    Possible substitutions:
    - a subdict lik SPEAKERPOS
    - a MemoryNode object
    - "NULL"

    Assumes spans have been substituted.
    """

    c = chat.split()
    if not type(d) is dict:
        return
    for k, v in d.items():
        if k == "location":
            # v is a location dict
            for k_ in v:
                if k_ == "contains_coreference":
                    val = SPEAKERPOS if "here" in c else SPEAKERLOOK
                    v["reference_object"] = val["reference_object"]
                    del v["contains_coreference"]
        elif k == "filters":
            # v is a reference object dict
            for k_ in v:
                if k_ == "contains_coreference":
                    if "this" in c or "that" in c:
                        v["location"] = SPEAKERLOOK
                        del v["contains_coreference"]
                    else:
                        mems = memory.get_recent_entities("BlockObject")
                        if len(mems) == 0:
                            mems = memory.get_recent_entities(
                                "Mob"
                            )  # if its a follow, this should be first, FIXME
                            if len(mems) == 0:
                                v[k_] = "NULL"
                            else:
                                v[k_] = mems[0]
                        else:
                            v[k_] = mems[0]
        if type(v) == dict:
            coref_resolve(memory, v, chat)
        if type(v) == list:
            for a in v:
                coref_resolve(memory, a, chat)
