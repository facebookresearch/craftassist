# TTAD-Annotate

This is a Turk tool that is used to annotate English sentences with their
corresponding action dicts.

The tool is a series of multiple-choice questions whose answers produce
additional questions, in a structure that mostly mirrors the structure of the
action dicts.

## The Main Files

- `ttad_annotate.py`: The main executable. Run this script to produce the HTML
  that should be copied into the Turk editor, or opened in Chrome to view.

- `flows.py`: A series of JSON dicts describing the questions. This is the file
  most often edited.

- `process_results.py`: The executable which takes a Turk-produced CSV results
  file and produces a series of action dicts. There is a simple mapping from
  question/answer keys (in `flows.py`) to nested action dicts, but some
  post-processing is necessary to produce the correct action dicts, and this is
  done here.

- `make_input_csv.py`: The executable which takes a list of English language
  sentences and produces a Turk input CSV that can be uploaded as a batch.


## How To Annotate a New Batch

1. Start with a text file containing a series of English language sentences,
   one per line, e.g. `data/humanbot.input.txt`

2. Run `python make_input_csv.py <input_txt>`, and pipe the output to a file, e.g.
```
python make_input_csv.py data/humanbot.input.txt > data/humanbot.input.csv
```

3. Navigate to requester.mturk.com, login as NoahTurkProject.1016@gmail.com,
   and use the `jgray_ttad_annotate` project.

4. Select "Publish Batch", and upload the CSV file that was produced in step 2.

5. When the results are in, download the results file, which I will call `results.csv`

6. Run `python process_results.py results.csv`. The results are printed to
   stdout in the following format:
```
command text 1
{"action-dict-1": ...}
{"action-dict-2": ...}  # as many action dicts as there are different answers

command text 2
...
```

By default, if there are three Turkers who provide three different sets of
answers, all three action dicts are printed. To view only commands where there
is agreement among different Turkers, use `--min-votes 2` or `--min-votes 3`


## How to Change the Tool

1. Most changes, e.g. adding a new action or changing the name of a key, will
   take place in `flows.py`. Some post-processing may need to be done in
   `process_results.py`. Rarely will the HTML skeleton need to be changed in
   `ttad_annotate.py` or the flows-to-HTML conversion code in
   `render_flows.py`.

2. To visualize the changes to `flows.py`, run `python ttad_annotate.py >
   blah.html` and then `open blah.html` to open file in Chrome.

3. To see a Turk results file (the input to `process_results.py`), I don't know
   of an easier way than using MTurk in sandbox mode).

4. When you are happy with your changes, run `python ttad_annotate.py | pbcopy`
   pipe the output to a file, open that file in an editor, and copy the
   contents to your clipboard.

5. Navigate to requester.mturk.com, `jgray_ttad_annotate` project, click
   "Edit", and paste the HTML into the editor.
