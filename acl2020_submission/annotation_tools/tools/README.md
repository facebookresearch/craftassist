# Tree to Text tool

This is a Turk tool that is used to annotate English sentences with their
corresponding logical forms.

There is a series of multiple-choice questions answers to which might produce
additional questions, in a structure that mostly mirrors the structure of our nested
dictionaries.

The complete annotation is a two step process:
- `annotation_tool_1` : This tool queries the user for top-level tree structure. Specifically,
it asks about the intended action, words of the sentence that correspond to the respective
children of the action and any repeat patterns.
- `annotation_tool_2`: This tool goes over the children. Specifically, it queries
the users about the highlighted words of subcomponents that came from the first tool
to determine specific properties like name, colour, size etc.

An example:
For the sentence : ```build a red cube here```
The first tool gets a sense of the action which is `Build` in this case, the words for
what needs to be built (`red cube`) and words for where the construction will
happen(`here`).

The second tool now queries the user for:
- The name, colour, size, height etc of the thing that needs to be
built(`red cube` in build a **red cube** here).
- The specifics of the location where the construction will
happen(`here` in build a **red cube** here), by asking questions about whether
the location is relative to something, described using words that are reference to a location etc.

## Render the html
- To see the first tool:
```
 python annotation_tool_1.py > step_1.html
```
- To see the second tool:

```
python annotation_tool_2.py > step_2.html
```
and then open these html filesin browser.

### Note
The second annotation tool needs some input to dynamically render content, most divs
are hidden by default and this is closely stitched to our Turk use case.
But you can do the following to see all questions:
In `annotation_tool_2.py` just change the `display` :
```
render_output = (
            """<div style='font-size:16px;display:none' id='"""
            + id_value
            + """'> <b>"""
            + sentence
            + """</b>"""
        )
```
from `display:none` to `display:block`.



## Integration in Turk and post-processing

### Creating the Turk interface tool
To integrate these tools in Mechanical Turk:

1. Copy the content of the tools: 
```
python annotation_tool_1.py | pbcopy
```

2. Go to your Mechanical Turk account
3. Create -> New Project
4. Set your project name and all other properties.
5. Go to Design Layout -> Source and paste the content of pbcopy here.

Note that: both tools have all the html, css and javascript needed for integration with Mechanical Turk interface already, so no additional changes need to be made.
The steps explained will be the same for `annotation_tool_2.py` as well as `composite_command_tool.py`.


### Constructing input for turk tools and postprocessing

1. Use the file: `construct_input_for_turk.py` with options `input_file` as a txt file with one command per line and `tool_num` = 1 to generate Turk input for tool1.
The output of above script will be a `.csv` file that can be used to publish batch for tool1.
2. When all results from Turkers are in, use the notebook : `postprocessing_tool_output_notebooks/step_1_construct_dict_from_tool_1.ipynb` to 
postprocess output of tool1 and create a file that has an agreement amongst the turkers for every command.
3. Now use: `postprocessing_tool_output_notebooks/step_2_create_tool_2_input_from_tool_1.ipynb` to create txt input for tool 2.
4. Use `construct_input_for_turk.py` with `input_file` as txt file generated from step #3 and `tool_num` = 2 to generate Turk input for tool2.
The output of above script will be a `.csv` file that can be used to publish batch for tool2.
5. When all results from Turkers are in, use the notebook : `postprocessing_tool_output_notebooks/step_3_construct_dict_from_tool_2.ipynb` to 
postprocess output of tool2 and create a file that has an agreement amongst the turkers for every child of command.
6. Now we will combine the dictionaries of children with their respective parents for each command in : `postprocessing_tool_output_notebooks/step_4_combine_tool_1_and_2.ipynb`
to get the final logical form / action dictionary.


## The Main Files

- `annotation_tool_1.py` and `annotation_tool_2.py`: The main executables. Run
  these scripts to produce the HTML files that can be copied into the Turk
  editor, or opened in Chrome to view.

- `question_flow_for_step_1.py` and `question_flow_for_step_2.py`: A series of
  JSONs describing the respective questions. This is the file you'll be editing
  if you want to change the questions.

- `render_questions_tool_1.py` and `render_questions_tool_2.py`: These files render the
  actual HTML elements using the JSON produced from questions flows.
