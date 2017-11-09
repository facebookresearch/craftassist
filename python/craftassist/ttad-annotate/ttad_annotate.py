MAX_WORDS = 30

CSS_SCRIPT = """
<script>
var node = document.createElement('style');
"""
for i in range(MAX_WORDS):
    CSS_SCRIPT += """
        if (! "${{word{i}}}") {{
            node.innerHTML += '.word{i} {{ display: none }} '
        }}
    """.format(
        i=i
    )
CSS_SCRIPT += """
document.body.appendChild(node);
</script>
"""

JS_SCRIPT = """
$(function () {
  $('[data-toggle="tooltip"]').tooltip()
})
"""

BEFORE = """
<!-- Bootstrap v3.0.3 -->
<link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />
<section class="container" id="Other" style="margin-bottom:15px; padding: 10px 10px;
  font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">
  <div class="row col-xs-12 col-md-12">

    <!-- Instructions -->
    <div class="panel panel-primary">
      <div class="panel-heading"><strong>Instructions</strong></div>

      <div class="panel-body">
        <p>Each of these sentences is spoken to an assistant
         who is tasked with helping the speaker.
         We are looking to determine the meaning of the commands given to the assistant.</p>
        <p>For each command, answer a series of questions. Each question is either multiple-choice,
         or requires you to select which words in the sentence
         correspond to which part of the command.</p>
        <p>For example, given the command <b>"Build a house next to the river"</b>
        <ul>
        <li>For "What action is being instructed?", the answer is "Build"</li>
        <li>For "What should be built?", select the button for "house"</li>
        <li>For "Where should it be built?", the answer is "Relative to other object(s)"
        <li>For "What other object(s)?", select the buttons for "the river"</li>
        <li>etc.</li>
        </ul>
        <p>There may not be a suitable answer that captures the meaning of a sentence.
        Don't be afraid to select "Other" if there is no good answer.</p>
      </div>
    </div>

    <div class="well" style="position:sticky;position:-webkit-sticky;top:0;z-index:9999">
    <b>Command: </b>${command}</div>

    <!-- Content Body -->
    <section>
"""

AFTER = """
    </section>
    <!-- End Content Body -->

  </div>
</section>

<style type="text/css">
  fieldset {{
    padding: 10px;
    background: #fbfbfb;
    border-radius: 5px;
    margin-bottom: 5px;
  }}
</style>

{CSS_SCRIPT}

<script src="https://code.jquery.com/jquery.js"></script>
<script src="https://netdna.bootstrapcdn.com/bootstrap/3.0.3/js/bootstrap.min.js"></script>

<script>{JS_SCRIPT}</script>
""".format(
    CSS_SCRIPT=CSS_SCRIPT, JS_SCRIPT=JS_SCRIPT
)

if __name__ == "__main__":
    import render_flows
    from flows import *

    print(
        BEFORE,
        render_flows.render_q(Q_ACTION, "root", show=True),
        render_flows.render_q(Q_ACTION_LOOP, "root", show=True),
        AFTER,
    )
