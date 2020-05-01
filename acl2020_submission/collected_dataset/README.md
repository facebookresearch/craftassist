If you haven't yet, download the annotated dataset from our S3 bucket :

```
curl -OL http://craftassist.s3-us-west-2.amazonaws.com/pubr/collected_dataset.tar.gz -o collected_dataset.tar.gz
tar -xzvf collected_dataset.tar.gz -C collected_dataset/ --strip-components 1
```

This folder contains the dataset collected using the tools. 

- `interactive_dataset.json` was collected in the interactive setting and has 2161 commands. 
- `prompts_dataset.json` was collected in the prompts setting and has 4532 commands.
- `combined_dataset.json` is a json file containing the data from both `interactive_dataset.json` and `prompts_dataset.json` along with 691 new commands that involved some additions to the grammar since the paper was submitted.

The folder: `experiment_splits` has the 5 fold train and test splits of these datasets that we used for experiments reported in the paper.
