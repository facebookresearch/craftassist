First do the following to collect the dataset from our S3 bucket:

```
curl -OL http://craftassist.s3-us-west-2.amazonaws.com/pubr/collected_dataset.tar.gz -o collected_dataset.tar.gz
tar -xzvf collected_dataset.tar.gz -C collected_dataset/ --strip-components 1
```

This folder now contains the following :
- `Grammar_Spec_doc.md` a specificcation of our grammar and logical forms.
- `annotation_tools/` : this folder contains the annotation tools with Turk support along with the postprocessing scripts for each.
- `collected_dataset/` : this folder contains the interactive an dprompts datasets alogn witht he cross validaion splits we used 
for our experiments.
- `model_training_code/` : this folder contains the training code for our models.
- `writeup/` : This folder contains the writeup of the paper 