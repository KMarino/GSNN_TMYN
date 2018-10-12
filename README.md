# Graph Search Neural Networks
This code contains the code for the CVPR Paper "The More You Know: Using Knowledge Graphs for Image Classification"

Please cite this paper if you use this code or find our work helpful.

```
@article{marino2016more,
  title={The more you know: Using knowledge graphs for image classification},
  author={Marino, Kenneth and Salakhutdinov, Ruslan and Gupta, Abhinav},
  journal={CVPR},
  year={2017}
}
```

The code is based off of https://github.com/yujiali/ggnn, and uses much of that code for the GGNN. Please also cite their paper.

```
@article{li2015gated,
  title={Gated graph sequence neural networks},
  author={Li, Yujia and Tarlow, Daniel and Brockschmidt, Marc and Zemel, Richard},
  journal={arXiv preprint arXiv:1511.05493},
  year={2015}
}
```

## Loading the data and graphs
To use our code, download our data from this google drive link and place in a good location: 
https://drive.google.com/file/d/1_ObEsWHHOrzAwLf5OvdY-64r3Ne4nTcR/view?usp=sharing

## Torch dependencies
Experiments were run using Torch 7. The following torch libraries are require to run. 

* cudnn
* nngraph
* csvigo
* json

## Training code
Go into the VisualGenome or COCO directories to run the graph or baseline code for VG or COCO respectively. You will need to change the data and locations to match your local directories. Other than that, all of the parameters should be set correctly, and you can simply run 

$ th train_*.lua

For instance, to run the COCO graph training:

$ cd COCO

$ th train_graph_COCO.lua

## Testing code
To test a trained model, first run the relevant testing script

$ th test_*.lua

Then, generate a prediction file using load_file=<experiment_location>_testresults.t7 save_file=<prediction_file>*.csv th saveoutputs.lua

Finally, run 

$ python calc_ap.py <prediction_file> <test_labels.csv file> <text_outputfile>.txt

For instance, to get results for testing GSNN on COCO, run

$th test_graph_COCO.lua

$load_file=<model_directory>/graph_vgonly_COCO_testresults.t7> save_file=<model_directory>/graph_vgonly_COCO_predictions.csv th saveoutputs.lua

$ python calc_ap.py <model_directory>/graph_vgonly_COCO_predictions.csv <data_dir>/COCO/torchdata/test_labels.csv <model_directory>/graph_vgonly_COCO_output.txt  
