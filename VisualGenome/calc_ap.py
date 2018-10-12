import numpy as np
import pandas as pd
import cap_eval_utils
import argparse

# Parse input
parser = argparse.ArgumentParser(description='Parser for evaluation code')
parser.add_argument('predfile', help='Location of output file')
parser.add_argument('gtfile', help='Location of ground truth file')
parser.add_argument('outfile', default='aps.csv', help='File to save aps to')
args = parser.parse_args()

# Load output and label files
# Load outputs
df = pd.read_csv(args.predfile, sep=',', header=None)
scr = df.values

# Load labels
df = pd.read_csv(args.gtfile, sep=',', header=None)
gts = df.values

# Calculate using Ross' code
aps_r = []
for c in range(316):
    pred = []
    gt = []
    for i in range(len(gts)):
        pred.append(scr[i][c])
        gt.append(gts[i][c])

    P, R, score, ap = cap_eval_utils.calc_pr_ovr_noref(np.asarray(gt), np.asarray(pred))    
    aps_r.append(ap)

print('mAP: ', np.mean(aps_r))

# Save APs to file
with open(args.outfile, 'w') as outfile:
    for ap in aps_r:
        outfile.write("%f," % ap)

