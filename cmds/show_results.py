import glob
import numpy
import sys
import os
import pickle, gzip
import bloscpack as bp
import pandas as pd

from io_func.model_io import  log
from utils.stop_handler import stop_if_stop_is_requested;

pred_file = sys.argv[1]

if '.gz' in pred_file:
    pred_mat = pickle.load(gzip.open(pred_file, 'rb'))
else:
    pred_mat = pickle.load(open(pred_file, 'rb'))
 
l = sorted(glob.glob(sys.argv[2]))
if len(l) == 0:  
	log("ERROR in show_results. Test partitions is empty. Argument "+sys.argv[2])

subclassificationMapping = pd.read_csv("/ssd/subclassificationMapping",sep="=",names=["NUMBER","NAME"],index_col="NUMBER")
print(subclassificationMapping.loc[121][0])


test_labels = bp.unpack_ndarray_file(l[0]+".labels")
test_labels = test_labels.astype(numpy.int32)

# Read the subclassifications
assert l[0][-4:] == '.blp' , "Invalid extension "+l[0][-4:]
fn = l[0][:-4]+".ignored.csv.gz"
df = pd.read_csv(fn,sep=';',usecols=['SUBCLASSIFICATION'],dtype=numpy.int32)
assert df.shape[0] == test_labels.shape[0],"Shapes not equal"

# End read the subclassification

for i in range (1,len(l)):
    stop_if_stop_is_requested()
    lab = bp.unpack_ndarray_file(l[i]+".labels")
    lab = lab.astype(numpy.int32)
    test_labels = numpy.concatenate((test_labels,lab))
    fn = l[i][:-4]+".ignored.csv.gz"
    dfTmp = pd.read_csv(fn,sep=';',usecols=['SUBCLASSIFICATION'],dtype=numpy.int32)
    df = pd.concat([df,dfTmp])

assert df.shape[0] == test_labels.shape[0],"Shape not equal"
assert df.shape[0] == pred_mat.shape[0],"Shape not equal"

subclass = df.as_matrix().flatten()

min = subclass[0]
max = subclass[1]
for x in subclass:
  if x < min:
    min = x
  if x > max:
    max = x

print("MIN "+str(min)+" MAX "+str(max))

correct_number = 0.0
print(subclass.shape)
subclassificationData = numpy.zeros((subclassificationMapping.shape[0],3),dtype=numpy.int32)
print("SHAPE "+str(subclassificationData.shape))
tp=0
tn=0
fp=0
fn=0
for i in range(0,pred_mat.shape[0]):
    stop_if_stop_is_requested()
    p = pred_mat[i, :]
    p_sorted = (-p).argsort()
    subclassificationNumber = subclass[i]-1 
    numberCorrect = 0
    if test_labels[i] == 1:
       if p_sorted[0] == tst_labels[i]:
             tp += 1
       else:
      	     fp += 1
    if test_labels[i == 0:
       if p_sorted[0] == test_labels[i]:
             tn += 1
       else
             fn += 1
    if p_sorted[0] == test_labels[i]:
        correct_number += 1
        numberCorrect = 1
    subclassificationData[subclassificationNumber-1,0] += 1
    subclassificationData[subclassificationNumber-1,1] += numberCorrect

# output the final error rate
error_rate = 100 * (1.0 - correct_number / pred_mat.shape[0])
auc=0.5*((tp/(tp+tn))+(tn/(tn+fp)))

print('Error rate is ' + str(error_rate) + ' (%)')
print('AUC is '+str(auc))

theList = []
for i in range(0,subclassificationData.shape[0]):
    total = subclassificationData[i,0]
    correct = subclassificationData[i,1]
    name = subclassificationMapping.loc[i+1][0]
    if total > 0: 
        theList.append((name,100 * ( 1.0 - correct  / total),total,correct)) 

theList = sorted(theList,key=lambda x: x[1])
print("[START BREAKDOWN]")
subTotal = 0
subCorrect = 0
for name,errorRate,total,correct in theList:
   if total > 0:
      assert correct <= total,"Inconsistency correct > total"
      print(str(name)+" = "+str(errorRate)+"% correct="+str(correct)+" total="+str(total))
      subTotal += total
      subCorrect += correct
print("[END BREAKDOWN]")
calculatedError = 100*(1.0-subCorrect / subTotal)
print("Total "+str(subTotal))
assert calculatedError == error_rate,"Error is wrong"
