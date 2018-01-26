'''
Created on Jan 23, 2017

@author: camilo
'''

import subprocess
from Extract import SVMFeatures

devdata = "../DevData/"
svm     = "../SVM/svm_learn -z c "


class Train(object):


    def __init__(self,myfile,maxsize,name,mode=""):
        '''
        Constructor
        '''
        
        svm = SVMFeatures(myfile,"train",maxsize,name)
        svm.saveSample("train",name)
        print "----------------------------------"        
        print("==> training ...")       
        self.trainSVM(name,mode)
        print "----------------------------------"
        print "done!"
     
        
    def trainSVM(self,name,mode):
        
        #print svm + mode
        
        subprocess.call(svm + mode + devdata + name +"-train.txt " + devdata + name + "-relmod > " + devdata + name + "-train.log", shell=True)