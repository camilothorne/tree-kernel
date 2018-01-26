'''
Created on Jan 23, 2017

@author: camilo
'''

import subprocess, time
from Extract import SVMFeatures

devdata = "../DevData/"
svm     = "../SVM/svm_classify "


class Test(object):


    def __init__(self,myfile,maxsize,name):
        '''
        Constructor
        '''
              
        self.max    = 0      
        self.result = None       
        svm         = SVMFeatures(myfile,"test",maxsize,name)
        svm.saveSample("test",name)
        print "----------------------------------"        
        print("==> predicting ...") 
        self.testSVM(name)
        time.sleep(5)
        self.printResults(name)
        print "----------------------------------"
        print "done!"    


    def testSVM(self,name):

        subprocess.call(svm + devdata + name + "-test.txt " + devdata + name + "-relmod " + devdata + name + "-relres > " + devdata + name + "-test.log", shell=True)
        
        
    def printResults(self,name):
            
        f0 = open(devdata + name + "-relations.csv")
        lines0 = f0.readlines()
            
        f1 = open(devdata + name + "-relres")
        lines1 = f1.readlines()        
        
        relations = ""
        
        # compute normalization factor
        margins   = []
        j = 0
        while j < len(lines1):
            margins.append(self.margin(float(lines1[j])))
            j = j + 1
        margins.sort(reverse=True)
        mmax = margins[0]
        
        # compute raw and normalized margins
        i = 0
        while i < len(lines0):
            line    = lines0[i].replace("\n", "")          
            pred    = self.sign(float(lines1[i]))
            margin  = self.margin(float(lines1[i]))
            prob    = self.margin(float(lines1[i]))/mmax
            relations = relations + line + pred + " ; " + `margin` + " ; " + `prob` + "\n"
            i = i + 1
        self.result = relations
        f = open(devdata + name + "-relations.csv", 'w')
        f.write(self.result)
        f.close()              
        
            
    def sign(self,myfloat):
        
        if myfloat > 0:
            return "1"
        if myfloat < 0:
            return "-1"
        
        
    def margin(self,myfloat):
        
        return abs(myfloat)