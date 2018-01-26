'''
Created on Jan 23, 2017

@author: camilo
'''


from Train import Train
from Test import Test


if __name__ == '__main__':

    #Train("CDR-train.xml",180,"CDR")
    #Test("DDI-test.xml",20,"CDR")
    
#    Train("DDI-train-*.xml",242,"DDI")
#    Test("DDI-test.xml",30,"DDI")
    
    Train("DDI-train-*.xml",120,"DDI",mode="-t 5 -C T+V ")
    Test("DDI-test.xml",60,"DDI")    
