'''
Created on Jan 16, 2017

@author: camilo
'''


import nltk
from xml.dom import minidom
from gensim.models import Word2Vec
from gensim.models import doc2vec
from nltk.corpus import stopwords
from corenlp import StanfordCoreNLP
import numpy as np
import glob
import os

devdata         = "../DevData/"

class MyExtract(object):
    '''
    classdocs
    '''
    

    def __init__(self):
        '''
        constructor
        '''
        
        self.rawcorpus  = None
        self.corpus     = []
        self.pars       = []
        self.wordspace  = None
        self.docspace   = None
        self.stop       = set(stopwords.words('english'))
        self.parser     = None
        self.prelations = []
        self.nrelations = []
        
        
    def buildRawCorpus(self,myfile):
        '''
        extract text from xml files
        '''
        
        corpus = ""
        for txtfile in glob.glob(devdata+myfile):
            
            print "reading " + txtfile
            
            xmldoc = minidom.parse(txtfile)
            itemlist = xmldoc.getElementsByTagName('text')
            for s in itemlist:
                text = s.firstChild.data
                if "." in text:
                    corpus = corpus + " " + text
        self.rawcorpus = corpus.encode("utf-8")
        
        
    def buildCorpus(self):
        '''
        preprocess raw text (tokenize, remove stopwords)
        '''
        
        sents = self.rawcorpus.split(".")
        for sent in sents:           
            toks = [w.lower() for w in nltk.word_tokenize(sent.decode('utf-8')) if w.lower() not in self.stop]
            self.corpus.append(toks) 


    def tokenizeAbs(self,parag):
        '''
        preprocess raw text (tokenize, remove stopwords)
        '''
                
        toks = [w.lower() for w in nltk.word_tokenize(parag) if w.lower() not in self.stop]
        return toks 
            
            
    def buildRawSents(self,myfile):
        
        for txtfile in glob.glob(devdata+myfile):            
            xmldoc = minidom.parse(txtfile)
            itemlist0 = xmldoc.getElementsByTagName('document')
            count = 0         
            for it0 in itemlist0:
                parag = ""
                itemlist = it0.getElementsByTagName('text')
                for item in itemlist:
                    if '.' in item.firstChild.data:
                        parag = parag + " " + item.firstChild.data               
                toks = self.tokenizeAbs(parag.encode("utf-8").decode('utf-8'))
                lab  = [txtfile+'_'+`count`]
                self.pars.append(doc2vec.LabeledSentence(words=toks,tags=lab))
                count = count + 1         
        
        
    def exploreCDRCorpus(self,myfile,maxsize):
        '''
        extract entities + relations from xml
        '''
        
        diseases    = {}
        chemicals   = {}
        relations   = []
        xmldoc = minidom.parse(myfile)
        itemlist0 = xmldoc.getElementsByTagName('document')
        count = 0
        for it0 in itemlist0:
            print "\t- processing abstract " + `count`

            parsed      = self.docspace.docvecs[myfile+"_"+`count`]

            itemlist1   = it0.getElementsByTagName('annotation')
            print "\t\t+ " +  `len(itemlist1)` + " entities"                          
            
            for it1 in itemlist1: 
                
                itemlist2   = it1.getElementsByTagName('infon')
                typ         = itemlist2[0].firstChild.data
                mesh        = itemlist2[len(itemlist2)-1].firstChild.data
                text        = it1.getElementsByTagName('text')[0].firstChild.data.lower()           
                codes       = mesh.split('|')
                
                for code in codes:
                    ent = MyEntity(text,code,typ)              
                    if (typ == 'Chemical'):
                        chemicals[code] = ent
                    if (typ == 'Disease'):
                        diseases[code]  = ent
            
            itemlist3 = it0.getElementsByTagName('relation')
            
            print "\t\t+ " +  `2*len(itemlist3)` + " positive and negative relations"
            print "\t\t\t* extracting features for positive relations"
            print "\t\t\t* extracting features for negative relations"
                                                   
            for it3 in itemlist3:
                
                itemlist4 = it3.getElementsByTagName('infon')    
                key1 = itemlist4[1].firstChild.data
                key2 = itemlist4[2].firstChild.data
                e1  = chemicals[key1]
                e2  =  diseases[key2]
                e1.bow = self.avgBOW(e1.text)
                e2.bow = self.avgBOW(e2.text)                
                rel = MyRelation(e1,e2,'1')
                rel.abs = parsed
                self.prelations.append(rel)                
                relations.append(key1 + "_" + key2)
                num = 0
                
            for key1 in chemicals.keys():
                for key2 in diseases.keys():
                    if key1 + "_" + key2 not in relations:
                        if num < len(itemlist3):
                            e1 = chemicals[key1]
                            e2 =  diseases[key2]                      
                            e1.bow = self.avgBOW(e1.text)
                            e2.bow = self.avgBOW(e2.text)
                            rel = MyRelation(e1,e2,'-1')
                            rel.abs = parsed
                            self.nrelations.append(rel)                      
                            num = num + 1
                            
            count = count + 1
            if (count == maxsize):
                break     
    
            
    def exploreDDICorpus(self,myfile,maxsize,ftyp):
        '''
        extract entities + relations from xml
        '''
        
        #print(myfile)
        
        xmldoc = minidom.parse(myfile)
        itemlist0 = xmldoc.getElementsByTagName('document')
        count = 0
        
        for it0 in itemlist0:
            
            # abstract with annotations
            print "\t- processing abstract " + `count`
            drugs       = {}

            # entities
            itemlist1   = it0.getElementsByTagName('annotation')
            print "\t\t+ " +  `len(itemlist1)` + " entities"                          
            for it1 in itemlist1: 
                
                itemlist2a   = it1.getElementsByTagName('infon')
                typ          = itemlist2a[0].firstChild.data
                print typ
                
                itemlist2b   = it1.getElementsByTagName('text')
                text         = itemlist2b[0].firstChild.data.lower()
                print text
                
                ent         = MyEntity(text,"",typ)
                ent.bow     = self.avgBOW(ent.text)            
                drugs[text] = ent
            
            # abstract
            itemlist3 = it0.getElementsByTagName('text')
            abstract = ""                  
            for it3 in itemlist3:
                if (len(it3.firstChild.data.split())>3):
                    abstract = abstract + it3.firstChild.data
            
            # parse abstract        
            parsed      = self.parseSentence(abstract) #stanford
            docvec      = self.docspace.docvecs[myfile+"_"+`count`] #doc2vec   
            
            #print len(drugs.keys())         

            if (len(drugs.keys()) > 1):
                
                e1 = drugs[drugs.keys()[0]]
                e2 = drugs[drugs.keys()[1]]                
                e1.bow = self.avgBOW(e1.text)
                e2.bow = self.avgBOW(e2.text)
                
                #print(ftyp)
                
                if (ftyp == "positive"):
                    
                    #print(parsed)
                
                    rel = MyRelation(e1,e2,'1')
                    rel.abs     = docvec
                    rel.parse   = parsed.encode("utf-8")
                    self.prelations.append(rel)                      
                    
                if (ftyp == "negative"):
                    
                    #print(docvec)
                
                    rel = MyRelation(e1,e2,'-1')
                    rel.abs     = docvec
                    rel.parse   = parsed.encode("utf-8")
                    self.nrelations.append(rel)                   
                            
            # increment counter                
            count = count + 1
            if (count == maxsize):
                break                                                                           
            
    
    def avgBOW(self,entity):
        bow = []
        ents = entity.split(" ")
        i = 0
        while i < self.wordspace.layer1_size:
            v = 0
            for ent in ents:
                if ent in self.wordspace.vocab:
                    v = v + self.wordspace[ent][i]
            bow.append(v/len(ents))
            i = i + 1
        return np.array(bow)
        
    
    def buildWordSpace(self,modelfile):
        '''
        compute distributional model
        '''
        
        model = Word2Vec(self.corpus,min_count=1,size=20,iter=100,workers=4)
        model.save(modelfile)
        self.wordspace = model
        
        
    def buildDocSpace(self,modelfile):
        '''
        compute distributional model
        '''
        
        model = doc2vec.Doc2Vec(self.pars,min_count=5,size=20,iter=100,workers=4)
        model.save(modelfile)
        self.docspace = model       
        
        
    def loadWordSpace(self,modelfile):
        '''
        compute distributional model
        '''
        
        self.wordspace = Word2Vec.load(devdata+modelfile)
        
        
    def loadDocSpace(self,modelfile):
        '''
        compute distributional model
        '''
        
        self.docspace = doc2vec.Doc2Vec.load(devdata+modelfile)
        
        
    def loadParser(self):

        corenlp_dir     = os.environ['STANFORD']
        self.parser     = StanfordCoreNLP(corenlp_dir+"/") # wait a few minutes...    
        
        
    def parseSentence(self,sentence):
        
        parsed = self.parser.raw_parse(sentence)['sentences'][0]['parsetree']
        return parsed
    
    
class SVMFeatures:
    
    
    def __init__(self,myfile,mode,size,corpus):
        
        if (corpus == "CDR"):
            self.extractCDR(myfile, mode, size)
        if (corpus == "DDI"):
            self.extractDDI(myfile, mode, size)
            
        
    def extractCDR(self,myfile,mode,size):
        '''
        CDR corpus
        '''
        
        ext = MyExtract()
        print "=================================="   
        print "1. reading corpus ..."             
        if (mode == "train"):
            if (os.path.exists(devdata+"wordspace") & os.path.exists(devdata+"docspace")):
                print "2. loading word space ..."
                ext.loadWordSpace(devdata+"wordspace")
                print "3. loading doc space ..."
                ext.loadDocSpace(devdata+"docspace")              
            else:
                print "\t+ reading words"  
                ext.buildRawCorpus("*.xml")
                ext.buildCorpus()
                print "\t+ reading sentences"  
                ext.buildRawSents("*.xml")              
                print "2. building word space ..."
                ext.buildWordSpace(devdata+"wordspace")
                print "3. building doc space ..."
                ext.buildDocSpace(devdata+"docspace")
        if (mode == "test"):   
            print "2. loading word space ..."
            ext.loadWordSpace(devdata+"wordspace")
            print "3. loading doc space ..."
            ext.loadDocSpace(devdata+"docspace")              
        print "4. extracting data ..."           
        ext.exploreCDRCorpus(devdata+myfile,size)
        print "5. building sample ..."
        neg = ""
        pos = ""
        csv = ""        
        for rel in ext.nrelations:
            neg = neg + self.extractExample(rel)
            csv = csv + self.extractCSV(rel)
        for rel in ext.prelations:
            pos = pos + self.extractExample(rel)
            csv = csv + self.extractCSV(rel)
        self.sample = pos + neg
        self.csv    = csv        
  
  
    def extractDDI(self,myfile,mode,size):
        '''
        DDI corpus
        '''
        
        ext = MyExtract()
        print "=================================="        
        print "1. reading corpus ..."             
        if (mode == "train"):
            if (os.path.exists(devdata+"wordspace") & os.path.exists(devdata+"docspace")):
                print "2. loading word space ..."
                ext.loadWordSpace(devdata+"wordspace")
                print "3. loading doc space ..."
                ext.loadDocSpace(devdata+"docspace")              
            else:
                print "\t+ reading words"  
                ext.buildRawCorpus("*.xml")
                ext.buildCorpus()
                print "\t+ reading sentences"  
                ext.buildRawSents("*.xml")              
                print "2. building word space ..."
                ext.buildWordSpace(devdata+"wordspace")
                print "3. building doc space ..."
                ext.buildDocSpace(devdata+"docspace")
            print "4. loading parser ..."
            ext.loadParser()                          
            print "5. extracting data ..."
            for txtfile in glob.glob(devdata+myfile):
                if ("pos" in txtfile):
                    ext.exploreDDICorpus(txtfile,size,"positive")
                if ("neg" in txtfile):
                    ext.exploreDDICorpus(txtfile,size,"negative")            
        if (mode == "test"): 
            print "2. loading word space ..."
            ext.loadWordSpace(devdata+"wordspace")
            print "3. loading doc space ..."
            ext.loadDocSpace(devdata+"docspace")   
            print "4. loading parser ..."
            ext.loadParser()                          
            print "5. extracting data ..."
            ext.exploreDDICorpus(devdata+myfile,size,"negative")
        print "6. building sample ..."
        neg = ""
        pos = ""
        csv = ""        
        for rel in ext.nrelations:
            neg = neg + self.extractExample2(rel)
            csv = csv + self.extractCSV2(rel)
        for rel in ext.prelations:
            pos = pos + self.extractExample2(rel)
            csv = csv + self.extractCSV2(rel)
        self.sample = pos + neg
        self.csv    = csv  
    
    
    def saveSample(self,typ,name):
        
        if (typ == "train"):
            
            f = open(devdata+name+"-train.txt", 'w')
            f.write(self.sample)
            f.close()
            
        if (typ == "test"):
            
            f0 = open(devdata+name+"-test.txt", 'w')
            f0.write(self.sample)
            f0.close()
            f1 = open(devdata+name+"-relations.csv", 'w')
            f1.write(self.csv)
            f1.close()             

    
    def extractCSV(self,relation):
    
        csv = ""
        en1 = relation.en1.text + " ; " + relation.en1.mesh  + " ; "
        en2 = relation.en2.text + " ; "+ relation.en2.mesh + " ; "
        rel = relation.sign + " ; "
        csv = csv + en1 + en2 + rel + "\n"
        return csv


    def extractCSV2(self,relation):
    
        csv = ""
        en1 = relation.en1.text + " ; " 
        en2 = relation.en2.text + " ; "
        rel = relation.sign + " ; "
        csv = csv + en1 + en2 + rel + "\n"
        return csv
    
    
    def extractExample(self,relation):
        
        en1     = relation.en1
        en2     = relation.en2
        sign    = relation.sign
        parse   = relation.abs
        
        vector  = sign + " "
        count   = 1
        for val in en1.bow:
            if (abs(val) > 0):
                vector = vector + `count` + ":" + `val` + " "
            count = count + 1
        for val in en2.bow:
            if (abs(val) > 0):
                vector = vector + `count` + ":" + `val` + " "
            count = count + 1
        for val in parse:
            if (abs(val) > 0):
                vector = vector + `count` + ":" + `val` + " "
            count = count + 1         
        return vector + "\n"


    def extractExample2(self,relation):
        
        en1     = relation.en1
        en2     = relation.en2
        sign    = relation.sign
        parse   = relation.abs
        tree    = relation.parse        
        
        res     = sign + " "
        res     = res + "|BT| " + tree + " |ET| "
        vector  = "|BV| "
        count   = 1
        for val in en1.bow:
            if (abs(val) > 0):
                vector = vector + `count` + ":" + `val` + " "
            count = count + 1
        for val in en2.bow:
            if (abs(val) > 0):
                vector = vector + `count` + ":" + `val` + " "
            count = count + 1
        for val in parse:
            if (abs(val) > 0):
                vector = vector + `count` + ":" + `val` + " "
            count = count + 1         
        res = res + vector + " |EV|\n"
        return res
        

class MyEntity(object):
    
    def __init__(self,text,uid,typ):
        
        self.text   = text
        self.mesh   = uid
        self.type   = typ
        self.bow    = None

    
class MyRelation(object):
    
    def __init__(self,en1,en2,sign):
        
        self.en1    = en1
        self.en2    = en2
        self.sign   = sign
        self.abs    = None
        self.parse   = None
        
        
class MySentence(object):
    
    def __init__(self,text):
        
        self.text    = text
        self.corenlp = None  
                                      
          
        