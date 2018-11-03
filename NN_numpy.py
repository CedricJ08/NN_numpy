# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 13:14:39 2018

@author: Alu
"""


import numpy as np


class Classifier :
    def __init__(self,input_dim,output_dim,hidden_layer_dim,batch_size,epochs,lr_min,lr_max):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.batch_size=batch_size
        self.epochs = epochs
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.W_1 = np.random.rand(self.hidden_layer_dim,self.input_dim)/(self.hidden_layer_dim*self.input_dim)
        self.W_2 = np.random.rand(self.output_dim,self.hidden_layer_dim)/(self.output_dim*self.hidden_layer_dim)
        self.b_1 = np.random.rand(self.hidden_layer_dim)/(self.hidden_layer_dim)
        self.b_2 =  np.random.rand(self.output_dim)/(self.output_dim)
        
    def get_Weigts(self):
        return (([self.W_2,self.W_1],[self.b_2,self.b_1]))
        
    def fit(self,X,y):
        y_len=len(y)
        assert y_len == len(X)
#        loss_epo=[]
        for epo in range (self.epochs):
            print ('Epochs : '+str(epo)+'/'+str(self.epochs))
            batch_counter=0
            lr=self.lr_max*np.exp(-((1/self.epochs)*np.log(self.lr_max/self.lr_min))*epo)
            Grb_2 = np.zeros(self.output_dim)
            GrW_2 = np.zeros((self.output_dim,self.hidden_layer_dim))
            Grb_1 = np.zeros(self.hidden_layer_dim)
            GrW_1 = np.zeros((self.hidden_layer_dim,self.input_dim))
#            loss = []
            for i in range (y_len):
#                print ('step : '+str(i)+'/'+str(y_len))
                x=X[i]
                y_i=y[i]
                z_1= np.dot( self.W_1 , x)  + self.b_1
                a_1= np.tanh(z_1)
                z_2= np.dot( self.W_2 , a_1)  + self.b_2
                a_2= 1/(1+np.exp(-z_2))
                grb_2 = np.array([(a_2[j]-y_i[j]) for j in range (self.output_dim)])
                grW_2 = np.array([[a_1[k]*grb_2[j] for k in range(self.hidden_layer_dim)] for j in range (self.output_dim)])
                grb_1 = np.array([(1-(a_1[j]**2))*sum([(a_2[k]-y_i[k])*self.W_2[k][j] for k in range (self.output_dim)]) for j in range(self.hidden_layer_dim) ])
                grW_1 = np.array([[x[k]*grb_1[j] for k in range (self.input_dim)] for j in range(self.hidden_layer_dim)])
                Grb_2 = Grb_2+ grb_2
                GrW_2 = GrW_2+ grW_2
                Grb_1 = Grb_1+ grb_1
                GrW_1 = GrW_1+ grW_1
#                loss.append(-sum([y_i[j]*np.log(a_2[j])+(1-y_i[j])*np.log(1-a_2[j]) for j in range (self.output_dim)]))
                if (i+1) % self.batch_size == 0 or i+1 == y_len:
                    self.b_1=self.b_1 -(lr/self.batch_size)*Grb_1
                    self.b_2=self.b_2 -(lr/self.batch_size)*Grb_2
                    self.W_1=self.W_1 -(lr/self.batch_size)*GrW_1
                    self.W_2=self.W_2 -(lr/self.batch_size)*GrW_2                   
                    Grb_2 = np.zeros(self.output_dim)
                    GrW_2 = np.zeros((self.output_dim,self.hidden_layer_dim))
                    Grb_1 = np.zeros(self.hidden_layer_dim)
                    GrW_1 = np.zeros((self.hidden_layer_dim,self.input_dim))
                    batch_counter+=1
#            loss_epo.append(np.mean(np.array(loss)))    




    def predict(self,X):
        y=np.zeros(len(X))
        for i in range (len(X)):
            x=X[i]
            z_1= np.dot( self.W_1 , x)  + self.b_1
            a_1= np.tanh(z_1)
            z_2= np.dot( self.W_2 , a_1)  + self.b_2
            y[i]= 1/(1+np.exp(-z_2))
        return(y)




class Classifier_2 :
    def __init__(self,input_dim,output_dim,hidden_layers_dim,batch_size,epochs,lr_min,lr_max):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers_dim = hidden_layers_dim
        self.nb_hidden_layers = len(hidden_layers_dim)
        self.batch_size=batch_size
        self.epochs = epochs
        self.lr_min = lr_min
        self.lr_max = lr_max
        layers_dim = [self.input_dim]+self.hidden_layers_dim+[self.output_dim]
        len_entrlay=len(layers_dim)-1
        self.W = [np.random.rand(layers_dim[len_entrlay-i],layers_dim[len_entrlay-i-1])/(layers_dim[len_entrlay-i-1]*layers_dim[len_entrlay-i]) for i in range (len_entrlay)]
        self.b = [np.random.rand(layers_dim[len_entrlay-i])/(layers_dim[len_entrlay-i]) for i in range (len_entrlay)]
        
    def get_Weigts(self):
        return ((self.W,self.b))
        
    def fit(self,X,y):
        layers_dim = [self.input_dim]+self.hidden_layers_dim+[self.output_dim]
        len_entrlay=len(layers_dim)-1
        y_len=len(y)
        
        assert y_len == len(X)
        assert len(y[0])==self.output_dim
        assert len(X[0])==self.input_dim
#        loss_epo=[]
        for epo in range (self.epochs):
#            loss = []
            lr=self.lr_max*np.exp(-((1/self.epochs)*np.log(self.lr_max/self.lr_min))*epo)
            print ('Epochs : '+str(epo)+'/'+str(self.epochs))
            print(lr)
            batch_counter=0
            Gr_W = np.flip(np.array([np.zeros((layers_dim[i+1],layers_dim[i])) for i in range (len_entrlay)]),axis=0)
            Gr_b = np.flip(np.array([np.zeros(layers_dim[i+1]) for i in range (len_entrlay)]),axis=0)
            for p in range (y_len):
                x=X[p]
                y_p=y[p]
                z= []
                a= [x]
                for i in range (len_entrlay-1):
                    z_i=np.dot(self.W[len_entrlay-1-i],a[i])+self.b[len_entrlay-1-i]
                    a_i=np.tanh(z_i)
                    z.append(z_i)
                    a.append(a_i)
                z_i=np.dot(self.W[0],a[len_entrlay-1])+self.b[0]
                a_i=1/(1+np.exp(-z_i))
                z.append(z_i)
                a.append(a_i)
                z=np.array(z)
                a=np.array(a)
                gr_b=[np.array([(a[len_entrlay][j]-y_p[j]) for j in range (layers_dim[len_entrlay])])]
                gr_W=[np.array([[a[len_entrlay-1][q]*gr_b[0][j] for q in range(layers_dim[len_entrlay-1])] for j in range (layers_dim[len_entrlay])])]
                for k in range (1,len_entrlay):
                    gr_b_i=np.array([(1-(a[len_entrlay-k][j])**2)*(sum([gr_b[k-1][t]*self.W[k-1][t][j] for t in range (layers_dim[len_entrlay-k+1])])) for j in range (layers_dim[len_entrlay-k])])
                    gr_W_i=np.array([[a[len_entrlay-1-k][q]*gr_b_i[j] for q in range(layers_dim[len_entrlay-1-k])] for j in range (layers_dim[len_entrlay-k])])
                    gr_b.append(gr_b_i)
                    gr_W.append(gr_W_i)    
                Gr_b = Gr_b+ np.array(gr_b)
                Gr_W = Gr_W+ np.array(gr_W)
#                loss.append(-sum([y_p[j]*np.log(a[len(a)-1][j])+(1-y_p[j])*np.log(1-a[len(a)-1][j]) for j in range (self.output_dim)]))
                if (p+1) % self.batch_size == 0 or p+1 == y_len:
                    self.b=self.b -lr*Gr_b
                    self.W=self.W -lr*Gr_W
                    Gr_W = np.flip(np.array([np.zeros((layers_dim[i+1],layers_dim[i])) for i in range (len_entrlay)]),axis=0)
                    Gr_b = np.flip(np.array([np.zeros(layers_dim[i+1]) for i in range (len_entrlay)]),axis=0)
                    batch_counter+=1
#            loss_epo.append(np.mean(np.array(loss)))  
    

    def predict(self,X):
        y_pred=[]
        layers_dim = [self.input_dim]+self.hidden_layers_dim+[self.output_dim]
        len_entrlay=len(layers_dim)-1
        for p in range (len(X)):
            x=X[p]
            z= []
            a= [x]                
            for i in range (len_entrlay-1):
                z_i=np.dot(self.W[len_entrlay-1-i],a[i])+self.b[len_entrlay-1-i]
                a_i=np.tanh(z_i)
                z.append(z_i)
                a.append(a_i)
            z_i=np.dot(self.W[0],a[len_entrlay-1])+self.b[0]
            y_pred.append( 1/(1+np.exp(-z_i)))
        return(y_pred)
