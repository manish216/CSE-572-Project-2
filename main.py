
# coding: utf-8

# In[507]:


import pandas as pd 
import random
from tqdm import tqdm_notebook 
from sklearn.cluster import KMeans # library for calculating clusters
import numpy as np # used for matrix operatyions 
import csv # for importing csv file type
import math # for implementing mathematical operations 
from matplotlib import pyplot as plt
import copy
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard


# In[508]:


IsSynthetic = False
main_Data = pd.read_csv('HumanObserved-Features-Data.csv')
same_Pair = pd.read_csv('same_pairs.csv')
diffn_Pair = pd.read_csv('diffn_pairs.csv')
main_Data_GSC = pd.read_csv('GSC-Features.csv')
same_Pair_GSC = pd.read_csv('same_pairs-GSC.csv')
diffn_Pair_GSC = pd.read_csv('diffn_pairs-GSC.csv')
count = []


# ## Functions

# In[509]:


def Subtractiondataset(DataSet):
    n = DataSet.shape[1]
    n1 = math.ceil(int(((n-3)/2)+1))
    print(n1)
    for x in range (1 , n1):
        DataSet['f'+str(x)]=abs((DataSet[str("f"+str(x)+"_x")])-(DataSet[str("f"+str(x)+"_y")]))
    return DataSet

def variance(Dataset):
    dv = []
    Var =[]
    v= []
    for i in range(0,(Dataset.shape[0])):
        for j in range(0,Dataset.shape[1]):
            v.append(Dataset[i][j])
        vari = np.var(v)
        if (vari == float(0)):
            dv.append(i)
        else:
            Var.append(vari)
    print("Values with variance equal to 0", dv)
    print("variance values",Var)
    return dv,Var 

def Preprocessing(DataSet,trp,vp,tep,C):
    if(C=='F'):
        TR_Len = int(math.ceil(len(DataSet)*(trp*0.01)))
        training= DataSet[:TR_Len] 
        V_Len = int(math.ceil(len(DataSet)*(vp*0.01))) 
        V_End = len(training) + V_Len
        validation = DataSet[(TR_Len+1):V_End,::]
        TE_Len = int(math.ceil(len(DataSet)*(tep*0.01)))
        TE_End = len(training)+len(validation) + TE_Len
        testing = DataSet[(V_End+1):TE_End,::]
    else:
        TR_Len = int(math.ceil(len(DataSet)*(trp*0.01))) 
        training= DataSet[:TR_Len] 
        V_Len = int(math.ceil(len(DataSet)*(vp*0.01))) 
        V_End = len(training) + V_Len
        validation = DataSet[(TR_Len+1):V_End]
        TE_Len = int(math.ceil(len(DataSet)*(tep*0.01)))
        TE_End = len(training)+len(validation) + TE_Len
        testing = DataSet[(V_End+1):TE_End]
    return (training,validation,testing)

def Clustering(TrainingDataSet):
    kmeans = KMeans(n_clusters= M, random_state=0).fit(TrainingDataSet)
    Mu = kmeans.cluster_centers_
    return Mu

def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic): #intializing the fuction 
    print(len(Data))
    BigSigma    = np.zeros((len(Data),len(Data))) # intializing all elements of a matrix to zero (41 x 69623)
    DataT       = np.transpose(Data) # transposing the matrix 69623 X 41
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01)) # caluclating the 80 % of the raw data which becomes thetraining length        
    varVect     = [] # intializing the variance matrix
    for i in range(0,len(DataT[0])): # running the loop from 0 to 69623
        vct = [] # 
        for j in range(0,int(TrainingLen)): # running the loop from 0 to 55699
            vct.append(Data[i][j]) #forming a matrix of 41 X 55699   
        varVect.append(np.var(vct)) #
    
    for j in range(len(Data)):# 41 X 69623
        BigSigma[j][j] = varVect[j]+0.2 # 
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma) # the 200 is the scaling value, which is multiplied by Bigsigma to get a better guassian scale 
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv): 
    R = np.subtract(DataRow,MuRow) # 1 X 41 features, Mu row  1 X 41
    T = np.dot(BigSigInv,np.transpose(R))  #  41 X 41 R is 1 X41 by taking transpose 41 X1
    L = np.dot(R,T) 
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv)) # calucalting the design matrix by calling GetScalar method and from that method we are calling GetScalar
    return phi_x

def inv(m):
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")

    i = np.eye(a, a)
    return np.linalg.lstsq(m, i)[0]
    

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent): # generating the design matrix
    
    DataT = np.transpose(Data) # transposing the raw data matrix where the dimension of the matrix will be 69623 X41
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01)) # calculating a trianing data         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) # intializing the design matrix with dimensions (55699 X 41)
    BigSigInv = np.linalg.inv(BigSigma) # taking the inversion of BigSigma linearly 
    for  C in range(0,len(MuMatrix)): # length of MuMatrix is 10 
        
        for R in range(0,int(TrainingLen)):# length of TrainingLen is 55699
            
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv) # 55699 X10
        
    return PHI

def GetValTest(VAL_PHI,W): 
    Y = np.dot(W,np.transpose(VAL_PHI))
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))

def Gradientdescent(Training_Phi,Validation_Phi,Testing_Phi,HTRTS,HVTS,HTETS,M):
    W =[]
    for x in range(M):
        W.append(random.randint(0,1))

    W_Now        = np.dot(100, W)
    L_Erms_Val   = []
    L_Erms_TR    = []
    R_Erms_Val   = []
    L_Erms_Test  = []
    L_Erms_Test_A =[]
    W_Mat        = []
    
    for i in tqdm_notebook(range(len(HTETS))):
    
        if(i < len(HTRTS)):
            #print ('---------Iteration: ' + str(i) + '--------------')
            Delta_E_D     = -np.dot((HTRTS[i] - np.dot(np.transpose(W_Now),Training_Phi[i])),Training_Phi[i])
            La_Delta_E_W  = np.dot(La,W_Now)
            Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
            Delta_W       = -np.dot(learningRate,Delta_E)
            W_T_Next      = W_Now + Delta_W
            W_Now         = W_T_Next
            #-----------------TrainingData Accuracy---------------------#
            TR_TEST_OUT   = GetValTest(Training_Phi,W_T_Next) 
            Erms_TR       = GetErms(TR_TEST_OUT,HTRTS)
            L_Erms_TR.append(float(Erms_TR.split(',')[1]))
            
            #-----------------ValidationData Accuracy---------------------#
            VAL_TEST_OUT  = GetValTest(Validation_Phi,W_T_Next) 
            Erms_Val      = GetErms(VAL_TEST_OUT,HVTS)
            L_Erms_Val.append(float(Erms_Val.split(',')[1]))
            
            #-----------------TestingData Accuracy---------------------#
            TEST_OUT      = GetValTest(Testing_Phi,W_T_Next) 
            Erms_Test = GetErms(TEST_OUT,HTETS)
            L_Erms_Test.append(float(Erms_Test.split(',')[1]))
            L_Erms_Test_A.append(float(Erms_Test.split(',')[0]))

    count.append(i)
    
    return L_Erms_TR,L_Erms_Val,L_Erms_Test,L_Erms_Test_A,count
   


# ## Hyper parameters

# In[510]:


M                 = 15
TrainingPercent   = 80
validationPercent = 10
TestingPercent    = 10
La                = 40 # lamba is introduced for the reguralization
learningRate      = 0.03


# ## Creation of concatenated Human Observed Set

# In[511]:


joined_data = pd.concat([same_Pair.head(791),diffn_Pair.head(791)],axis =0)
j_1 = pd.merge(joined_data,main_Data,left_on='img_id_A', right_on='img_id')
j_2 = pd.merge(j_1,main_Data,left_on ='img_id_B',right_on ='img_id')
j_final = j_2.drop(['img_id_x','img_id_y','Unnamed: 0_x','Unnamed: 0_y'],axis =1)
j_final=j_final.sample(frac=1).reset_index(drop=True)
print(j_final.shape)
subtraction_Set = copy.deepcopy(j_final)
Neural_network_HC = copy.deepcopy(j_final )
#j_final.to_csv('combined_human_feature_data_C.csv')
j_features = np.asarray(j_final)
j_features = j_features[:,3:]## this dataset contains only the features. D= 293823 X 18 [without randomizing]
H_target = j_final['target']
H_target1 =np.asarray(H_target)
#v =variance(j_features)
#H_target2 =np.asarray(H_target.head(791))
print(H_target1.shape)
#H_target = copy.deepcopy(H_target.head(791))
(HTRFS,HVFS,HTEFS) = Preprocessing(j_features,TrainingPercent,validationPercent,TestingPercent,'F')
(HTRTS,HVTS,HTETS) = Preprocessing(H_target1,TrainingPercent,validationPercent,TestingPercent,'T')
H_Mu = Clustering(HTRFS)
print(HTRFS.shape[0])
#d,v = variance(HTRFS)
#print(d)
#print(v)
BigSigma     = GenerateBigSigma(np.transpose(j_features), H_Mu, TrainingPercent,IsSynthetic)
#d,v =variance(HTRFS)
H_Training_Phi = GetPhiMatrix(np.transpose(j_features), H_Mu, BigSigma, TrainingPercent)
H_Validation_Phi = GetPhiMatrix(np.transpose(HVFS),H_Mu,BigSigma,100)
H_Testing_Phi = GetPhiMatrix(np.transpose(HTEFS),H_Mu,BigSigma,100)
H_Erms_TE =[]

#print(count)


#j_final


# ### Testing

# In[512]:


H_Erms_TR,H_Erms_V,H_Erms_TE,H_Erms_Test_A,count=Gradientdescent(H_Training_Phi,H_Validation_Phi,H_Testing_Phi,HTRTS,HVTS,HTETS,M)


# In[513]:


print("========UBID========")
print("======manishre======")
print("====personNumber====")
print("======50289714======")
print("Dimension of Data frame after merging:        ",j_final.shape)
print("Feature matrix Dimensions:                    ",j_features.shape)
print("Target Vector Dimensions:                     ",H_target.shape)
print("Human Training Feature Matrix Dimensions:     ",HTRFS.shape)
print("Human Validation Feature Matrix Dimensions:   ",HVFS.shape)
print("Human Testing Feature Matrix Dimensions:      ",HTEFS.shape)
print("===============================================================")
print("Dimensions Mu Matrix:                         ",H_Mu.shape)
print("Dimensions of BigSigma Matrix:                ",BigSigma.shape)
print("Dimensions of Training Phi:                   ",H_Training_Phi.shape)
print("Dimensions of Validation Phi:                 ",H_Validation_Phi.shape)
print("Dimensions of Testing Phi:                    ",H_Testing_Phi.shape)
print("==================================================================")
print ("E_rms Training   = " + str(np.around(min(H_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(H_Erms_V),5)))
print ("E_rms Testing    = " + str(np.around(min(H_Erms_TE),5)))
print ("E_rms Accuracy    = "+ str(np.around(max(H_Erms_Test_A),5)))

print("==================================================================")


# ## Creation of Subtraction Human Observed Set

# In[514]:


j_final_s = copy.deepcopy(subtraction_Set)
print(j_final_s.shape[1])
j_final_s1 = Subtractiondataset(j_final_s)
for i in range(1,10):
    j_final_s1 = j_final_s1.drop(['f'+str(i)+'_x','f'+str(i)+'_y'],axis =1)


Neural_network_HS = copy.deepcopy(j_final_s1) 
HS_target = j_final_s1['target']

HS_target = np.asarray(HS_target)
jsf = np.asarray(j_final_s1)
jsf1 = jsf[:,3:]
#(vhs,vts) =variance(jsf,HS_target)


# In[515]:


print('Thefeature matrix ',jsf1.shape)
print('the target vector',HS_target.shape)
(HTRFSS,HVFSS,HTEFSS) = Preprocessing(jsf1,TrainingPercent,validationPercent,TestingPercent,'F')
(HTRTSS,HVTSS,HTETSS) = Preprocessing(HS_target,TrainingPercent,validationPercent,TestingPercent,'T')
H_Mu = Clustering(HTRFSS)
BigSigma     = GenerateBigSigma(np.transpose(jsf1), H_Mu, TrainingPercent,IsSynthetic)
H_Training_Phi = GetPhiMatrix(np.transpose(HS_target), H_Mu, BigSigma, TrainingPercent)
H_Validation_Phi = GetPhiMatrix(np.transpose(HVFSS),H_Mu,BigSigma,100)
H_Testing_Phi = GetPhiMatrix(np.transpose(HTEFSS),H_Mu,BigSigma,100)


# In[516]:



H_Erms_TR,H_Erms_V,H_Erms_TE,HS_Erms_Test_A,count=Gradientdescent(H_Training_Phi,H_Validation_Phi,H_Testing_Phi,HTRTSS,HVTSS,HTETSS,M)
print('completed')    


# In[517]:


print("========UBID========")
print("======manishre======")
print("====personNumber====")
print("======50289714======")
print("Dimension of Data frame after merging:        ",j_final_s1.shape)
print("Feature matrix Dimensions:                    ",jsf1.shape)
print("Target Vector Dimensions:                     ",HS_target.shape)
print("Human Training Feature Matrix Dimensions:     ",HTRFSS.shape)
print("Human Validation Feature Matrix Dimensions:   ",HVFSS.shape)
print("Human Testing Feature Matrix Dimensions:      ",HTEFSS.shape)
print("===============================================================")
print("Dimensions Mu Matrix:                         ",H_Mu.shape)
print("Dimensions of BigSigma Matrix:                ",BigSigma.shape)
print("Dimensions of Training Phi:                   ",H_Training_Phi.shape)
print("Dimensions of Validation Phi:                 ",H_Validation_Phi.shape)
print("Dimensions of Testing Phi:                    ",H_Testing_Phi.shape)
print("==================================================================")
print ("E_rms Training   = " + str(np.around(min(H_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(H_Erms_V),5)))
print ("E_rms Testing    = " + str(np.around(min(H_Erms_TE),5)))
print ("E_rms Accuracy    = "+ str(np.around(max(HS_Erms_Test_A),5)))


# In[518]:


main_Data_GSC = pd.read_csv('GSC-Features.csv')
same_Pair_GSC = pd.read_csv('same_pairs-GSC.csv')
diffn_Pair_GSC = pd.read_csv('diffn_pairs-GSC.csv')


# ## Creation of GSC Concatenated Dataset

# In[519]:


joined_data_GSC = pd.concat([same_Pair_GSC.head(512),diffn_Pair_GSC.head(512)],axis =0)
j_1_GSC = pd.merge(joined_data_GSC,main_Data_GSC,left_on='img_id_A', right_on='img_id')
j_2_GSC= pd.merge(j_1_GSC,main_Data_GSC,left_on ='img_id_B',right_on ='img_id')
j_final_GSC = j_2_GSC.drop(['img_id_x','img_id_y'],axis =1)
j_final_GSC_s=j_final_GSC.sample(frac=1).reset_index(drop=True)
subtraction_set_GSC =copy.deepcopy(j_final_GSC_s)
Neural_network_GSC = copy.deepcopy(j_final_GSC_s)
#j_final_GSC_s.to_csv('trail.csv')
#j_final.to_csv('combined_human_feature_data_C.csv')
##======================================================
j_features_GSC = np.asarray(j_final_GSC_s)
j_features_GSC1 = j_features_GSC[:,3:]## this dataset contains only the features. D= 293823 X 18 [without randomizing]
G_target = j_final_GSC_s['target']
G_target1 =np.asarray(G_target)
#v,t =variance(j_features_GSC[:,3:],G_target1)
(GTRFS,GVFS,GTEFS) = Preprocessing(j_features_GSC1,TrainingPercent,validationPercent,TestingPercent,'F')
print("Feature Processing Completed")
(GTRTS,GVTS,GTETS) = Preprocessing(G_target1,TrainingPercent,validationPercent,TestingPercent,'T')
print("Target Processing Completed")
G_Mu = Clustering(GTRFS)
print("clustering Finished")
BigSigma     = GenerateBigSigma(np.transpose(j_features_GSC1), G_Mu, TrainingPercent,IsSynthetic)

#print(BigSigma)
print("Bigsigma Matrix generation finished")
print("==Process for generating phi Matrix started==")
print("====Please wait the generation of Phi matrix takes time===")
G_Training_Phi = GetPhiMatrix(np.transpose(j_features_GSC1), G_Mu, BigSigma, TrainingPercent)

print("Training Phi Matrix generated")
G_Validation_Phi = GetPhiMatrix(np.transpose(GVFS),G_Mu,BigSigma,100)
print("validation Phi Matrix generated")
G_Testing_Phi = GetPhiMatrix(np.transpose(GTEFS),G_Mu,BigSigma,100)
print("Testing Phi Matrix generated")

#print(count)


# ## Testing

# In[520]:


G_Erms_TR,G_Erms_V,G_Erms_TE,GSC_Erms_Test_A,count=Gradientdescent(G_Training_Phi,G_Validation_Phi,G_Testing_Phi,GTRTS,GVTS,GTETS,M)


# In[521]:


print("========UBID========")
print("======manishre======")
print("====personNumber====")
print("======50289714======")
print("Dimension of Data frame after merging:        ",j_final_GSC.shape)
print("Feature matrix Dimensions:                    ",j_features_GSC1.shape)
print("Target Vector Dimensions:                     ",G_target1.shape)
print("Human Training Feature Matrix Dimensions:     ",GTRFS.shape)
print("Human Validation Feature Matrix Dimensions:   ",GVFS.shape)
print("Human Testing Feature Matrix Dimensions:      ",GTEFS.shape)
print("Human Training Target Matrix Dimensions:     ",GTRTS.shape)
print("Human Validation Target Matrix Dimensions:   ",GVTS.shape)
print("Human Testing Target Matrix Dimensions:      ",GTETS.shape)
print("===============================================================")
print("Dimensions Mu Matrix:                         ",G_Mu.shape)
print("Dimensions of BigSigma Matrix:                ",BigSigma.shape)
print("Dimensions of Training Phi:                   ",G_Training_Phi.shape)
print("Dimensions of Validation Phi:                 ",G_Validation_Phi.shape)
print("Dimensions of Testing Phi:                    ",G_Testing_Phi.shape)
print("==================================================================")
print ("E_rms Training   = " + str(np.around(min(G_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(G_Erms_V),5)))
print ("E_rms Testing    = " + str(np.around(min(G_Erms_TE),5)))
print ("E_rms Accuracy    = "+ str(np.around(max(GSC_Erms_Test_A),5)))


# ## Creation of GSC subtraction Data set
# 

# In[522]:


j_final_sub = copy.deepcopy(subtraction_set_GSC)
print(j_final_sub.shape)
j_final_sub1 = Subtractiondataset(j_final_sub)

for i in range(1,513):
    j_final_sub1 = j_final_sub1.drop(['f'+str(i)+'_x','f'+str(i)+'_y'],axis =1)

Neural_network_GSS = copy.deepcopy(j_final_sub1)
GS_target = j_final_sub1['target']

GS_target = np.asarray(GS_target)
jssf = np.asarray(j_final_sub1)
jssf1 = jssf[:,3:]

    


# In[523]:


print('Thefeature matrix ',jssf1.shape)
print('the target vector',GS_target.shape)
(GSTRFSS,GSVFSS,GSTEFSS) = Preprocessing(jssf1,TrainingPercent,validationPercent,TestingPercent,'F')
(GSTRTSS,GSVTSS,GSTETSS) = Preprocessing(GS_target,TrainingPercent,validationPercent,TestingPercent,'T')
GS_Mu = Clustering(GSTRFSS)
BigSigma     = GenerateBigSigma(np.transpose(jssf1), GS_Mu, TrainingPercent,IsSynthetic)
GS_Training_Phi = GetPhiMatrix(np.transpose(GS_target), GS_Mu, BigSigma, TrainingPercent)
GS_Validation_Phi = GetPhiMatrix(np.transpose(GSVFSS),GS_Mu,BigSigma,100)
GS_Testing_Phi = GetPhiMatrix(np.transpose(GSTEFSS),GS_Mu,BigSigma,100)


# In[524]:



GS_Erms_TR,GS_Erms_V,GS_Erms_TE,GSCS_Erms_Test_A,count = Gradientdescent(GS_Training_Phi,GS_Validation_Phi,GS_Testing_Phi,GSTRTSS,GSVTSS,GSTETSS,M)
print('completed')  


# In[525]:


print("========UBID========")
print("======manishre======")
print("====personNumber====")
print("======50289714======")
print("Dimension of Data frame after merging:        ",j_final_sub1.shape)
print("Feature matrix Dimensions:                    ",jssf1.shape)
print("Target Vector Dimensions:                     ",GS_target.shape)
print("Human Training Feature Matrix Dimensions:     ",GSTRFSS.shape)
print("Human Validation Feature Matrix Dimensions:   ",GSVFSS.shape)
print("Human Testing Feature Matrix Dimensions:      ",GSTEFSS.shape)
print("===============================================================")
print("Dimensions Mu Matrix:                         ",GS_Mu.shape)
print("Dimensions of BigSigma Matrix:                ",BigSigma.shape)
print("Dimensions of Training Phi:                   ",GS_Training_Phi.shape)
print("Dimensions of Validation Phi:                 ",GS_Validation_Phi.shape)
print("Dimensions of Testing Phi:                    ",GS_Testing_Phi.shape)
print("==================================================================")
print ("E_rms Training   = " + str(np.around(min(GS_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(GS_Erms_V),5)))
print ("E_rms Testing    = " + str(np.around(min(GS_Erms_TE),5)))
print ("E_rms Accuracy    = "+ str(np.around(max(GSCS_Erms_Test_A),5)))


# ## Logistic Regression 

# ## Functions

# In[526]:


def sigmoid(x):
    return 1/(1+np.exp(-x))
def gradientdescent_L(features,target,learningrate,W):
    
    for y in range(0,1000):
        z = np.asarray(np.dot(features,np.transpose(W)),dtype =np.float32)
        h = sigmoid(z)
        gradient = np.dot(np.transpose(features),(h-target))/target.size
        W -= learningrate * gradient
        if(y %100000 ==0):
            z = np.asarray(np.dot(features,W),dtype=np.float32)
            h =sigmoid(z)
            loss = (-target * np.log(h) - (1 - target) * np.log(1 - h)).mean()
        
    return W,loss
def predict_prob(X,weights):
    return sigmoid(np.asarray(np.dot(X, weights),dtype=np.float32))
    
def predict(features,weights,threshold):
    predictval = predict_prob(features,weights)
    for i in range(len(predictval)):
        
        if (predictval[i]>threshold):
            predictval[i] =1
        else:
            predictval[i] =0
    return predictval
def accuracy(target,predicted):
    if(len(target)==len(predicted)):
        correct =[]
        wrong =[]
        for i in range(0,len(predicted)):
            if (predicted[i]==target[i]):
                correct.append(i)
            else:
                wrong.append(i)
    else:
        print('error')
    accuracy = (len(correct)/len(target))*100
    return accuracy


# ## Hyperparameters

# In[527]:


learningRate = 0.05
threshold =0.6


# ### Logistic Regression for Human Concatenated Dataset 

# In[528]:


Features = HTRFS #,HVFS,HTEFS
Target = HTRTS #HTRTS,HVTS,HTETS
W=[]
for x in range(Features.shape[1]):
    W.append(0)
w,hclt =gradientdescent_L(Features,Target,learningRate,W)

vw,hclv = gradientdescent_L(HVFS,HVTS,learningRate,w)

predictval = predict(HTEFS,vw,threshold)
acc = accuracy(HTETS,predictval)


# In[529]:


print("========UBID========")
print("======manishre======")
print("====personNumber====")
print("======50289714======")
print('Loss for training set',hclt)
print('Loss for validation set',hclv)
print('Accuracy is:'+str(math.floor(acc))+'%')


# ### Logistic Regression for Human Subtracction Dataset

# In[530]:


Features = HTRFSS #HTRFSS,HVFSS,HTEFSS
Target = HTRTSS #HTRTSS,HVTSS,HTETSS
W=[]
for x in range(Features.shape[1]):
    W.append(0)
w_s,hclt_s =gradientdescent_L(Features,Target,learningRate,W)

vw_s,hclv_s = gradientdescent_L(HVFSS,HVTSS,learningRate,w_s)

predictval_s = predict(HTEFSS,vw_s,threshold)
acc_s = accuracy(HTETSS,predictval_s)

        


# In[531]:


print("========UBID========")
print("======manishre======")
print("====personNumber====")
print("======50289714======")
print('Loss for training set',hclt_s)
print('Loss for validation set',hclv_s)
print('Accuracy is:'+str(math.floor(acc_s))+'%')


# ### Logistic Regression for GSC concatenated Dataset

# In[532]:


Features = GTRFS #GTRFS,GVFS,GTEFS
Target = GTRTS #GTRTS,GVTS,GTETS
W=[]
for x in range(Features.shape[1]):
    W.append(0)
Gw,Ghclt =gradientdescent_L(Features,Target,learningRate,W)

Gvw,Ghclv = gradientdescent_L(GVFS,GVTS,learningRate,Gw)

Gpredictval = predict(GTEFS,Gvw,threshold)
Gacc = accuracy(GTETS,Gpredictval)


# In[533]:


print("========UBID========")
print("======manishre======")
print("====personNumber====")
print("======50289714======")
print('Loss for training set',Ghclt)
print('Loss for validation set',Ghclv)
print('Accuracy is:'+str(math.floor(Gacc))+'%')


# ### Logistic Regression for GSC subtracted Dataset

# In[534]:


Features = GSTRFSS #GSTRFSS,GSVFSS,GSTEFSS
Target = GSTRTSS #GSTRTSS,GSVTSS,GSTETSS
W=[]
for x in range(Features.shape[1]):
    W.append(0)
Gw_S,Ghclt_S =gradientdescent_L(Features,Target,learningRate,W)

Gvw_S,Ghclv_S = gradientdescent_L(GSVFSS,GSVTSS,learningRate,Gw_S)

Gpredictval_S = predict(GSTEFSS,Gvw_S,threshold)
Gacc_S = accuracy(GSTETSS,Gpredictval_S)


# In[535]:


print("========UBID========")
print("======manishre======")
print("====personNumber====")
print("======50289714======")
print('Loss for training set',Ghclt_S)
print('Loss for validation set',Ghclv_S)
print('Accuracy is:'+str(math.floor(Gacc_S))+'%')


# # Neural Networks

# ### Function

# In[536]:


def Targetclassifier1(value):
    if(value['target']==0):
        return 1
    else:
        return 0
def Targetclassifier2(value):
    if(value['target']==1):
        return 1
    else:
        return 0
    
def PreprocessingN(DataSet,trp,vp,C):
    if(C=='F'):
        TR_Len = int(math.ceil(len(DataSet)*(trp*0.01)))
        training= DataSet[:TR_Len] 
        V_Len = int(math.ceil(len(DataSet)*(vp*0.01))) 
        V_End = len(training) + V_Len
        validation = DataSet[(TR_Len+1):V_End,::]
    else:
        TR_Len = int(math.ceil(len(DataSet)*(trp*0.01))) 
        training= DataSet[:TR_Len] 
        V_Len = int(math.ceil(len(DataSet)*(vp*0.01))) 
        V_End = len(training) + V_Len
        validation = DataSet[(TR_Len+1):V_End]
    return (training,validation)


def get_model(x1):
    print(x1)
    input_size = x1
    drop_out = 0.2
    first_dense_layer_nodes  = 256
    second_dense_layer_nodes = 2
    
    model = Sequential() 
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))

    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
   
    
    model.summary()
    
   
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def runModel(processedData,processedLabel,validation_split,epochs,tb_batch_size,model_batch_size,early_patience):
    
    tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')


    history = model.fit(processedData
                        , processedLabel
                        , validation_split=validation_data_split
                        , epochs=num_epochs
                        , batch_size=model_batch_size
                        , callbacks = [tensorboard_cb,earlystopping_cb]
                       )
    return history


# ### HyperParameters

# In[537]:


testing_percent =10
validation_data_split = 0.2
num_epochs = 10000
model_batch_size = 200
tb_batch_size = 50
early_patience = 1000


# ### Creation of Human Concatenation Dataset

# In[567]:


Human_concatenated = copy.deepcopy(Neural_network_HC)
Human_concatenated['T1'] = Human_concatenated.apply(lambda row: Targetclassifier1(row),axis =1)
Human_concatenated['T2'] = Human_concatenated.apply(lambda row: Targetclassifier2(row),axis =1)
Human_concatenated1 = Human_concatenated['T1']
Human_concatenated2 = Human_concatenated['T2']
Human_concatenated_neural = pd.concat([Human_concatenated1,Human_concatenated2],axis =1)
Human_neural_features =np.asarray(Human_concatenated)
print(Human_neural_features.shape)
Human_neural_features = Human_neural_features[:,3:21]
Human_neural_features_c,Human_neural_testing = PreprocessingN(Human_neural_features,80,10,'F')
Human_neural_ftarget_c,Human_neural_tetarget = PreprocessingN(Human_concatenated_neural,80,10,'T')
print('Dimensions of training feature Dataset:',Human_neural_features_c.shape)
print('Dimensions of testing feature Dataset :',Human_neural_testing.shape)
print('Dimensions of training target set     :',Human_neural_ftarget_c.shape)
print('Dimensions of testing target set      :',Human_neural_tetarget.shape)


# In[568]:


model = get_model(len(np.transpose(Human_neural_features_c)))


# ### Run Model

# In[569]:



history = runModel(processedData = Human_neural_features_c,
         processedLabel = Human_neural_ftarget_c,
         validation_split = validation_data_split,
         epochs = num_epochs,
         tb_batch_size = tb_batch_size,
         model_batch_size = model_batch_size,
         early_patience = early_patience)


# In[570]:


get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))


# ### Testing 

# In[543]:


accuracy_HC = model.evaluate(Human_neural_testing,Human_neural_tetarget,verbose=0)
print("========UBID========")
print("======manishre======")
print("====personNumber====")
print("======50289714======")
print("Test Loss:",accuracy_HC[0])
print("Test Accuracy:",accuracy_HC[1])
opdf = pd.DataFrame(accuracy_HC)


# ###  Creation of Data for Human Subtracted Data Set

# In[544]:


Human_concatenated_S = copy.deepcopy(Neural_network_HS)
Human_concatenated_S['T1'] = Human_concatenated_S.apply(lambda row: Targetclassifier1(row),axis =1)
Human_concatenated_S['T2'] = Human_concatenated_S.apply(lambda row: Targetclassifier2(row),axis =1)
Human_concatenated_S1 = Human_concatenated_S['T1']
Human_concatenated_S2 = Human_concatenated_S['T2']
Human_concatenated_neural_S = pd.concat([Human_concatenated_S1,Human_concatenated_S2],axis =1)
Human_neural_features_S =np.asarray(Human_concatenated_S)
print(Human_neural_features_S.shape)
Human_neural_features_S = Human_neural_features_S[:,3:12]
Human_neural_features_sc,Human_neural_testing_sc = PreprocessingN(Human_neural_features_S,80,10,'F')
Human_neural_ftarget_sc,Human_neural_tetarget_sc = PreprocessingN(Human_concatenated_neural_S,80,10,'T')
print('Dimensions of training feature Dataset:',Human_neural_features_sc.shape)
print('Dimensions of testing feature Dataset :',Human_neural_testing_sc.shape)
print('Dimensions of training target set     :',Human_neural_ftarget_sc.shape)
print('Dimensions of testing target set      :',Human_neural_tetarget_sc.shape)


# ### Get Model

# In[545]:


model = get_model(len(np.transpose(Human_neural_features_S)))


# ### Run Model

# In[546]:


history = runModel(processedData = Human_neural_features_sc,
         processedLabel = Human_neural_ftarget_sc,
         validation_split = validation_data_split,
         epochs = num_epochs,
         tb_batch_size = tb_batch_size,
         model_batch_size = model_batch_size,
         early_patience = early_patience)


# In[547]:


get_ipython().run_line_magic('matplotlib', 'inline')
df1 = pd.DataFrame(history.history)
df1.plot(subplots=True, grid=True, figsize=(10,15))


# ### Testing

# In[548]:


accuracy_HS = model.evaluate(Human_neural_testing_sc,Human_neural_tetarget_sc,verbose=0)
print("========UBID========")
print("======manishre======")
print("====personNumber====")
print("======50289714======")
print("Test Loss:",accuracy_HS[0])
print("Test Accuracy:",accuracy_HS[1])
opdf = pd.DataFrame(accuracy_HS)


# ## Creation of GSC concatenated Dataset

# In[549]:


Human_concatenated_GSC = copy.deepcopy(Neural_network_GSC)
Human_concatenated_GSC['T1'] = Human_concatenated_GSC.apply(lambda row: Targetclassifier1(row),axis =1)
Human_concatenated_GSC['T2'] = Human_concatenated_GSC.apply(lambda row: Targetclassifier2(row),axis =1)
Human_concatenated_GSC1 = Human_concatenated_GSC['T1']
Human_concatenated_GSC2 = Human_concatenated_GSC['T2']
Human_concatenated_neural_GSC = pd.concat([Human_concatenated_GSC1,Human_concatenated_GSC2],axis =1)
Human_neural_features_GSC =np.asarray(Human_concatenated_GSC)
print(Human_neural_features_GSC.shape)
Human_neural_features_GSC = Human_neural_features_GSC[:,3:1027]
Human_neural_features_gsc,Human_neural_testing_gsc = PreprocessingN(Human_neural_features_GSC,80,10,'F')
Human_neural_ftarget_gsc,Human_neural_tetarget_gsc = PreprocessingN(Human_concatenated_neural_GSC,80,10,'T')
print('Dimensions of training feature Dataset:',Human_neural_features_gsc.shape)
print('Dimensions of testing feature Dataset :',Human_neural_testing_gsc.shape)
print('Dimensions of training target set     :',Human_neural_ftarget_gsc.shape)
print('Dimensions of testing target set      :',Human_neural_tetarget_gsc.shape)


# In[550]:


Human_concatenated_GSC


# ### Get Model

# In[551]:


model = get_model(len(np.transpose(Human_neural_features_GSC)))


# ### Run Model

# In[552]:


history = runModel(processedData = Human_neural_features_gsc,
         processedLabel = Human_neural_ftarget_gsc,
         validation_split = validation_data_split,
         epochs = num_epochs,
         tb_batch_size = tb_batch_size,
         model_batch_size = model_batch_size,
         early_patience = early_patience)


# In[553]:


get_ipython().run_line_magic('matplotlib', 'inline')
df1 = pd.DataFrame(history.history)
df1.plot(subplots=True, grid=True, figsize=(10,15))


# ### Testing

# In[554]:


accuracy_GC = model.evaluate(Human_neural_testing_gsc,Human_neural_tetarget_gsc,verbose=0)
print("========UBID========")
print("======manishre======")
print("====personNumber====")
print("======50289714======")
print("Test Loss:",accuracy_GC[0])
print("Test Accuracy:",accuracy_GC[1])
opdf = pd.DataFrame(accuracy_GC)


# ## Creation of GSC subtraction Dataset 

# In[555]:


Human_concatenated_GSC_S = copy.deepcopy(Neural_network_GSS)
Human_concatenated_GSC_S['T1'] = Human_concatenated_GSC_S.apply(lambda row: Targetclassifier1(row),axis =1)
Human_concatenated_GSC_S['T2'] = Human_concatenated_GSC_S.apply(lambda row: Targetclassifier2(row),axis =1)
Human_concatenated_GSC_S1 = Human_concatenated_GSC_S['T1']
Human_concatenated_GSC_S2 = Human_concatenated_GSC_S['T2']
Human_concatenated_neural_GSC_S = pd.concat([Human_concatenated_GSC_S1,Human_concatenated_GSC_S2],axis =1)
Human_neural_features_GSC_S =np.asarray(Human_concatenated_GSC_S)
print(Human_neural_features_GSC_S.shape)
Human_neural_features_GSC_S = Human_neural_features_GSC_S[:,3:515]
Human_neural_features_gscs,Human_neural_testing_gscs = PreprocessingN(Human_neural_features_GSC_S,80,10,'F')
Human_neural_ftarget_gscs,Human_neural_tetarget_gscs = PreprocessingN(Human_concatenated_neural_GSC_S,80,10,'T')
print('Dimensions of training feature Dataset:',Human_neural_features_gscs.shape)
print('Dimensions of testing feature Dataset :',Human_neural_testing_gscs.shape)
print('Dimensions of training target set     :',Human_neural_ftarget_gscs.shape)
print('Dimensions of testing target set      :',Human_neural_tetarget_gscs.shape)


# ### Get Model

# In[556]:


model = get_model(len(np.transpose(Human_neural_features_GSC_S)))


# ### Run Model

# In[557]:


history = runModel(processedData = Human_neural_features_gscs,
         processedLabel = Human_neural_ftarget_gscs,
         validation_split = validation_data_split,
         epochs = num_epochs,
         tb_batch_size = tb_batch_size,
         model_batch_size = model_batch_size,
         early_patience = early_patience)


# In[558]:


get_ipython().run_line_magic('matplotlib', 'inline')
df1 = pd.DataFrame(history.history)
df1.plot(subplots=True, grid=True, figsize=(10,15))


# ### Testing

# In[559]:


accuracy_GCS = model.evaluate(Human_neural_testing_gscs,Human_neural_tetarget_gscs,verbose=0)
print("========UBID========")
print("======manishre======")
print("====personNumber====")
print("======50289714======")
print("Test Loss:",accuracy_GCS[0])
print("Test Accuracy:",accuracy_GCS[1])
opdf = pd.DataFrame(accuracy_GCS)

