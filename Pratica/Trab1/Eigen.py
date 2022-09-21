import numpy as np
import cv2
import math 
from math import atan2,degrees
from numpy import linalg as LA
import os

pathimage=".\\faceDB_20_21\\originais\\"

path=".\\haarcascades\\"

face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(path +'haarcascade_eye.xml')

TrainPath=".\\faceDB_20_21\\training\\"

TestPath=".\\faceDB_20_21\\test\\"

def Train(imagearr):
    facesarr=[]
    for i in imagearr:
         pathimg=TrainPath+i
         faces=cv2.imread(pathimg)
         facesgray = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
         facesarr.append(facesgray)
    return facesarr   
 
def Test(imagearr):
    facesarr=[]
    for i in imagearr:
         pathimg=TestPath+i
         faces=cv2.imread(pathimg)
         facesgray = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
         facesarr.append(facesgray)
    return facesarr   

        
def check_eyes(eyes,facerecognition):
    facey=facerecognition[0,1]
    facew=facerecognition[0,3]
    eyesrec=[]
    for (x,y,w,h) in eyes:
        eyecenter=(y+h)/2
        if(y<facey+facew/3 and y>facey):
            eyesrec.append([x,y,w,h])
    return eyesrec

def flatten(facesarr):
    FlattenedArr=np.empty((2576,1))
    for img in facesarr:
        img_matrix=np.matrix(img)
        img_flattened=img_matrix.flatten()
        img_flattened_transposed=img_flattened.transpose()
        FlattenedArr=np.concatenate((FlattenedArr,img_flattened_transposed),axis=1) #B=[X1|X2...Xn]
    FlattenedArr=np.delete(FlattenedArr,0,1)
    return FlattenedArr

def calculate_meanFace(facesarr):
    FlattenedArr=flatten(facesarr)
    MeanArr=np.mean(FlattenedArr,axis=1)
    return MeanArr

def apply_object(face,scale):
    pathObject=".\\Objects\\"
    obj=cv2.imread(pathObject+"Glasses.png")
    mask=np.all(obj==[0,0,0],axis=-1).astype(int)
    #mask_inv = cv2.bitwise_not(mask)
    [h,w]=obj[:,:,1].shape
    scale=15/w
    scaledsize=(int(w*scale),int(h*scale))
    objresized=cv2.resize(mask,scaledsize,interpolation=cv2.INTER_NEAREST)
    translation_matrix=np.float32([[1,0,14],[0,1,22]])
    
    obj_translation = cv2.warpAffine(objresized.astype(np.uint8), translation_matrix,(46,56))
    
    face_obj=face+obj_translation
    

def imagenormalize(imagearr):
    facesarr=[]
    for i in imagearr:
        pathimg=pathimage+i
        faces=cv2.imread(pathimg) # Read image 
        facesgray = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
        facerecognition = face_cascade.detectMultiScale(facesgray)
        
        eyes = eye_cascade.detectMultiScale(facesgray) # (ex,ey,ew,eh)
        eyesrec=check_eyes(eyes,facerecognition)
        
        lefteyeCenter=(int(eyesrec[0][0]+eyesrec[0][2]/2), int(eyesrec[0][1]+eyesrec[0][3]/2), 1)
        righteyeCenter=(int(eyesrec[1][0]+eyesrec[1][2]/2), int(eyesrec[1][1]+eyesrec[1][3]/2), 1)
        
        xDiff = righteyeCenter[0] - lefteyeCenter[0]
        yDiff = righteyeCenter[1] - lefteyeCenter[1]
        angle=degrees(atan2(yDiff, xDiff))
        print("Original angle: " + str(angle))
        
        #cv2.line(faces, lefteyeCenter, righteyeCenter, (0, 255, 0), 1)
        
        
        image_center = tuple(np.array(faces.shape[1::-1]) / 2)

        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(faces, rot_mat, faces.shape[1::-1], flags=cv2.INTER_LINEAR)
        startpoint = np.asarray([lefteyeCenter, righteyeCenter])

        

        newpoint = np.dot(rot_mat, startpoint.T);

        
        newLefteyeCenter = (newpoint[0][0], newpoint[1][0])
        newRighteyeCenter = (newpoint[0][1], newpoint[1][1])
        
        newXDiff = newRighteyeCenter[0] - newLefteyeCenter[0]
        newYDiff = newRighteyeCenter[1] - newLefteyeCenter[1]
        
        newAngle=degrees(atan2(newYDiff, newXDiff))
        
        print("New angle: " + str(newAngle))

        nle = (int(newpoint[0][0]), int(newpoint[1][0]))
        nre = (int(newpoint[0][1]), int(newpoint[1][1]))
        
        eyedist= nre[0]-nle[0]
        scaling=15/eyedist 
        
        nlescaled=[int(nle[0]*scaling),int(nle[1]*scaling)]
        nrescaled=[int(nre[0]*scaling),int(nre[1]*scaling)]
        
        newsize=(int(result.shape[1]*scaling),int(result.shape[0]*scaling))
        
        imgresized=cv2.resize(result,newsize,interpolation=cv2.INTER_NEAREST)
        
        x_t=16-nlescaled[0]
        y_t=24-nlescaled[1]
        translation_matrix=np.float32([[1,0,x_t],[0,1,y_t]])
        img_translation = cv2.warpAffine(imgresized, translation_matrix, (newsize[1],newsize[0]))
        
        img_translation_gray = cv2.cvtColor(img_translation, cv2.COLOR_BGR2GRAY)
        
        img_normalized = img_translation_gray[0:56, 0:46]
        
        np.append(facesarr,img_normalized)
    return facesarr

def eigen_face(facesarr,MeanArr,mvalues):
    #Flatten arrays
    FlattenedArr=flatten(facesarr)
    A=np.matrix(FlattenedArr-MeanArr)
    R=A.transpose()*A
    values,vector=LA.eig(R)
    idx_arr=np.argsort(values)[::-1][:mvalues]
    sz=len(facesarr)
    V=np.empty((sz,1))
    for idx in idx_arr:
        vec=vector[:,idx]
        V=np.concatenate((V,vec),axis=1)
    V=np.delete(V,0,1)
    W=A*V # Matriz W ortogonal mas não normalizada
    norm=np.linalg.norm(W, axis=0) #Calcular a norma da coluna de W
    W_norm=W/norm # W=[n×m] norm=[n*1]
    Y=W_norm.transpose()*A
    return Y,W_norm

def train_Knn(Knn,Yarr,Label):
    Yarr = np.array(Yarr).reshape(-1, 1)
    Knn.train(np.float32(Yarr),cv2.ml.ROW_SAMPLE, np.float32(Label))
    

def classify_Faces(Knn,Y,K):
   #Classify faces
    Y = np.array(Y).reshape(-1, 1)
    ret, res, neighbours, distance = Knn.findNearest(Y, K)
    return ret, res, neighbours, distance



def FisherFaces(FacesArr,classMean, totalMean, facesPerclass ,numberOfclasses):
    #class_mean -> média da face das classes com a media de cada em cada posição
    Y,W=eigen_face(FacesArr,totalMean,facesPerclass*numberOfclasses-numberOfclasses)
    
    Sw =np.zeros((2576 , 1))
    flattenedArr=flatten(FacesArr)
    for face in flattenedArr.T:
        face_col=face.T
        Sw=Sw+ np.dot(face_col-classMean,(face_col-classMean).T)
    Sb =np.zeros((2576 , 1))
    for class_mean in classMean.T:
        class_mean=class_mean.T
        Sb=Sb+ facesPerclass*np.dot(classMean-totalMean,(classMean-totalMean).T)
    SbTil=W.T*Sb*W
    SwTil=W.T*Sw*W
    SwTilInverse=LA.inv( SwTil )
    values,vector=LA.eig(SwTilInverse*SbTil)
    idx_arr=np.argsort(values)[::-1][:numberOfclasses-1]
    sz=facesPerclass*numberOfclasses-numberOfclasses
    Yfld=np.empty((sz,1))
    for idx in idx_arr:
        vec=vector[:,idx]
        Yfld=np.concatenate((Yfld,vec),axis=1)
    Yfld=np.delete(Yfld,0,1)
    return Yfld


def main():
    Knn = cv2.ml.KNearest_create()
    imagearr1=["angelaMerkel1.jpg","angelaMerkel2.jpg","angelaMerkel3.jpg","angelaMerkel4.jpg","angelaMerkel5.jpg","angelaMerkel6.jpg"]
    imagearr2=["angelinaJolie1.jpg","angelinaJolie2.jpg","angelinaJolie3.jpg","angelinaJolie4.jpg","angelinaJolie5.jpg","angelinaJolie6.jpg"]
    imagearr3=["baracObama1.jpg","baracObama2.jpg","baracObama3.jpg","baracObama4.jpg","baracObama5.jpg","baracObama6.jpg"]
    
   
    arr1=Train(imagearr1)
    arr2=Train(imagearr2)
    arr3=Train(imagearr3)

    imgs=[arr1,arr2,arr3]
    
    fullfaces=[]
    #for arr in imgs:
    for arr in imgs:        
        fullfaces=fullfaces+arr
    totalMean = calculate_meanFace(fullfaces)
    mvalues=len(arr1)-1
    Y1,W=eigen_face(arr1,totalMean,mvalues)
    Y2,W=eigen_face(arr2,totalMean,mvalues)
    Y3,W=eigen_face(arr3,totalMean,mvalues)
    
    
    i=0
    label=np.zeros((1,1))
    for arr in imgs:
       label=np.concatenate((label,np.full((1,mvalues*len(arr)),i)),axis=1)
       i+=1
    label=np.delete(label,0,1)
    train_Knn(Knn,[Y1,Y2,Y3],label)
    
    imagearr4=["baracObama7.jpg","baracObama8.jpg","baracObama9.jpg"]
    arr4=Test(imagearr4)
    Y4,W=eigen_face(arr4,totalMean,len(arr4)-1)
    ret, res, neighbours, distance=classify_Faces(Knn,np.float32(Y4),1)
    print(ret)
    print(res)
    print(neighbours)
    print(distance)
    
    classMean = calculate_meanFace(arr1)
    classMean=np.concatenate((classMean,calculate_meanFace(arr2),calculate_meanFace(arr3)),axis=1)
    
    Yfld=FisherFaces(fullfaces,classMean, totalMean, 6 ,3)
    #train_Knn(Knn,Yfld,label)
    
    YfldTest=FisherFaces(arr4,classMean, totalMean, 3 ,1)
    #ret, res, neighbours, distance=classify_Faces(Knn,np.float32(YfldTest),1)
    apply_object(arr1[0],15)
    
if __name__ == "__main__":
    main()
