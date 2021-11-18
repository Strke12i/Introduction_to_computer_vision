import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self,staticMode=False,maxFaces=2,refine_landmarks=False,minDetectionCon=0.5,minTrackCon=0.5):
        #definição dos parametros
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.refine_landmarks=refine_landmarks
        self.minDetectionCon=minDetectionCon
        self.minTrackCon=minTrackCon
        
        self.mpDraw=mp.solutions.drawing_utils
        self.mpFaceMesh=mp.solutions.face_mesh
        #criação do faceMash com os parâmetros do init 
        self.faceMesh=self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.refine_landmarks,self.minDetectionCon,self.minTrackCon)
        self.drawSpec=self.mpDraw.DrawingSpec(thickness=1,circle_radius=1)

    #função que acha e desenha(se true) os pontos na face
    def findFaceMesh(self,img,draw=True):
        #transforma a imagem de BGR para RGB
        self.imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #armazena os resultados da Mesh na var results
        self.results=self.faceMesh.process(self.imgRGB)
        faces=[]
        if(self.results.multi_face_landmarks):
            
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_CONTOURS,self.drawSpec,self.drawSpec)
                
                face=[]
                for id,lm in enumerate(faceLms.landmark):
                    ih,iw,ic=img.shape
                    x,y=int(lm.x*iw),int(lm.y*ih)
                    cv2.putText(img,str(id),(x,y),cv2.FONT_ITALIC,0.2,(0,255,0),1)
                    face.append([x,y])
                    
                faces.append(face)
        return img, faces
    
def main():
    cap=cv2.VideoCapture(0)
    pTime=0

    detector=FaceMeshDetector()

    while True:
        sucess,img=cap.read()
        img,faces=detector.findFaceMesh(img,False)
        
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img,f'fps:{int(fps)}',(20,70),cv2.FONT_ITALIC,3,(255,255,0),3)


        cv2.imshow('img',img)
        cv2.waitKey(1)
    
if __name__ == "__main__":
    main()