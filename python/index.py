import cv2
import numpy as np
from time import sleep



dikdörtgen_min_height=70     # DİKDÖRTGENİN YÜKSEKLİĞİ
dikdörtgen_min_width=70     # DİKDÖRTGENİN GENİŞLİĞİ
offset=5 # piksel hatası toleransı arttırırsak sayaç manasız hızlı artıyor azaltırsak geçişleri okumuyor
line_height=775 # çizginin yüksekliği
delay= 60 # karelerin geçme hızı
array = []
value= 0

def dikdortgen_origin(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy
video = cv2.VideoCapture('video.mp4')

dinamik_object = cv2.bgsegm.createBackgroundSubtractorMOG()   # hareketli nesneleri bulma Bu fonksiyon, bir görüntüdeki arka planı modelleyerek ve ardından bir sonraki kare ile karşılaştırarak hareketli nesnelerin tespitini gerçekleştirir. chatgpt uzun yazısı bulunmakta.

while True:
    ret,frame1 = video.read()
    tempo = float(1/delay)
    sleep(tempo) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # gri tonlamaya çevirme
    blur = cv2.GaussianBlur(grey,(3,3),5) # bulanıklaştırma
    dinamik = dinamik_object.apply(blur) # hareketli nesneleri bulma
    dilate = cv2.dilate(dinamik,np.ones((5,5)))  # beyazları büyütme
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #5x5 boyutunda bir filtre oluşturulmuştur.
    morph = cv2.morphologyEx (dilate, cv2. MORPH_CLOSE , kernel) # gereksiz beyaz silme
    morph = cv2.morphologyEx (morph, cv2. MORPH_CLOSE , kernel) # gereksiz beyaz silme
    contour,h=cv2.findContours(morph,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #kontür bulma
    
    cv2.line(frame1, (210, line_height), (1500, line_height), (0,0,255), 3) 
    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        check_contour = (w >= dikdörtgen_min_height) and (h >= dikdörtgen_min_width)
        if not check_contour:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)   # dikdörtgen çizdirme     
        origin = dikdortgen_origin(x, y, w, h)
        array.append(origin) # dikdörtgenin orta noktasını listeye ekleme
        cv2.circle(frame1, origin, 4, (0, 0,255), -1) # dikdörtgenin orta noktasına çember çizdirme

        for (x,y) in array:
            if y<(line_height+offset) and y>(line_height-offset):
                value+=1
                cv2.line(frame1, (210,line_height), (1500,line_height), (255,0,255), 3)  
                array.remove((x,y))
                      
        

        
    cv2.putText(frame1, "Arac sayisi: "+str(value), (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),4)
    cv2.imshow("Video Original" , frame1)
    
    

    if cv2.waitKey(1) == 27: # esc tuşu ascii değeri programı kapatır. 
        break
    
cv2.destroyAllWindows()
video.release()
