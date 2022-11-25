import cv2
import numpy as np

def prueba():
    blank = np.zeros((180,190,3), dtype="uint8")
    cv2.imshow("Prueba",blank)
    cv2.waitKey(0)

def rayTracer(origen, direccion,profundidad,color):
    vector3D = np.array(origen) + 3*np.array(direccion)
    punto_interseccion = vector3D
    direccion_reflejada = vector3D
    direccion_transparente = vector3D
    color_local1,color_reflejado,color_transperente = color 
    if profundidad > 5:
        color = black
    else:
        interseccion = intersectarRayo(origen,direccion) #####
        if interseccion is None:
            color = black 
        else:
            color_local1 = iluminarDirectly() 
            rayo_reflejado = rayoReflejado() #####
            punto_interseccion = interseccion[0] ###
            #direccion_reflejada = interseccion[1] ###
            rayTracer(punto_interseccion, direccion_reflejada,profundidad+1, color_reflejado) 
            direccion = rayoTransmite() #####
            rayTracer(punto_interseccion, direccion_transparente,profundidad+1,color_transperente)
            color = combinarContribColor()
    return color

def intersectarRayo(origen,direccion):
    radio = 2
    centro = [4,4,4] 
    xo,yo,zo = origen[0],origen[1],origen[2]
    xd,yd,zd = direccion[0],direccion[1],direccion[2]
    ecA = xd**2 +yd**2 + zd**2
    ecB = 2*xd*(xo-centro[0])+2*yd*(yo-centro[1])+2*zd*(zo-centro[2])
    ecC = (xo-xd)**2+(yo-yd)**2+(zo-zd)**2-radio**2
    discriminante = ecB**2-4*ecA*ecC
    if discriminante < 0 :
        return None 
    else:
        t1 = (-ecB+ np.sqrt(discriminante))/(2*ecA)
        t2 = (-ecB- np.sqrt(discriminante))/(2*ecA)
        hayInterseccion = True if t1 > 0 or t2 > 0 else False
        if hayInterseccion and t1 > 0:
            pointInter = np.array(origen) + t1*np.array(direccion)
        elif hayInterseccion and t2 > 0:
            pointInter = np.array(origen) + t2*np.array(direccion)
        else:
            return None
        vectorNormal = (np.substract(pointInter,centro))/radio
        vector = []
        vector[0] = pointInter
        vector[1] = vectorNormal
        return vector 

