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
    color_local,color_reflejado,color_transparente = color 
    if profundidad > 5:
        color = (0,0,0)
    else:
        interseccion = intersectarRayo(origen,direccion)     #####ahora funciona con una esfera
        if interseccion is None:
            color = (0,0,0) 
        else:
            color_local = iluminarDirectly(colorFigura) 
            direccion_reflejada = interseccion[1]           ### direccion donde intersecta con the shape
            direccion_reflejada = rayoReflejado(direccion_reflejada, direccion)                 #####cambiar a direccion reflejada (?)
            punto_interseccion = interseccion[0]             ### punto donde intersecta con la figura, cambiar (?)
            rayTracer(punto_interseccion, direccion_reflejada,profundidad+1, color_reflejado) 
            direccion_transparente = rayoTransmite(direccion_transparente, direccion_reflejada)         #### direccion a cambiar?
            rayTracer(punto_interseccion, direccion_transparente,profundidad+1,color_transparente)
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

def iluminarDirectly(colorObjeto):
    color_amb = (0.5*colorObjeto[0], 0.5*colorObjeto[1], 0.5*colorObjeto[2])
    return color_amb

def rayoReflejado(normal,vectorObs):
    productoPunto = 0
    for i in range (0,len(normal)):
        productoPunto += normal[i]*vectorObs[i]
    normalArray = np.array(normal)
    paralelaNormal = normalArray * (2*productoPunto)
    paralelaNormal = np.array(paralelaNormal)
    direccionReflejado =  np.substract(paralelaNormal,vectorObs)
    return direccionReflejado

def rayoTransmite(normal, vector):
    n21 = 1
    normalA = np.array(normal)
    vectorA = np.array(vector)
    vectorC = np.cross(normalA, vectorA)# producto cruz en vectores
    vectorCcuadrado = np.dot(vectorC, vectorC) #producto punto en vector c
    direccionTransmite = n21*vectorA + (n21*vectorC - np.sqrt(1+(n21**2*(vectorCcuadrado-1))))*normalA
    return direccionTransmite
