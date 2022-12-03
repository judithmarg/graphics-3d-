import cv2
import numpy as np

def intersectarEsfera(origen,direccion):
    radio = 6
    centro = [4,4,4] 
    xo,yo,zo = origen[0],origen[1],origen[2]
    xd,yd,zd = direccion[0],direccion[1],direccion[2]
    ecA = xd**2 +yd**2 + zd**2
    ecB = 2*xd*(xo-centro[0])+2*yd*(yo-centro[1])+2*zd*(zo-centro[2])
    ecC = (xo-xd)**2+(yo-yd)**2+(zo-zd)**2-radio**2
    discriminante = ecB**2-4*ecA*ecC
    if discriminante < 0 :
        return (np.inf,None) 
    else:
        t1 = (-ecB+ np.sqrt(discriminante))/(2*ecA)
        t2 = (-ecB- np.sqrt(discriminante))/(2*ecA)
        hayInterseccion = True if t1 > 0 or t2 > 0 else False
        if hayInterseccion and t1 > 0:
            pointInter = np.array(origen) + t1*np.array(direccion)
        elif hayInterseccion and t2 > 0:
            pointInter = np.array(origen) + t2*np.array(direccion)
        else:
            return (np.inf,None)
        vectorNormal = (np.subtract(pointInter,centro))/radio
        vector = [pointInter, vectorNormal]
        t = t1 if t1 > 0 else t2
        return (t, vector) 

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
    direccionReflejado =  np.subtract(paralelaNormal,vectorObs)
    return direccionReflejado

def rayoTransmite(normal, vector):
    n21 = 1
    normalA = np.array(normal)
    vectorA = np.array(vector)
    vectorC = np.cross(normalA, vectorA)# producto cruz en vectores
    vectorCcuadrado = np.dot(vectorC, vectorC) #producto punto en vector c
    direccionTransmite = n21*vectorA + (n21*vectorC - np.sqrt(1+(n21**2*(vectorCcuadrado-1))))*normalA
    return direccionTransmite

def norma(vector):
    vector = vector / np.linalg.norm(vector)
    return vector

def combinarContribColor(intensidad, luz_amb,l,n,v,vn):
    kdifuso = 0.10
    kspecular = 0.10
    kambiente = 0.5
    intAmb = iluminarDirectly((91, 171, 170))
    r = np.subtract(2*np.array(n)*(np.dot(l,n)),l) 
    t = rayoTransmite(n,v)
    kd_red = kdifuso*luz_amb[0]
    ks_red = kspecular *luz_amb[0]
    kst_red = ks_red *1
    ka_red = kambiente*luz_amb[0]
    red_resultante = (intensidad[0])*(kd_red*np.dot(l,n)+ks_red*(np.dot(r,v)**vn)+kst_red*(np.dot(t,v))**vn)+ka_red*intAmb[0]
    kd_green = kdifuso*luz_amb[1]
    ks_green = kspecular *luz_amb[1]
    kst_green = ks_green *1
    ka_green = kambiente*luz_amb[1]
    green_resultante = (intensidad[1])*(kd_green*np.dot(l,n)+ks_green*(np.dot(r,v)**vn)+kst_green*(np.dot(t,v))**vn)+ka_green*intAmb[1]
    kd_blue = kdifuso*luz_amb[2]
    ks_blue= kspecular *luz_amb[2]
    kst_blue = ks_blue *1
    ka_blue = kambiente*luz_amb[2]
    blue_resultante = (intensidad[2])*(kd_blue*np.dot(l,n)+ks_blue*(np.dot(r,v)**vn)+kst_blue*(np.dot(t,v))**vn)+ka_blue*intAmb[2]
    return (int(red_resultante),int(green_resultante),int(blue_resultante))

def interseccion(vectorOrigen, vectoDireccion, objeto):
    if objeto['type'] == 'esfera':
        intersectarEsfera(vectorOrigen,vectorDireccion,objeto['pos'],objeto['normal'])

###################
camara = np.array([0.,0.35,-1.])
punteroCamara = np.array([0.,0.,0.])
luz = np.array([5.,5.,-10.])
ancho , alto = 300,200
diferencia =float(ancho)/alto 
space = (-1., -1/(diferencia)+.25 , 1., 1./diferencia+.25)
vectorDireccion = norma(punteroCamara - camara)

def prueba():
    blank = np.zeros((180,180,3), dtype="uint8")
    for i in enumerate(np.linspace(space[0],space[2],180)):
        for j in enumerate(np.linspace(space[1],space[3],180)):
            blank[i,j] = rayTracer(camara, vectorDireccion, 0, (91, 171, 170))
            punteroCamara[:2] = (i,j) 
            vectorDireccion = norma(punteroCamara-camara)
    cv2.imshow("Prueba",blank)
    cv2.waitKey(0)

def rayTracer(origen, direccion,profundidad,color):
    ##interseccion con cada figura
    vector3D = np.array(origen) + 3*np.array(direccion)  ##cambiar vector3D a rayo
    punto_interseccion = vector3D
    direccion_reflejada = vector3D
    direccion_transparente = vector3D
    color_local,color_reflejado,color_transparente = color , color, color
    if profundidad < 5:
        color = (0,0,0)
        return color
    else:
        interseccion = interseccion(origen,direccion)     #####ahora funciona con una esfera
        if interseccion[2] is None:
            color = (0,0,0) 
            return color
        else:
            color_local = iluminarDirectly(color) 
            direccion_reflejada = interseccion[1]           ### direccion donde intersecta con the shape
            direccion_reflejada = rayoReflejado(direccion_reflejada, direccion)                 #####cambiar a direccion reflejada (?)
            punto_interseccion = interseccion[0]             ### punto donde intersecta con la figura, cambiar (?)
            rayTracer(punto_interseccion, direccion_reflejada,profundidad+1, color_reflejado) 
            direccion_transparente = rayoTransmite(direccion_transparente, direccion_reflejada)         #### direccion a cambiar?
            rayTracer(punto_interseccion, direccion_transparente,profundidad+1,color_transparente)
            colorFin = combinarContribColor(color_local,color_local,direccion_reflejada,direccion_reflejada,direccion_transparente,1)
    return colorFin


def main():
    prueba()
if __name__ == '__main__':
    main()

