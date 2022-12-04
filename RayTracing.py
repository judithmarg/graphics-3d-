from PIL import Image
import numpy as np

def intersectarEsfera(origen,direccion,centro,radio):
    xo,yo,zo = origen[0],origen[1],origen[2]
    xd,yd,zd = direccion[0],direccion[1],direccion[2]
    xc,yc,zc = centro[0], centro[1],centro[2]
    ecA = xd**2 +yd**2 + zd**2
    ecB = 2*xd*(xo-xc)+2*yd*(yo-yc)+2*zd*(zo-zc)
    ecC = (xo-xc)**2+(yo-yc)**2+(zo-zc)**2-radio**2
    discriminante = ecB**2-4*ecA*ecC
    if discriminante < 0 :
        return (np.inf,None) 
    else:
        t1 = (-ecB+ np.sqrt(discriminante))/(2.00)
        t2 = (-ecB- np.sqrt(discriminante))/(2.00)
        aux = t1 if ecB < 0 else t2
        t1 = aux / ecA
        t2 = ecC / aux  
        hayInterseccion =  t1 > 0 or t2 > 0 
        if hayInterseccion and t1 > 0:
            pointInter = np.array(origen) + t1*np.array(direccion)
        elif hayInterseccion and t2 > 0:
            pointInter = np.array(origen) + t2*np.array(direccion)
        else:
            return (np.inf,None)
        vectorNormal = (np.subtract(pointInter,centro))/radio
        vector = [pointInter, vectorNormal]
        t1, t2 = min(t1, t2), max(t1, t2)
        if t2 >= 0:
            t = t2 if t1 < 0 else t1
        return (t,vector) 

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
    vectorC = np.cross(normalA, vectorA)
    vectorCcuadrado = np.dot(vectorC, vectorC) 
    direccionTransmite = n21*vectorA + (n21*vectorC 
    - np.sqrt(1+(n21**2*(vectorCcuadrado-1))))*normalA
    return direccionTransmite

def norma(vector):
    vector = vector / np.linalg.norm(vector)
    return vector

def combinarContribColor(intensidad, luz_amb,l,n,v,vn):
    
    intAmb = intensidad
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
    return (max(red_resultante,0),max(green_resultante,0),max(blue_resultante,0))

def normalizado(obj, rayo):
    if obj['type'] == 'esfera':
        return norma(rayo - obj['posicion'])

def get_color(obj, rayo):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(rayo)
    return color 

def interseccion1(vectorOrigen, vectorDireccion, objeto):
    if objeto['type'] == 'esfera':
        return intersectarEsfera(vectorOrigen,vectorDireccion,objeto['posicion'],objeto['radio'])

###################

def add_esfera(posicion, radio, color):
    return dict(type='esfera',posicion=np.array(posicion), radio=np.array(radio),color=np.array(color))

escena = [add_esfera([.75,.1,1.], .6, [0.,0.,1.]),
          add_esfera([-.75, .1, 2.25], .6, [.5, .223, .5]),
          add_esfera([-2.75, .1, 3.5], .6, [1., .572, .184])]

camara = np.array([0.,0.35,-1.])
punteroCamara = np.array([0.,0.,0.])
luz = np.array([5.,5.,-10.])
ancho , alto = 400,300
diferencia =float(ancho)/alto 
space = (-1., -1./(diferencia)+.25 , 1., 1./diferencia+.25)
vectorDireccion = norma(punteroCamara - camara)
kdifuso = 1.
kspecular = 1.
kambiente = 0.05

def prueba():
    img = np.zeros((alto,ancho,3))
    vectorDireccion = norma(punteroCamara - camara)
    for i,x in enumerate(np.linspace(space[0],space[2],ancho)): 
        if i % 10 == 0:
            print(i / float(ancho) * 100, "%")
        for j,y in enumerate(np.linspace(space[1],space[3],alto)):
            punteroCamara[:2] = (x,y) 
            vectorDireccion = norma(punteroCamara-camara)
            colorFinal = rayTracer(camara, vectorDireccion, 0, (0,0,0), escena)
            img[alto - j - 1, i, :] = np.clip(colorFinal, 0, 1)

    im = Image.fromarray((255 * img).astype(np.uint8), "RGB")
    im.save("figurita.png")

def rayTracer(origen, direccion,profundidad,color,escena):

    if profundidad >= 2:
        color = (0,0,0)
        return color
    else:
        ###
        t = np.inf
        for i, figura in enumerate(escena):
            t_figura = interseccion1(origen,direccion,figura)[0]
            if t_figura < t:
                t, figura_ind = t_figura, i 
        if t == np.inf:
            return (0,0,0)
        figura = escena[figura_ind]
        rayo = origen + t* direccion
        rayoNormalizado = normalizado(figura, rayo)
        paraLuz = norma(luz - rayo)
        paraOrigen = norma(camara - rayo)
        punto_interseccion = rayoNormalizado
        direccion_reflejada = rayoNormalizado
        direccion_transparente = rayoNormalizado
        figuraColor = get_color(figura, rayo)
        color_local,color_reflejado,color_transparente = figuraColor , figuraColor, figuraColor
        
        # ##
        interseccionRes = interseccion1(origen,direccion,figura)     #####ahora funciona con una esfera
        if interseccionRes[1] is None:
            color = (0,0,0) 
            return color
        else:
            color_local = iluminarDirectly(figuraColor) 
            direccion_reflejada = interseccionRes[1][1]           ### direccion donde intersecta con the shape
            direccion_reflejada = rayoReflejado(direccion_reflejada, rayoNormalizado)                 #####cambiar a direccion reflejada (?)
            punto_interseccion = interseccionRes[1][0]             ### punto donde intersecta con la figura, cambiar (?)
            rayTracer(punto_interseccion, direccion_reflejada,profundidad+1, color_reflejado,escena) 
            direccion_transparente = rayoTransmite(direccion_transparente, direccion_reflejada)         #### direccion a cambiar?
            rayTracer(punto_interseccion, direccion_transparente,profundidad+1,color_transparente,escena)
            color = combinarContribColor(figuraColor,color_local,[1,1,1],direccion_reflejada,direccion_transparente,1)
            
    return color

def main():
    prueba()

if __name__ == '__main__':
    main()

