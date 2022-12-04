from PIL import Image
import numpy as np

w = 400
h = 300

def normalize(x):
    x /= np.linalg.norm(x)
    return x

def intersect_plane(O, D, P, rayoDireccion):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # plane (P, rayoDireccion), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, rayoDireccion)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, rayoDireccion) / denom
    if d < 0:
        return np.inf
    return d

def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def intersect(O, D, obj):
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])

def get_normal(obj, rayo):
    # Find normal.
    if obj['type'] == 'sphere':
        rayoDireccion= normalize(rayo - obj['position'])
    elif obj['type'] == 'plane':
        rayoDireccion= obj['normal']
    return rayoDireccion
    
def get_color(obj, rayo):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(rayo)
    return color

def trace_ray(rayO, rayD):
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    if t == np.inf:
        return

    obj = scene[obj_idx]
    rayo = rayO + rayD * t
    rayoDireccion= get_normal(obj, rayo)
    color = get_color(obj, rayo)
    toL = normalize(Light - rayo)
    toO = normalize(origen - rayo)

    col_ray = ambient
    col_ray += obj.get('diffuse_c', diffuse_c) * max(np.dot(rayoDireccion, toL), 0) * color
    col_ray += obj.get('specular_c', specular_c) * max(np.dot(rayoDireccion, normalize(toL + toO)), 0) ** specular_k * color_light
    return obj, rayo,  rayoDireccion, col_ray

def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position), 
        radius=np.array(radius), color=np.array(color), reflection=.5)
    
def add_plane(position, normal):
    return dict(type='plane', position=np.array(position), 
        normal=np.array(normal),
        color=lambda M: (color_plane1 
            if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane0),
        
        diffuse_c=.75, specular_c=.5, reflection=.25)
    
# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)
scene = [add_plane([0.,0.,4.],[0.,0.,1.]),
         add_sphere([.75, .1, 1.], .6, [0., 0., 1.]),
         add_sphere([-.75, .1, 2.25], .6, [.5, .223, .5]),
         add_sphere([-2.75, .0, 3.5], .5, [1., .572, .184]),
         add_plane([0., -.5, 0.], [0., 1., 0.]),
         
    ]

Light = np.array([5., 5., -10.])
color_light = np.ones(3)

ambient = .05
diffuse_c = 1.
specular_c = 1.
specular_k = 50

depth_max = 5  
col = np.zeros(3)  
origen = np.array([0., 0.35, -1.])   #En un principio es la camara
punteroOrigen = np.array([0., 0., 0.])  
img = np.zeros((h, w, 3))

r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)

# Loop through all pixels.
def recursive_tray(rayO, rayD, depth, reflection):
    global col,depth_max
 
    if depth==depth_max:
        return
 
    traced = trace_ray(rayO, rayD)
    if not traced:
        return
    obj, rayo,  rayoDireccion, col_ray = traced
    # Reflection: create a new ray.
    rayO, rayD = rayo + rayoDireccion* .0001, normalize(rayD - 2 * np.dot(rayD, rayoDireccion) * rayoDireccion)
    col += reflection * col_ray
    reflection *= obj.get('reflection', 1.)
    recursive_tray(rayO,rayD,depth+1,reflection)
 
 
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print(i / float(w) * 100, "%")
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col[:] = 0
        punteroOrigen[:2] = (x, y)#modifico x y de deonde apunta camara
        D = normalize(punteroOrigen - origen)
        depth = 0
        rayO, rayD = origen, D
        reflection = 1.
        recursive_tray(rayO,rayD,0,1)
        # Loop through initial and secondary rays.
        # while depth < depth_max:
        #     traced = trace_ray(rayO, rayD)
        #     if not traced:
        #         break
        #     obj, M, N, col_ray = traced
        #     # Reflection: create a new ray.
        #     rayO, rayD = M + N * .0001, normalize(rayD - 2 * np.dot(rayD, N) * N)
        #     depth += 1
        #     col += reflection * col_ray
        #     reflection *= obj.get('reflection', 1.)
        img[h - j - 1, i, :] = np.clip(col, 0, 1)

im = Image.fromarray((255 * img).astype(np.uint8), "RGB")
im.save("fig3.png")