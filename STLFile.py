from PIL import Image 
import numpy as np
from stl import mesh

def generateSTL(imagePath, imageDest):
    img = Image.open(imagePath)
    img = np.array(img)

    Normal_img = img / 255.0

    height, width = Normal_img.shape
    print(height, width)

    vertices = np.zeros((height, width, 3))

    for i in range(height):
        for j in range(width):
            vertices[i][j] = [j, i, (Normal_img[i, j]) * 100]
    
    bottom_vertices = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            bottom_vertices[i][j] = [j, i, -20]
    
    t_flat = vertices.reshape(-1, 3)
    b_flat = bottom_vertices.reshape(-1, 3)

    vertices = np.vstack((t_flat, b_flat))
    faces = []

    for y in range(height-1):
        for x in range(1, width - 1):
            center = y * width + x
            below  = center + width
            right  = center + 1
            diagonal = center + width + 1

            off_center = t_flat.shape[0] + (y * width + x)
            off_right = off_center + 1
            off_below = off_center + width
            off_diagonal = off_center + width + 1

            faces.append([center, below, right])
            faces.append([right, diagonal, below])
            faces.append([off_center, off_below, off_right])
            faces.append([off_right, off_below, off_diagonal])

    for j in range(width - 1):
        top1 = j
        top2 = j + 1
        bot1 = top1 + t_flat.shape[0]
        bot2 = top2 + t_flat.shape[0]
        faces.append([top1, bot1, top2])
        faces.append([top2, bot1, bot2])
        
    for j in range(width - 1):
        top1 = (height - 1) * width + j
        top2 = (height - 1) * width + (j + 1)
        bot1 = top1 + t_flat.shape[0]
        bot2 = top2 + t_flat.shape[0]
        faces.append([top2, bot1, top1])
        faces.append([top2, bot2, bot1])
        
    for i in range(height - 1):
        top1 = i * width
        top2 = (i + 1) * width
        bot1 = top1 + t_flat.shape[0]
        bot2 = top2 + t_flat.shape[0]
        faces.append([top1, top2, bot1])
        faces.append([top2, bot2, bot1])
        
    for i in range(height - 1):
        top1 = i * width + (width - 1)
        top2 = (i + 1) * width + (width - 1)
        bot1 = top1 + t_flat.shape[0]
        bot2 = top2 + t_flat.shape[0]
        faces.append([top1, bot1, top2])
        faces.append([top2, bot1, bot2])

    faces_all = np.array(faces)

    stl_mesh = mesh.Mesh(np.zeros(faces_all.shape[0], dtype=mesh.Mesh.dtype))
    print("About to Enumerate them")
    for i, face in enumerate(faces_all):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertices[face[j]]
    stl_mesh.save(imageDest)


generateSTL("Personal/Sasha/GSasha.png", "Personal/Sasha/STLSasha.stl")