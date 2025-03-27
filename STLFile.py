from PIL import Image, ImageFilter
import numpy as np
from stl import mesh

def generateSTL(imagePath):
    img = Image.open(imagePath)
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    img = np.array(img)

    height, width = img.shape

    vertices = []
    faces = []

    for i in range(height):
        for j in range(width):
            if img[i][j] == 255:
                vertices.append([j, i, 1])
            else:
                vertices.append([j, i, img[i][j]])
    
    vertices = np.array(vertices)

    for y in range(1, height-1):
        for x in range(1, width - 1):
            center = y * width + x
            above = (y-1) * width + x
            below = (y+1) * width + x
            left = y * width + (x+1)
            right = y * width + (x-1)

            faces.append([center, above, right])
            faces.append([center, above, left])
            faces.append([center, below, right])
            faces.append([center, below, left])

    faces = np.array(faces)

    stl = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl.vectors[i][j] = vertices[f[j]]
    
    stl.save("GaussianBenchyMesh.stl")


generateSTL('Gray3DBenchy.png')