import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def pose_vis(keypoints, flip_pairs, parent_ids):
    fig = plt.figure(figsize=(4, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')

    keypoints = np.array(keypoints)
    N = keypoints.shape[0]

    colors = ['black'] * N
    for i, j in flip_pairs:
        colors[i] = 'blue'
        colors[j] = 'red'

    for i in range(N):
        ax.scatter(*keypoints[i], color=colors[i], s=20)

    ax.scatter(0, 0, 0, color='orange', s=100, marker='*')

    for child_id, parent_id in enumerate(parent_ids):
        x = [keypoints[child_id][0], keypoints[parent_id][0]]
        y = [keypoints[child_id][1], keypoints[parent_id][1]]
        z = [keypoints[child_id][2], keypoints[parent_id][2]]
        ax.plot(x, y, z, color='black', linewidth=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    ax.set_box_aspect([1, 1, 1])

    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return img


def render_mesh_A800(verts, faces):
    fig = plt.figure(figsize=(4, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor('lightgrey')
    ax.add_collection3d(mesh)

    ax.scatter(0, 0, 0, color='orange', s=100, marker='*')

    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])

    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return img