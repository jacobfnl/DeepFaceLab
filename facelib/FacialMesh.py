# https://github.com/patrikhuber/eos/blob/master/python/demo.py
# https://github.com/patrikhuber/eos/blob/master/examples/fit-model.cpp
# http://patrikhuber.github.io/eos/doc/index.html
import os
import eos
import numpy as np

"""
This app demonstrates estimation of the camera and fitting of the shape
model of a 3D Morphable Model from an ibug LFPW image with its landmarks.
In addition to fit-model-simple, this example uses blendshapes, contour-
fitting, and can iterate the fitting.

68 ibug landmarks are loaded from the .pts file and converted
to vertex indices using the LandmarkMapper.
"""

EOS_DIR = os.path.join(os.path.dirname(__file__), 'eos')
EOS_MODEL = os.path.join(EOS_DIR, 'sfm_shape_3448.bin')
EOS_BLENDSHAPES = os.path.join(EOS_DIR, 'expression_blendshapes_3448.bin')
EOS_MAPPER = os.path.join(EOS_DIR, 'ibug_to_sfm.txt')
EOS_EDGE_TOPO = os.path.join(EOS_DIR, 'sfm_3448_edge_topology.json')
EOS_CONTOURS = os.path.join(EOS_DIR, 'sfm_model_contours.json')


def get_mesh_landmarks(landmarks, image):
    image_height, image_width, _ = image.shape
    eos_landmarks = _format_landmarks_for_eos(landmarks)
    mesh, pose = _predict_3d_mesh(image_width, image_height, eos_landmarks)

    v = np.asarray(mesh.vertices)
    points_2d = _project_points(v, pose, image_width, image_height)
    points = points_2d[:, :2] + [image_width/2, image_height/2]

    isomap = get_texture(mesh, pose, image)

    # iso_2d = _project_isomap(np.asarray(isomap), pose, image_width, image_height)
    # iso_points = iso_2d[:, :, :2] # + [image_width/2, image_height/2]

    render(mesh, pose, isomap, image_width, image_height)
    return points, isomap


def _format_landmarks_for_eos(landmarks):
    eos_landmarks = []
    ibug_index = 1  # count from 1 to 68 for all ibug landmarks
    for coords in landmarks:
        eos_landmarks.append(eos.core.Landmark(str(ibug_index), [coords[0], coords[1]]))
        ibug_index = ibug_index + 1
    return eos_landmarks


def _predict_3d_mesh(image_width, image_height, landmarks):
    model = eos.morphablemodel.load_model(EOS_MODEL)

    # The expression blendshapes:
    blendshapes = eos.morphablemodel.load_blendshapes(EOS_BLENDSHAPES)

    # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
    # morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
    #                                                                     color_model=eos.morphablemodel.PcaModel(),
    #                                                                     vertex_definitions=None,
    #                                                                     texture_coordinates=model.get_texture_coordinates())

    # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
    morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                        color_model=eos.morphablemodel.PcaModel(),
                                                                        vertex_definitions=None,
                                                                        texture_coordinates=model.get_texture_coordinates())

    # The landmark mapper is used to map 2D landmark points (e.g. from the ibug scheme) to vertex ids:
    landmark_mapper = eos.core.LandmarkMapper(EOS_MAPPER)

    # The edge topology is used to speed up computation of the occluding face contour fitting:
    edge_topology = eos.morphablemodel.load_edge_topology(EOS_EDGE_TOPO)

    # These two are used to fit the front-facing contour to the ibug contour landmarks:
    contour_landmarks = eos.fitting.ContourLandmarks.load(EOS_MAPPER)
    model_contour = eos.fitting.ModelContour.load(EOS_CONTOURS)

    # Fit the model, get back a mesh and the pose:
    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(morphablemodel_with_expressions,
                                                                                   landmarks, landmark_mapper,
                                                                                   image_width, image_height,
                                                                                   edge_topology, contour_landmarks,
                                                                                   model_contour)
    # can be saved as *.obj, *.isomap.png
    return mesh, pose


def get_pitch_yaw_roll(pose):
    # // The 3D head pose can be recovered as follows:
    # float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
    # // and similarly for pitch and roll.
    pitch, yaw, roll = pose.get_rotation_euler_angles()
    print('pitch, yaw, roll:', pitch, yaw, roll)


# Extract the texture from the image using given mesh and camera parameters:
def get_texture(mesh, pose, image):
    isomap = eos.render.extract_texture(mesh, pose, image, isomap_resolution=1024)
    # print('isomap:', isomap)
    print('isomap shape', np.shape(isomap))
    return isomap


# based on https://github.com/patrikhuber/eos/issues/140#issuecomment-314775288
def _get_opencv_viewport(width, height):
    return np.array([0, height, width, -height])


def _get_viewport_matrix(width, height):
    viewport = _get_opencv_viewport(width, height)
    viewport_matrix = np.zeros((4, 4))
    viewport_matrix[0, 0] = 0.5 * viewport[2]
    viewport_matrix[3, 0] = 0.5 * viewport[2] + viewport[0]
    viewport_matrix[1, 1] = 0.5 * viewport[3]
    viewport_matrix[3, 1] = 0.5 * viewport[3] + viewport[1]
    viewport_matrix[2, 2] = 0.5
    viewport_matrix[3, 2] = 0.5
    return viewport_matrix


def _project_points(v, pose, width, height):
    # project through pose
    points = np.copy(v)
    vpm = _get_viewport_matrix(width, height)
    projection = pose.get_projection()
    modelview = pose.get_modelview()

    points = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    return np.asarray([vpm.dot(projection).dot(modelview).dot(point) for point in points])


def _project_isomap(isomap, pose, width, height):
    # project through pose
    isomap = np.copy(isomap)
    vpm = _get_viewport_matrix(width, height)
    projection = pose.get_projection()
    modelview = pose.get_modelview()

    # result = np.array([])
    for i, row in enumerate(isomap):
        # result_row = np.array([])
        for j, col in enumerate(row):
            isomap[i, j, :] = vpm.dot(projection).dot(modelview).dot(isomap[i, j, :])

    return isomap

def render(mesh, pose, texture, width, height):
    projection = pose.get_projection()
    modelview = pose.get_modelview()
    vpm = _get_viewport_matrix(width, height)
    eos.render.render(mesh, modelview, projection, vpm, texture)
