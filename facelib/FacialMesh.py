# https://github.com/patrikhuber/eos/blob/master/python/demo.py
import os
import eos
import numpy as np

EOS_DIR = os.path.join(os.path.dirname(__file__), 'eos')
EOS_MODEL = os.path.join(EOS_DIR, 'sfm_shape_3448.bin')
EOS_BLENDSHAPES = os.path.join(EOS_DIR, 'expression_blendshapes_3448.bin')
EOS_MAPPER = os.path.join(EOS_DIR, 'ibug_to_sfm.txt')
EOS_EDGE_TOPO = os.path.join(EOS_DIR, 'sfm_3448_edge_topology.json')
EOS_CONTOURS = os.path.join(EOS_DIR, 'sfm_model_contours.json')


def get_mesh_landmarks(landmarks, image, bbox):
    l, t, r, b = bbox
    # image_height, image_width = b - t, r - l
    image_height, image_width, _ = image.shape
    eos_landmarks = _format_landmarks_for_eos(landmarks)
    mesh, pose = _predict_3d_mesh(image_width, image_height, eos_landmarks)
    v = np.asarray(mesh.vertices)
    points_2d = _project_points(v, pose, image_width, image_height)
    points = points_2d[:, :2] + [t + image_height/2, l + image_width/2]
    return points


def _format_landmarks_for_eos(landmarks):
    eos_landmarks = []
    ibug_index = 1  # count from 1 to 68 for all ibug landmarks
    for coords in landmarks:
        eos_landmarks.append(eos.core.Landmark(str(ibug_index), [coords[0], coords[1]]))
        ibug_index = ibug_index + 1
    return eos_landmarks


def _predict_3d_mesh(image_width, image_height, landmarks):
    model = eos.morphablemodel.load_model(EOS_MODEL)
    blendshapes = eos.morphablemodel.load_blendshapes(EOS_BLENDSHAPES)
    # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
    morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                        color_model=eos.morphablemodel.PcaModel(),
                                                                        vertex_definitions=None,
                                                                        texture_coordinates=model.get_texture_coordinates())
    landmark_mapper = eos.core.LandmarkMapper(EOS_MAPPER)
    edge_topology = eos.morphablemodel.load_edge_topology(EOS_EDGE_TOPO)
    contour_landmarks = eos.fitting.ContourLandmarks.load(EOS_MAPPER)
    model_contour = eos.fitting.ModelContour.load(EOS_CONTOURS)

    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(morphablemodel_with_expressions,
                                                                                   landmarks, landmark_mapper,
                                                                                   image_width, image_height,
                                                                                   edge_topology, contour_landmarks,
                                                                                   model_contour)
    return mesh, pose


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
