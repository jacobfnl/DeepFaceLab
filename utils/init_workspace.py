import os


def init_workspace(working_path=None):
    # make the workspace
    current_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    if working_path is not None:
        if not os.path.exists(working_path):
            os.makedirs(working_path)
            if not os.path.exists(working_path):
                print("unable to use/create workspace at {}".format(working_path))
                exit(-1)
        current_directory = working_path
    workspace = os.path.join(current_directory, 'workspace')
    # mkdir  workspace/data_src/aligned
    src_aligned = os.path.join(workspace, 'data_src', 'aligned')
    # mkdir workspace/data_dst/aligned
    dst_aligned = os.path.join(workspace, 'data_dst', 'aligned')
    # mkedir workspace/model
    workspace_model = os.path.join(workspace, 'model')

    if not os.path.exists(src_aligned):
        os.makedirs(src_aligned)
        if not os.path.exists(src_aligned):
            print("unable to use/create workspace at {}".format(workspace))
            exit(-1)
        print("made dir: {}".format(src_aligned))

    if not os.path.exists(dst_aligned):
        os.makedirs(dst_aligned)
        print("made dir: {}".format(dst_aligned))

    if not os.path.exists(workspace_model):
        os.makedirs(workspace_model)
        print("made dir: {}".format(workspace_model))

    # print("workspace ready: {}".format(workspace))
    return workspace, src_aligned, dst_aligned, workspace_model

