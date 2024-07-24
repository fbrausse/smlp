import tensorflow as tf
import numpy as np
import h5py


def read_h5_weights(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        weights = {}
        for layer_name in f.keys():
            layer = f[layer_name]
            for weight_name in layer.keys():
                weights[layer_name + '/' + weight_name] = np.array(layer[weight_name])
    return weights


def verify_pb_file(pb_file_path):
    # Verify the protobuf file is valid
    try:
        with tf.io.gfile.GFile(pb_file_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        return True
    except tf.errors.InvalidArgumentError as e:
        print(f"Error verifying the PB file: {e}")
        return False


def read_pb_weights(pb_file_path):
    # Verify the file first
    if not verify_pb_file(pb_file_path):
        raise ValueError(f"The file at {pb_file_path} is not a valid TensorFlow protobuf file.")

    # Load the protobuf graph
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(pb_file_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    # Import the graph and get the weights
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    weights = {}
    with tf.compat.v1.Session(graph=graph) as sess:
        for op in graph.get_operations():
            if op.type == "Const":
                weights[op.name] = sess.run(op.outputs[0])
    return weights


def compare_weights(h5_weights, pb_weights):
    for key in h5_weights:
        if key in pb_weights:
            if np.allclose(h5_weights[key], pb_weights[key]):
                print(f"Weights for {key} match.")
            else:
                print(f"Weights for {key} do not match.")
        else:
            print(f"Weight {key} not found in PB model.")

    for key in pb_weights:
        if key not in h5_weights:
            print(f"Weight {key} not found in H5 model.")


def check_weights(h5_file_path, pb_file_path):
    h5_weights = read_h5_weights(h5_file_path)
    pb_weights = read_pb_weights(pb_file_path)
    compare_weights(h5_weights, pb_weights)


# Example usage:
# check_weights('path_to_model.h5', 'path_to_model.pb')

# Example usage:
check_weights("/home/ntinouldinho/Desktop/smlp/result/abc_smlp_toy_basic_nn_keras_model_complete.h5", "/home/ntinouldinho/Desktop/smlp/src/smlp_py/NN_verifiers/saved_model.pb")
