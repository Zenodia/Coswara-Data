import tensorrt as trt
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
def build_engine(onnx_path, shape = [1,1,32000]):

    """
    This is the function to create the TensorRT engine
    Args:
      onnx_path : Path to onnx_file. 
      shape : Shape of the input of the ONNX file. 
    """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = (256 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_cuda_engine(network)
        return engine

def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)
def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def inspect_engine(engine):
    profile_meta = {}
    num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
    for profile_index in range(engine.num_optimization_profiles):
        start_binding = profile_index * num_bindings_per_profile
        end_binding = start_binding + num_bindings_per_profile
        
        binding_meta = {}
        for binding_index in range(start_binding, end_binding):
            key = "Binding {}".format(binding_index)
            binding_meta[key] = {
                "profile": profile_index,
                "binding_index": binding_index,
                "binding_shape": engine.get_binding_shape(binding_index),
                "binding_dtype": engine.get_binding_dtype(binding_index),
                "binding_name": engine.get_binding_name(binding_index),
            }

            if engine.binding_is_input(binding_index):
                binding_meta[key]["binding_type"] = "INPUT"
                binding_meta[key]["profile_shape"] = engine.get_profile_shape(profile_index, binding_index)
            else:
                binding_meta[key]["binding_type"] = "OUTPUT"

        profile_meta["Profile {}".format(profile_index)] = binding_meta

    from pprint import pprint
    pprint(profile_meta)
