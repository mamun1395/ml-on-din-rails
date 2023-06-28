import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt

TRT_SAVED_MODEL_DIR = '../trt_saved_models/'

if not os.path.exists(TRT_SAVED_MODEL_DIR):
    os.mkdir(TRT_SAVED_MODEL_DIR)

input_saved_model_dir = '../tensorflow_saved_models/c_lstm_saved_model'
output_saved_model_dir = TRT_SAVED_MODEL_DIR + 'c_lstm_saved_model_TFTRT_FP32'

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32, max_workspace_size_bytes=1000000000)
converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir, conversion_params=conversion_params)

converter.convert()
converter.save(output_saved_model_dir=output_saved_model_dir)
