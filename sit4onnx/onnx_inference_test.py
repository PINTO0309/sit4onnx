#! /usr/bin/env python

import os
import sys
import time
import onnx
import onnxruntime
import numpy as np
from typing import Optional, List
from argparse import ArgumentParser

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'


ONNX_DTYPES_TO_NUMPY_DTYPES: dict = {
    f'{onnx.TensorProto.FLOAT}': np.float32,
    f'{onnx.TensorProto.DOUBLE}': np.float64,
    f'{onnx.TensorProto.INT32}': np.int32,
    f'{onnx.TensorProto.INT64}': np.int64,
}

ONNX_EXECUTION_PROVIDERS: dict = {
    'tensorrt': {
        'provider_info': (
            'TensorrtExecutionProvider', {
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': '',
                'trt_fp16_enable': True,
            }
        ),
        'sub_info': {},
    },
    'cuda': {
        'provider_info': 'CUDAExecutionProvider',
        'sub_info': {},
    },
    'openvino_cpu': {
        'provider_info': 'OpenVINOExecutionProvider',
        'sub_info': {
            'set_openvino_device': 'CPU_FP32',
        },
    },
    'openvino_gpu': {
        'provider_info': 'OpenVINOExecutionProvider',
        'sub_info': {
            'set_openvino_device': 'GPU_FP16',
        },
    },
    'cpu': {
        'provider_info': 'CPUExecutionProvider',
        'sub_info': {},
    },
}


def inference(
    input_onnx_file_path: str,
    batch_size: Optional[int] = 1,
    fixed_shapes: Optional[List[int]] = None,
    test_loop_count: Optional[int] = 10,
    onnx_execution_provider: Optional[str] = 'tensorrt',
    input_numpy_file_paths_for_testing: Optional[List[str]] = None,
    numpy_ndarrays_for_testing: Optional[List[np.ndarray]] = None,
    output_numpy_file: Optional[bool] = False,
    non_verbose: Optional[bool] = False,
) -> List[np.ndarray]:
    """inference

    Parameters
    ----------
    input_onnx_file_path: str
        Input onnx file path.

    batch_size: Optional[int]
        Value to be substituted if input batch size is undefined.\n\
        This is ignored if the input dimensions are all of static size.\n\
        Also ignored if input_numpy_file_paths_for_testing or\n\
        numpy_ndarrays_for_testing is specified.\n\
        Default: 1

    fixed_shapes: Optional[List[int]]
        Input OPs with undefined shapes are changed to the specified shape.\n\
        This parameter can be specified multiple times depending on the number of input OPs in the model.\n\
        Also ignored if input_numpy_file_paths_for_testing or numpy_ndarrays_for_testing is specified.\n\
        e.g.\n\
            [\n\
                [1, 3, 224, 224],\n\
                [1, 5],\n\
                [1, 1, 224, 224],\n\
            ]\n\
        Default: None

    test_loop_count: Optional[int]
        Number of times to run the test.\n\
        The total execution time is divided by the number of times the test is executed,\n\
        and the average inference time per inference is displayed.\n\
        Default: 10

    onnx_execution_provider: Optional[str]
        ONNX Execution Provider.\n\
        "tensorrt" or "cuda" or "openvino_cpu" or "openvino_gpu" or "cpu"\n\
        Default: "tensorrt"

    input_numpy_file_paths_for_testing: Optional[List[str]]
        Use an external file of numpy.ndarray saved using np.save as input data for testing.\n\
        If this parameter is specified, the value specified for batch_size and fixed_shapes are ignored.\n\
        numpy_ndarray_for_testing Cannot be specified at the same time.\n\
        For models with multiple input OPs, specify multiple numpy file paths in list format.\n\
        e.g. ['aaa.npy', 'bbb.npy', 'ccc.npy']\n\
        Default: None

    numpy_ndarrays_for_testing: Optional[List[np.ndarray]]
        Specify the numpy.ndarray to be used for inference testing.\n\
        If this parameter is specified, the value specified for batch_size and fixed_shapes are ignored.\n\
        input_numpy_file_paths_for_testing Cannot be specified at the same time.\n\
        For models with multiple input OPs, specify multiple numpy.ndarrays in list format.\n\
        e.g. [np.asarray([[[1.0],[2.0],[3.0]]], dtype=np.float32), np.asarray([1], dtype=np.int64)]\n\
        Default: None

    output_numpy_file: Optional[bool]
        Outputs the last inference result to an .npy file.\n\
        Default: False

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False

    Returns
    -------
    final_results: List[np.ndarray]
        Last Reasoning Results.
    """

    if not batch_size:
        batch_size = 1

    if not test_loop_count:
        test_loop_count = 10

    # file existence check
    if not os.path.exists(input_onnx_file_path) or \
        not os.path.isfile(input_onnx_file_path) or \
        not os.path.splitext(input_onnx_file_path)[-1] == '.onnx':

        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The specified file (.onnx) does not exist. or not an onnx file. File: {input_onnx_file_path}'
        )
        sys.exit(1)

    # test_loop_count check
    if test_loop_count and test_loop_count <= 0:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'test_loop_count must be 1 or greater.'
        )
        sys.exit(1)

    # Test Data Check
    if input_numpy_file_paths_for_testing and numpy_ndarrays_for_testing:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'Only one of input_numpy_file_path_for_testing and numpy_ndarray_for_testing can be specified.'
        )
        sys.exit(1)

    # file existence check
    if input_numpy_file_paths_for_testing:
        for input_numpy_file_path_for_testing in input_numpy_file_paths_for_testing:
            if not os.path.exists(input_numpy_file_path_for_testing) or \
                not os.path.isfile(input_numpy_file_path_for_testing):
                    print(
                        f'{Color.RED}ERROR:{Color.RESET} '+
                        f'The specified input_numpy_file_path_for_testing does not exist. File: {input_numpy_file_path_for_testing}'
                    )
                    sys.exit(1)

    # Test data load
    if input_numpy_file_paths_for_testing:
        numpy_ndarrays_for_testing = []
        for input_numpy_file_path_for_testing in input_numpy_file_paths_for_testing:
            numpy_ndarrays_for_testing.append(np.load(input_numpy_file_path_for_testing))

    # trt_engine_cache_path
    trt_engine_cache_path = os.path.dirname(input_onnx_file_path)
    if not trt_engine_cache_path:
        trt_engine_cache_path = '.'

    # Provider
    provider = ONNX_EXECUTION_PROVIDERS[onnx_execution_provider]
    provider_info = provider['provider_info']
    if onnx_execution_provider == 'tensorrt':
        provider_info[1]['trt_engine_cache_path'] = trt_engine_cache_path
    sub_info = provider['sub_info']

    # Session option
    session_option = onnxruntime.SessionOptions()
    session_option.log_severity_level = 4
    if sub_info:
        if onnx_execution_provider in ['openvino_cpu', 'openvino_gpu']:
            session_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            onnxruntime.capi._pybind_state.set_openvino_device(sub_info['set_openvino_device'])

    onnx_session = onnxruntime.InferenceSession(
        input_onnx_file_path,
        sess_options=session_option,
        providers=[provider_info],
    )

    # onnx load (For parameter acquisition)
    onnx_model = onnx.load(input_onnx_file_path)

    # Generation of dict for onnxruntime input
    ort_inputs = onnx_session.get_inputs()
    ort_outputs = onnx_session.get_outputs()
    onnx_inputs = onnx_model.graph.input

    ort_input_names = [
        ort_input.name for ort_input in ort_inputs
    ]
    ort_output_names = [
        ort_output.name for ort_output in ort_outputs
    ]

    ort_input_shapes = []

    if fixed_shapes:
        # Check if the number of input OPs in the model matches the number of inputs in the test data
        if len(ort_inputs) != len(fixed_shapes):
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'The number of input OPs in the model must match the number of test data inputs.\n'+
                f'Number of model input OPs: {len(ort_input_names)}\n'+
                f'Number of test data inputs: {len(fixed_shapes)}'
            )
            sys.exit(1)
        for fixed_shape in fixed_shapes:
            ort_input_shapes.append(fixed_shape)

    else:
        for ort_input in ort_inputs:
            input_shape = []
            for shape_idx, shape in enumerate(ort_input.shape):
                if shape_idx == 0 and (isinstance(shape, str) or shape <= 0):
                    # Batch size check
                    if batch_size <= 0:
                        print(
                            f'{Color.RED}ERROR:{Color.RESET} '+
                            f'batch_size must be 1 or greater.'
                        )
                        sys.exit(1)
                    input_shape.append(batch_size)
                else:
                    input_shape.append(shape)
            ort_input_shapes.append(input_shape)

    onnx_input_types = [
        ONNX_DTYPES_TO_NUMPY_DTYPES[f'{onnx_input.type.tensor_type.elem_type}'] for onnx_input in onnx_inputs
    ]

    input_dict = None
    if not numpy_ndarrays_for_testing and not fixed_shapes:
        input_dict = {
            ort_input_name: np.ones(
                ort_input_shape,
                onnx_input_type,
            ) for ort_input_name, ort_input_shape, onnx_input_type in zip(ort_input_names, ort_input_shapes, onnx_input_types)
        }

    elif not numpy_ndarrays_for_testing and fixed_shapes:
        input_dict = {
            ort_input_name: np.ones(
                fixed_shape,
                onnx_input_type,
            ) for ort_input_name, fixed_shape, onnx_input_type in zip(ort_input_names, fixed_shapes, onnx_input_types)
        }

    elif numpy_ndarrays_for_testing:
        # Check if the number of input OPs in the model matches the number of inputs in the test data
        if len(ort_input_names) != len(numpy_ndarrays_for_testing):
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'The number of input OPs in the model must match the number of test data inputs.\n'+
                f'Number of model input OPs: {len(ort_input_names)}\n'+
                f'Number of test data inputs: {len(numpy_ndarrays_for_testing)}'
            )
            sys.exit(1)
        input_dict = {
            ort_input_name: numpy_ndarray_for_testing \
                for ort_input_name, numpy_ndarray_for_testing in zip(ort_input_names, numpy_ndarrays_for_testing)
        }

    # Print info
    if not non_verbose:
        print(\
            f'{Color.GREEN}INFO:{Color.RESET} '+ \
            f'{Color.BLUE}file:{Color.RESET} {input_onnx_file_path}'
        )
        print(\
            f'{Color.GREEN}INFO:{Color.RESET} '+ \
            f'{Color.BLUE}providers:{Color.RESET} {onnx_session.get_providers()}'
        )
        for idx, ort_input_name, ort_input_shape, onnx_input_type in zip(range(1, len(ort_input_names)+1), ort_input_names, ort_input_shapes, onnx_input_types):
            print(\
                f'{Color.GREEN}INFO:{Color.RESET} '+ \
                f'{Color.BLUE}input_name.{idx}:{Color.RESET} {ort_input_name} '+ \
                f'{Color.BLUE}shape:{Color.RESET} {ort_input_shape} '+ \
                f'{Color.BLUE}dtype:{Color.RESET} {onnx_input_type.__name__}'
            )

    # Add one additional test for warm-up
    test_loop_count += 1
    # Init elapced time, result
    e = 0.0
    results = []

    # Inference
    for n in range(test_loop_count):
        s = time.time()
        results = onnx_session.run(
            ort_output_names,
            input_dict,
        )
        if n > 0:
            e += (time.time() - s)

    # Output numpy file
    if output_numpy_file and results:
        for ort_output_name, result in zip(ort_output_names, results):
            np.save(
                f'result_{ort_output_name.replace("/","_").replace(";","_").replace(":","_")}.npy',
                result
            )

    # Print results
    if not non_verbose:
        print(\
            f'{Color.GREEN}INFO:{Color.RESET} '+ \
            f'{Color.BLUE}test_loop_count:{Color.RESET} '+ \
            f'{test_loop_count - 1}'
        )
        print(\
            f'{Color.GREEN}INFO:{Color.RESET} '+ \
            f'{Color.BLUE}total elapsed time: {Color.RESET} {e * 1000} ms'
        )
        print(\
            f'{Color.GREEN}INFO:{Color.RESET} '+ \
            f'{Color.BLUE}avg elapsed time per pred: {Color.RESET} {e / (test_loop_count - 1) * 1000} ms'
        )
        for idx, ort_output_name, result in zip(range(1,len(ort_output_names)+1), ort_output_names, results):
            print(\
                f'{Color.GREEN}INFO:{Color.RESET} '+ \
                f'{Color.BLUE}output_name.{idx}:{Color.RESET} {ort_output_name} '+ \
                f'{Color.BLUE}shape:{Color.RESET} {[dim for dim in result.shape]} '+ \
                f'{Color.BLUE}dtype:{Color.RESET} {result.dtype}'
            )

    # Return
    return results


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='Input onnx file path.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help=\
            'Value to be substituted if input batch size is undefined. \n'+
            'This is ignored if the input dimensions are all of static size. \n'+
            'Also ignored if input_numpy_file_paths_for_testing or \n'+
            'numpy_ndarrays_for_testing or fixed_shapes is specified.'
    )
    parser.add_argument(
        '--fixed_shapes',
        type=int,
        nargs='+',
        action='append',
        help=\
            'Input OPs with undefined shapes are changed to the specified shape. \n'+
            'This parameter can be specified multiple times depending on the number of input OPs in the model. \n'+
            'Also ignored if input_numpy_file_paths_for_testing or numpy_ndarrays_for_testing is specified. \n'+
            'e.g. \n'+
            '--fixed_shapes 1 3 224 224 \n'+
            '--fixed_shapes 1 5 \n'+
            '--fixed_shapes 1 1 224 224'
    )
    parser.add_argument(
        '--test_loop_count',
        type=int,
        default=10,
        help=\
            'Number of times to run the test. \n'+
            'The total execution time is divided by the number of times the test is executed, \n'+
            'and the average inference time per inference is displayed.'
    )
    parser.add_argument(
        '--onnx_execution_provider',
        type=str,
        choices=ONNX_EXECUTION_PROVIDERS,
        default='tensorrt',
        help='ONNX Execution Provider.'
    )
    parser.add_argument(
        '--input_numpy_file_paths_for_testing',
        type=str,
        action='append',
        help=\
            'Use an external file of numpy.ndarray saved using np.save as input data for testing. \n'+
            'This parameter can be specified multiple times depending on the number of input OPs in the model. \n'+
            'If this parameter is specified, the value specified for batch_size and fixed_shapes are ignored. \n'+
            'e.g. \n'+
            '--input_numpy_file_paths_for_testing aaa.npy \n'+
            '--input_numpy_file_paths_for_testing bbb.npy \n'+
            '--input_numpy_file_paths_for_testing ccc.npy'
    )
    parser.add_argument(
        '--output_numpy_file',
        action='store_true',
        help='Outputs the last inference result to an .npy file.'
    )
    parser.add_argument(
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    input_onnx_file_path = args.input_onnx_file_path
    batch_size = args.batch_size
    fixed_shapes=args.fixed_shapes
    test_loop_count = args.test_loop_count
    onnx_execution_provider = args.onnx_execution_provider
    input_numpy_file_paths_for_testing = args.input_numpy_file_paths_for_testing
    output_numpy_file = args.output_numpy_file
    non_verbose = args.non_verbose

    final_results = inference(
        input_onnx_file_path=input_onnx_file_path,
        batch_size=batch_size,
        fixed_shapes=fixed_shapes,
        test_loop_count=test_loop_count,
        onnx_execution_provider=onnx_execution_provider,
        input_numpy_file_paths_for_testing=input_numpy_file_paths_for_testing,
        numpy_ndarrays_for_testing=None,
        output_numpy_file=output_numpy_file,
        non_verbose=non_verbose,
    )

if __name__ == '__main__':
    main()

