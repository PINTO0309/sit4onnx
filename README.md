# sit4onnx
Tools for simple inference testing using TensorRT, CUDA and OpenVINO CPU/GPU and CPU providers. **S**imple **I**nference **T**est for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/sit4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/sit4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/sit4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/sit4onnx?color=2BAF2B)](https://pypi.org/project/sit4onnx/) [![CodeQL](https://github.com/PINTO0309/sit4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/sit4onnx/actions?query=workflow%3ACodeQL)

<p align="center">
  <img src="https://user-images.githubusercontent.com/33194443/170160356-132ddea5-9ef1-4f93-b5cf-50764f72036d.png" />
</p>

## ToDo

- [x] Add an interface to allow arbitrary test data to be specified as input parameters.
  1. numpy.ndarray
  2. numpy file
- [x] Allow static fixed shapes to be specified when dimensions other than batch size are undefined.
- [x] Returns numpy.ndarray of the last inference result as a return value when called from a Python script.
- [x] Add `--output_numpy_file` option. Output the final inference results to a numpy file.
- [x] Add `--non_verbose` option.

## 1. Setup
### 1-1. HostPC
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
$ pip install -U onnx \
&& pip install -U sit4onnx
```
### 1-2. Docker
https://github.com/PINTO0309/simple-onnx-processing-tools#docker

## 2. CLI Usage
```
$ sit4onnx -h

usage:
  sit4onnx [-h]
  -if INPUT_ONNX_FILE_PATH
  [-b BATCH_SIZE]
  [-fs DIM0 [DIM1 DIM2 ...]]
  [-tlc TEST_LOOP_COUNT]
  [-oep {tensorrt,cuda,openvino_cpu,openvino_gpu,cpu}]
  [-iont INTRA_OP_NUM_THREADS]
  [-ifp INPUT_NUMPY_FILE_PATHS_FOR_TESTING]
  [-ofp]
  [-n]

optional arguments:
  -h, --help
      show this help message and exit.

  -if, --input_onnx_file_path INPUT_ONNX_FILE_PATH
      Input onnx file path.

  -b, --batch_size BATCH_SIZE
      Value to be substituted if input batch size is undefined.
      This is ignored if the input dimensions are all of static size.
      Also ignored if input_numpy_file_paths_for_testing
      or numpy_ndarrays_for_testing or fixed_shapes is specified.

  -fs, --fixed_shapes DIM0 [DIM1 DIM2 ...]
      Input OPs with undefined shapes are changed to the specified shape.
      This parameter can be specified multiple times depending on
      the number of input OPs in the model.
      Also ignored if input_numpy_file_paths_for_testing is specified.
      e.g.
      --fixed_shapes 1 3 224 224
      --fixed_shapes 1 5
      --fixed_shapes 1 1 224 224

  -tlc, --test_loop_count TEST_LOOP_COUNT
      Number of times to run the test.
      The total execution time is divided by the number of times the test is executed,
      and the average inference time per inference is displayed.

  -oep, --onnx_execution_provider {tensorrt,cuda,openvino_cpu,openvino_gpu,cpu}
      ONNX Execution Provider.

  -iont, --intra_op_num_threads INTRA_OP_NUM_THREADS
      Sets the number of threads used to parallelize the execution within nodes.
      Default is 0 to let onnxruntime choose.

  -ifp, --input_numpy_file_paths_for_testing INPUT_NUMPY_FILE_PATHS_FOR_TESTING
      Use an external file of numpy.ndarray saved using np.save as input data for testing.
      This parameter can be specified multiple times depending on the number of input OPs
      in the model.
      If this parameter is specified, the value specified for batch_size and fixed_shapes
      are ignored.
      e.g.
      --input_numpy_file_paths_for_testing aaa.npy
      --input_numpy_file_paths_for_testing bbb.npy
      --input_numpy_file_paths_for_testing ccc.npy

  -ofp, --output_numpy_file
      Outputs the last inference result to an .npy file.

  -n, --non_verbose
      Do not show all information logs. Only error logs are displayed.
```

## 3. In-script Usage
```python
>>> from sit4onnx import inference
>>> help(inference)

Help on function inference in module sit4onnx.onnx_inference_test:

inference(
  input_onnx_file_path: str,
  batch_size: Union[int, NoneType] = 1,
  fixed_shapes: Union[List[int], NoneType] = None,
  test_loop_count: Union[int, NoneType] = 10,
  onnx_execution_provider: Union[str, NoneType] = 'tensorrt',
  intra_op_num_threads: Optional[int] = 0,
  input_numpy_file_paths_for_testing: Union[List[str], NoneType] = None,
  numpy_ndarrays_for_testing: Union[List[numpy.ndarray], NoneType] = None,
  output_numpy_file: Union[bool, NoneType] = False,
  non_verbose: Union[bool, NoneType] = False
) -> List[numpy.ndarray]

    Parameters
    ----------
    input_onnx_file_path: str
        Input onnx file path.

    batch_size: Optional[int]
        Value to be substituted if input batch size is undefined.
        This is ignored if the input dimensions are all of static size.
        Also ignored if input_numpy_file_paths_for_testing or
        numpy_ndarrays_for_testing is specified.
        Default: 1

    fixed_shapes: Optional[List[int]]
        Input OPs with undefined shapes are changed to the specified shape.
        This parameter can be specified multiple times depending on the number of input OPs
        in the model.
        Also ignored if input_numpy_file_paths_for_testing or numpy_ndarrays_for_testing
        is specified.
        e.g.
            [
                [1, 3, 224, 224],
                [1, 5],
                [1, 1, 224, 224],
            ]
        Default: None

    test_loop_count: Optional[int]
        Number of times to run the test.
        The total execution time is divided by the number of times the test is executed,
        and the average inference time per inference is displayed.
        Default: 10

    onnx_execution_provider: Optional[str]
        ONNX Execution Provider.
        "tensorrt" or "cuda" or "openvino_cpu" or "openvino_gpu" or "cpu"
        Default: "tensorrt"

    intra_op_num_threads: Optional[int]
        Sets the number of threads used to parallelize the execution within nodes.
        Default is 0 to let onnxruntime choose.

    input_numpy_file_paths_for_testing: Optional[List[str]]
        Use an external file of numpy.ndarray saved using np.save as input data for testing.
        If this parameter is specified, the value specified for batch_size and fixed_shapes
        are ignored.
        numpy_ndarray_for_testing Cannot be specified at the same time.
        For models with multiple input OPs, specify multiple numpy file paths in list format.
        e.g. ['aaa.npy', 'bbb.npy', 'ccc.npy']
        Default: None

    numpy_ndarrays_for_testing: Optional[List[np.ndarray]]
        Specify the numpy.ndarray to be used for inference testing.
        If this parameter is specified, the value specified for batch_size and fixed_shapes
        are ignored.
        input_numpy_file_paths_for_testing Cannot be specified at the same time.
        For models with multiple input OPs, specify multiple numpy.ndarrays in list format.
        e.g.
        [
            np.asarray([[[1.0],[2.0],[3.0]]], dtype=np.float32),
            np.asarray([1], dtype=np.int64),
        ]
        Default: None

    output_numpy_file: Optional[bool]
        Outputs the last inference result to an .npy file.
        Default: False

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.
        Default: False

    Returns
    -------
    final_results: List[np.ndarray]
        Last Reasoning Results.
```

## 4. CLI Execution
```bash
$ sit4onnx \
--input_onnx_file_path osnet_x0_25_msmt17_Nx3x256x128.onnx \
--batch_size 10 \
--test_loop_count 10 \
--onnx_execution_provider tensorrt
```

## 5. In-script Execution
```python
from sit4onnx import inference

inference(
  input_onnx_file_path="osnet_x0_25_msmt17_Nx3x256x128.onnx",
  batch_size=10,
  test_loop_count=10,
  onnx_execution_provider="tensorrt",
)
```

## 6. Sample
```bash
$ sudo pip install -U sit4onnx
$ sit4onnx \
--input_onnx_file_path osnet_x0_25_msmt17_Nx3x256x128.onnx \
--batch_size 10 \
--test_loop_count 10 \
--onnx_execution_provider tensorrt
```
![image](https://user-images.githubusercontent.com/33194443/168086414-0a228097-9ffa-4088-887e-c3b7ab9fd796.png)
![1](https://user-images.githubusercontent.com/33194443/168458657-53df36fd-ad23-498f-a2ce-bcfcc38691be.gif)

```bash
$ sudo pip install -U sit4onnx
$ sit4onnx \
--input_onnx_file_path sci_NxHxW.onnx \
--fixed_shapes 100 3 224 224 \
--onnx_execution_provider tensorrt
```
![image](https://user-images.githubusercontent.com/33194443/168458796-f5d5e71d-6136-435c-a59a-98c089b38071.png)
![2](https://user-images.githubusercontent.com/33194443/168458889-2afa6d20-7132-4e53-9b22-1696bb2347b5.gif)

```bash
https://github.com/daquexian/onnx-simplifier/issues/178

$ sudo pip install -U sit4onnx
$ sit4onnx \
--input_onnx_file_path hitnet_xl_sf_finalpass_from_tf_720x1280_cast.onnx \
--onnx_execution_provider tensorrt
```
![image](https://user-images.githubusercontent.com/33194443/168459313-53f4de79-f7ce-4f09-b455-6496105c2d37.png)
![3](https://user-images.githubusercontent.com/33194443/168459950-eeed8042-fa38-414e-bc21-c102200b6c2a.gif)

## 7. Reference
1. https://github.com/onnx/onnx/blob/main/docs/Operators.md
2. https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
3. https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon
4. https://github.com/PINTO0309/simple-onnx-processing-tools
5. https://github.com/PINTO0309/PINTO_model_zoo

## 8. Issues
https://github.com/PINTO0309/simple-onnx-processing-tools/issues
