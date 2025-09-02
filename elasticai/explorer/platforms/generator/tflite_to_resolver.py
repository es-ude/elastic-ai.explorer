from tflite.Model import Model
import tflite
from pathlib import Path

# Mapping from TFLite builtin op name -> MicroMutableOpResolver method
OP_TO_RESOLVER = {
    "ADD": "AddAdd",
    "RELU": "AddRelu",
    "FULLY_CONNECTED": "AddFullyConnected",
    "RESHAPE": "AddReshape",
    "DEPTHWISE_CONV_2D": "AddDepthwiseConv2D",
    "TRANSPOSE": "AddTranspose",
    "CONV_2D": "AddConv2D",
    "LOGISTIC": "AddLogistic",
    "MAX_POOL_2D": "AddMaxPool2D",
    "AVERAGE_POOL_2D": "AddAveragePool2D",
    "SOFTMAX": "AddSoftmax",
    "MUL": "AddMul",
    "SUB": "AddSub",
    "DIV": "AddDiv",
    "EXP": "AddExp",
    "MEAN": "AddMean",
    "SQUEEZE": "AddSqueeze",
    "PAD": "AddPad",
    "RESHAPE": "AddReshape",
    "TANH": "AddTanh",
    "SIGMOID": "AddLogistic",  # logistic is sigmoid
    "ABS": "AddAbs",
    "NEG": "AddNeg",
    "SLICE": "AddSlice",
    "STRIDED_SLICE": "AddStridedSlice",
    "RELU6": "AddRelu6",
    "L2_NORMALIZATION": "AddL2Normalization",
    "CONCATENATION": "AddConcatenation",
    "TRANSPOSE_CONV": "AddTransposeConv",
    "EXPAND_DIMS": "AddExpandDims",
    "SPLIT": "AddSplit",
    "SPLIT_V": "AddSplitV",
    "UNPACK": "AddUnpack",
    "PACK": "AddPack",
    "RESIZE_NEAREST_NEIGHBOR": "AddResizeNearestNeighbor",
    "RESIZE_BILINEAR": "AddResizeBilinear",
    "CAST": "AddCast",
    "FLOOR": "AddFloor",
    "CEIL": "AddCeil",
}


def get_tflite_ops(model_path):
    with open(model_path, "rb") as f:
        buf = f.read()
    model = Model.GetRootAs(buf, 0)

    op_codes = []
    for i in range(model.OperatorCodesLength()):
        op_code = model.OperatorCodes(i)
        if op_code:
            builtin_code = op_code.BuiltinCode()
            name = tflite.opcode2name(builtin_code)
            op_codes.append(name)
    return op_codes


def generate_resolver_h(tflite_file, output_file):
    ops = get_tflite_ops(tflite_file)
    print(f"Detected ops in model: {ops}")

    resolver_calls = []
    for op in ops:
        if op in OP_TO_RESOLVER:
            resolver_calls.append(f"    resolver->{OP_TO_RESOLVER[op]}();")
        else:
            resolver_calls.append(f"    // Add mapping for {op}")

    with open(output_file, "w") as f:
        f.write("// Auto-generated resolver ops\n")
        f.write("// Generated from: {}\n".format(Path(tflite_file).name))
        f.write("\n".join(resolver_calls))
        f.write("\n")

    print(f"Generated {output_file} with {len(resolver_calls)} entries.")
