load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

common_copts = ["-Wall"] + tflite_copts()

cc_library(
    name = "util",
    srcs = [
        "util.sc.cc",
        "util.sc.h",
        ],
    copts = common_copts,
    deps = [
        "//tensorflow/lite/kernels:cpu_backend_gemm_lib",
        "@systemc//:systemc",
    ],
)

cc_library(
    name = "systolic_lib",
    srcs = [
        "systolic.sc.cc",
        "util.sc.h",
        "sc_template.sc.cc",
        "systolic_stim.sc.cc",
        "systolic_mon.sc.cc",
        "comm_channels.sc.cc",
        "comm_channels.sc.h",
        ],
    hdrs = [
        "systolic.sc.h",
        "sc_template.sc.h",
        "systolic_stim.sc.h",
        "systolic_mon.sc.h",
    ],
    copts = common_copts,
    deps = [
        "//tensorflow/lite/kernels:cpu_backend_gemm_lib",
        "@systemc//:systemc",
    ],
)

cc_binary(
    name = "systolic_run",
    srcs = [
        "systolic_test.sc.cc",
        ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite/kernels:cpu_backend_gemm_lib",
        "//tensorflow/lite/kernels/modeling:systolic_lib",
        "@systemc//:systemc",
    ],
)

cc_binary(
    name = "channel_run",
    srcs = [
        "comm_channels_test.sc.cc",
        ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite/kernels/modeling:systolic_lib",
        "@systemc//:systemc",
    ],
)