package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ggml",
    srcs = [
        "ggml.c",
    ],
    hdrs = [
        "ggml.h",
    ],
    copts=[
        "-msse4.2",
       "-mavx2",
       "-march=znver1",
    ]
)


cc_library(
    name="llama",
    srcs=[
        "llama.cpp",
    ],
    hdrs=[
        "llama.h"
    ],
    deps=[
        ":ggml",
        "@tokme//:tokme",
    ],
    
)


cc_binary(
    name = "quantize",
    srcs = [
        "quantize.cpp",
    ],
    linkopts = select({
        "//conditions:default": [],
        "@bazel_tools//src/conditions:linux_x86_64": ["-lpthread"],
    }),
    deps = [
        ":ggml",
        "//examples:common",
    ],
)
