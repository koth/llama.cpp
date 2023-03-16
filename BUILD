package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ggml",
    srcs = [
        "ggml.c",
    ],
    hdrs = [
        "ggml.h",
    ],
)


cc_library(
    name = "utils",
    srcs = [
        "utils.cpp",
    ],
    hdrs = [
        "utils.h",
    ],
)

cc_binary(
    name = "llama",
    srcs = [
        "main.cpp",
    ],
    linkopts = select({
        "//conditions:default": [],
        "@bazel_tools//src/conditions:linux_x86_64": ["-lpthread"],
    }),
    deps = [
        ":ggml",
        ":utils"
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
        ":utils",
    ],
)
