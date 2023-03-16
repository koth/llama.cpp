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

cc_binary(
    name = "llama",
    srcs = [
        "main.cpp",
        "utils.cpp",
        "utils.h",
    ],
    linkopts = select({
        "//conditions:default": [],
        "@bazel_tools//src/conditions:linux_x86_64": ["-lpthread"],
    }),
    deps = [
        ":ggml",
    ],
)
