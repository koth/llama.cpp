package(default_visibility = ["//visibility:public"])
cc_binary(
    name = "perplexity",
    srcs = [
        "perplexity.cpp",
    ],
    linkopts = select({
        "//conditions:default": [],
        "@bazel_tools//src/conditions:linux_x86_64": ["-lpthread"],
    }),
    deps = [
        "//:llama",
        "//examples:common",
    ],
)