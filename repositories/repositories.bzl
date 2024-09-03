load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repositories():
    """Import third party repositories."""
    maybe(
        new_local_repository,
        name = "opencv",
        build_file = "//repositories:opencv.BUILD.bazel",
        path = "/usr",
    )
