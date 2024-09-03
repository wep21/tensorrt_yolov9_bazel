load("@tensorrt_yolo//repositories:repositories.bzl", "repositories")

def _non_module_deps_impl(ctx):
    repositories()

non_module_deps = module_extension(
    implementation = _non_module_deps_impl,
)
