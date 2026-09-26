"""@secda_core: which SECDA-Core checkout the build uses.

In order:
  1. $SECDA_CORE_DIR, if set.
  2. Inside SECDA-DS, the suite's own SECDA-Core (<SECDA-DS>/SECDA-Core), found
     through the suite's .gitmodules. The framework repo's third_party/secda_core
     submodule is then not needed and SECDA-DS's setup.sh leaves it unchecked-out,
     so the suite holds one SECDA-Core instead of one per framework.
  3. The framework repo's third_party/secda_core submodule (a standalone clone).

A replacement for local_repository(path = ...): the checkout's top-level entries
are symlinked in, as local_repository does, minus .git and Bazel's bazel-*
convenience links, which would loop.
"""

def _secda_core_repository_impl(rctx):
    workspace = rctx.path(Label("//:WORKSPACE")).dirname
    repo = workspace
    for _ in range(rctx.attr.repo_root_up):
        repo = repo.dirname

    env = rctx.os.environ.get("SECDA_CORE_DIR", "")
    suite = repo.dirname
    gitmodules = suite.get_child(".gitmodules")
    if env:
        src = rctx.path(env)
        origin = "$SECDA_CORE_DIR"
    elif (gitmodules.exists and "path = SECDA-Core" in rctx.read(gitmodules) and
          suite.get_child("SECDA-Core").get_child("secda-core").exists):
        src = suite.get_child("SECDA-Core")
        origin = "SECDA-DS"
    else:
        src = repo.get_child("third_party").get_child("secda_core")
        origin = "third_party/secda_core"

    if not src.get_child("secda-core").exists:
        fail(("SECDA-Core not found at %s (from %s). Set SECDA_CORE_DIR, or run " +
              "`git submodule update --init third_party/secda_core`.") % (src, origin))

    for child in src.readdir():
        name = child.basename
        if name == ".git" or name.startswith("bazel-"):
            continue
        rctx.symlink(child, name)
    rctx.file("SECDA_CORE_ORIGIN", "%s\n%s\n" % (origin, src))

secda_core_repository = repository_rule(
    implementation = _secda_core_repository_impl,
    attrs = {
        # Directories from this WORKSPACE up to the framework repo's root.
        "repo_root_up": attr.int(default = 0),
    },
    environ = ["SECDA_CORE_DIR"],
    local = True,
)
