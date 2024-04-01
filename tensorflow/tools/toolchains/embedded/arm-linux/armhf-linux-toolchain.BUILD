package(default_visibility = ["//visibility:public"])

filegroup(
    name = "gcc",
    srcs = [
        "bin/arm-linux-gnueabihf-gcc", # Jude: Edited
    ],
)

filegroup(
    name = "ar",
    srcs = [
        "bin/arm-linux-gnueabihf-ar", # Jude: Edited
    ],
)

filegroup(
    name = "ld",
    srcs = [
        "bin/arm-linux-gnueabihf-ld", # Jude: Edited
    ],
)

filegroup(
    name = "nm",
    srcs = [
        "bin/arm-linux-gnueabihf-nm", # Jude: Edited
    ],
)

filegroup(
    name = "objcopy",
    srcs = [
        "bin/arm-linux-gnueabihf-objcopy", # Jude: Edited
    ],
)

filegroup(
    name = "objdump",
    srcs = [
        "bin/arm-linux-gnueabihf-objdump", # Jude: Edited
    ],
)

filegroup(
    name = "strip",
    srcs = [
        "bin/arm-linux-gnueabihf-strip", # Jude: Edited
    ],
)

filegroup(
    name = "as",
    srcs = [
        "bin/arm-linux-gnueabihf-as", # Jude: Edited
    ],
)

filegroup(
    name = "compiler_pieces",
    srcs = glob([
        "arm-linux-gnueabihf/**", # Jude: Edited
        "libexec/**",
        "lib/gcc/arm-linux-gnueabihf/**", # Jude: Edited
        "include/**",
    ]),
)

filegroup(
    name = "compiler_components",
    srcs = [
        ":ar",
        ":as",
        ":gcc",
        ":ld",
        ":nm",
        ":objcopy",
        ":objdump",
        ":strip",
    ],
)
