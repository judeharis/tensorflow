# Copyright Glasgow University
# All Rights Reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Author: Jude Haris

licenses(["notice"])
package(default_visibility = ["//visibility:public"])
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")



filegroup(
  name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "systemc_lib",
    lib_source =  ":all",
    build_args = ["-v -j 12"],
    cache_entries = {
        "CMAKE_CXX_STANDARD": "17",
        "BUILD_SHARED_LIBS": "off",
    },
    env = {
        "CXXFLAGS": "-fPIC",
        "CXX": "clang++",
        "CC": "clang",
    },
    includes = ["systemc-2.3.3/install/include"],
    install = True,
    out_static_libs = ["libsystemc.a"],
    working_directory = "systemc-2.3.3/",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "systemc",
    srcs = [":systemc_lib" ],
    output_group = "systemc",
    visibility = ["//visibility:public"],
)



