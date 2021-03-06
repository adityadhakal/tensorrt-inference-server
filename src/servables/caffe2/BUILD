# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package(
    default_visibility = ["//visibility:public"],
)

load("@protobuf_archive//:protobuf.bzl", "cc_proto_library")

cc_proto_library(
    name = "netdef_bundle_proto",
    srcs = ["netdef_bundle.proto"],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@protobuf_archive//:cc_wkt_protos",
    ],
)

cc_library(
    name = "autofill",
    srcs = ["autofill.cc"],
    hdrs = ["autofill.h"],
    deps = [
        "//src/core:autofill_header",
        "//src/core:constants",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

# netdef_bundle_c2 is copied into and built along with Caffe2 to
# ensure that it uses the same protobuf as Caffe2.
cc_library(
    name = "netdef_bundle",
    srcs = [
         "netdef_bundle.cc",
#        "netdef_bundle_c2.cc" - see comment above
    ],
    hdrs = [
         "netdef_bundle.h",
         "netdef_bundle_c2.h"
    ],
    deps = [
        "//src/core:backend",
        "//src/core:constants",
        "//src/core:logging",
        "//src/core:model_config_proto",
        "//src/core:model_config",
        "//src/core:provider",
        "//src/core:server_status",
        "//src/core:model_config_utils",
    ],
)

cc_library(
    name = "netdef_bundle_source_adapter",
    srcs = ["netdef_bundle_source_adapter.cc"],
    hdrs = ["netdef_bundle_source_adapter.h"],
    deps = [
        ":netdef_bundle",
        ":netdef_bundle_proto",
        "//src/core:constants",
        "//src/core:logging",
        "//src/core:model_config",
        "//src/core:model_config_proto",
         "//src/core:model_config_utils",
        "@tf_serving//tensorflow_serving/core:loader",
        "@tf_serving//tensorflow_serving/core:simple_loader",
        "@tf_serving//tensorflow_serving/core:source_adapter",
        "@tf_serving//tensorflow_serving/core:storage_path",
        "@tf_serving//tensorflow_serving/util:optional",
    ],
    alwayslink = 1,
)

filegroup(
    name = "caffe2_testdata",
    srcs = glob(
        ["testdata/**/*"]),
)

cc_test(
    name = "netdef_bundle_test",
    srcs = ["netdef_bundle_test.cc"],
    linkopts = [
        "-L/opt/tensorrtserver/lib",
        "-lcaffe2_gpu",
        "-lcaffe2",
        "-lnvinfer",
        "-L/usr/local/cuda/lib64/stubs",
        "-lnvidia-ml",
        "-lnvonnxparser_runtime",
    ],
    data = [
        ":caffe2_testdata"
    ],
    deps = [
        ":netdef_bundle",
        "//src/core:constants",
        "//src/test:model_config_test_base",
        "//src/test:testmain",
        "@local_config_cuda//cuda:cudart",
    ],
    tags = ["exclusive"],
)
