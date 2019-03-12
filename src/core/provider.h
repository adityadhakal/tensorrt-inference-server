// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include "libevent/include/event2/buffer.h"
#include "src/core/api.pb.h"
#include "src/core/grpc_service.pb.h"
#include "src/core/model_config.h"
#include "tensorflow/core/lib/core/errors.h"

struct evbuffer;

namespace nvidia { namespace inferenceserver {

class InferenceBackend;

//
// Memory block used for accessing / storing data for providers
// Note:
//   - It is only responsible for deallocating memory obtained by ReserveBlock()
//   - The functions are not thread-safe
//
class SystemMemoryBlock {
 public:
  // Create an SystemMemoryBlock object with block size 0
  SystemMemoryBlock();

  // Create an SystemMemoryBlock that points to 'block' provided
  // 'block' is the memory block that has been allocated. For instance,
  // content returned by InferRequestProvider.GetNextInputContent().
  // 'block_size' is the size of the block
  SystemMemoryBlock(void* block, size_t block_size);

  // Reserve a block of memory with size 'byte_size'
  // 'byte_size' is the size of the memory to be reserved
  // Return error if the SystemMemoryBlock has reserved memory
  // with non-zero size
  tensorflow::Status ReserveBlock(size_t byte_size);

  // Return a pointer to the memory location. It is caller's responsibility
  // to make sure the access to the memory is exclusive on write.
  void* GetBlock() const;

  // Return the byte size of the memory
  size_t ByteSize() const { return byte_size_; }

 private:
  size_t byte_size_;
  std::unique_ptr<char[]> buffer_;
  void* referencing_block_;
};

//
// Provide inference request inputs and meta-data
//
class InferRequestProvider {
 public:
  explicit InferRequestProvider(
      const std::string& model_name, const int64_t version)
      : model_name_(model_name), version_(version)
  {
  }

  // Return the requested model name.
  const std::string& ModelName() const { return model_name_; }

  // Return the requested model version, or -1 if no specific version
  // was requested.
  int64_t ModelVersion() const { return version_; }

  // Get the request header for this inference request that has been
  // validated and normalized so that all inputs have shape and
  // batch-byte-size defined.
  const InferRequestHeader& RequestHeader() const { return request_header_; }

  // Get the next contiguous chunk of bytes for the 'name'd
  // input. Return a pointer to the chunk in 'content'.
  // 'content_byte_size' acts as both input and output. On input
  // 'content_byte_size' is a hint of the maximum chunk size that
  // should be returned in 'content' and must be non-zero unless no
  // additional input is expected. On return 'content_byte_size' gives
  // the actual size of the chunk pointed to by 'content'. If there
  // are no more bytes for the input return 'content' == nullptr. If
  // 'force_contiguous' is true then the entire (remaining) input will
  // be returned as a single chunk. In some cases this will require
  // copying the data.
  virtual tensorflow::Status GetNextInputContent(
      const std::string& name, const void** content, size_t* content_byte_size,
      bool force_contiguous) = 0;

  // Set content for named inputs. If the input already has content,
  // this content will be in-place of existing content.
  struct InputOverride {
    std::vector<uint8_t> content_;
    DimsList dims_;
    DataType datatype_;
  };

  using InputOverrideMap =
      std::unordered_map<std::string, std::shared_ptr<InputOverride>>;
  const std::shared_ptr<InputOverrideMap>& GetInputOverride() const;
  tensorflow::Status SetInputOverride(
      const std::shared_ptr<InputOverrideMap>& override);

 protected:
  // Validate request header and modify as necessary so that every
  // input has a shape and a batch-byte-size.
  tensorflow::Status NormalizeRequestHeader(const InferenceBackend& is);

  // Get the override content for 'name'd input. Return a pointer to
  // the override content in 'content'.  Return the override content
  // byte-size in 'content_byte_size'.  Return true if there is
  // override content (and so 'content' and 'content_byte_size' are
  // valid) or false if there is no override content (and so 'content'
  // and 'content_byte_size' are unchanged).
  bool GetInputOverrideContent(
      const std::string& name, const void** content, size_t* content_byte_size);

  const std::string model_name_;
  const int64_t version_;
  InferRequestHeader request_header_;

  // Input content overrides.
  std::shared_ptr<InputOverrideMap> overrides_;

  // The inputs that have had their override content consumed by a
  // call to GetInputOverrideContent. A given input override will only
  // return the content once and on subsequent calls will return
  // 'content' == nullptr to indicate that all the override content
  // has been consumed.
  std::set<std::string> overrides_consumed_;
};

//
// Inference input provider that delivers all-zero tensor
// content. This provider is only used internally to replace another
// provider for a request that is cancelled or otherwise doesn't have
// input available.
//
class NULLInferRequestProvider : public InferRequestProvider {
 public:
  explicit NULLInferRequestProvider(const InferRequestHeader& request_header)
      : InferRequestProvider("<NULL>", -1)
  {
    request_header_ = request_header;
  }

  tensorflow::Status GetNextInputContent(
      const std::string& name, const void** content, size_t* content_byte_size,
      bool force_contiguous) override;

 private:
  // A buffer of zero bytes that is used commonly as the NULL input.
  static std::vector<uint8_t> buf_;

  // Mutex to guard buf_
  static std::mutex mu_;
};

//
// Inference input provider for a GRPC inference request
//
class GRPCInferRequestProvider : public InferRequestProvider {
 public:
  // Create a GRPCInferRequestProvider object. The 'request' object is
  // captured by reference to avoid copying all the raw input tensor
  // data... but this means that it's lifetime must persist longer
  // than this provider.
  static tensorflow::Status Create(
      const InferenceBackend& is, const InferRequest& request,
      std::shared_ptr<GRPCInferRequestProvider>* infer_provider);

  tensorflow::Status GetNextInputContent(
      const std::string& name, const void** content, size_t* content_byte_size,
      bool force_contiguous) override;

 private:
  GRPCInferRequestProvider(const InferRequest& request, const int64_t version);

  const InferRequest& request_;
  std::vector<bool> content_delivered_;

  // Map from input name to the index in the request of that input.
  std::unordered_map<std::string, size_t> input_map_;
};

//
// Inference input provider for an HTTP inference request
//
class HTTPInferRequestProvider : public InferRequestProvider {
 public:
  // Initialize based on HTTP request
  static tensorflow::Status Create(
      evbuffer* input_buffer, const InferenceBackend& is,
      const std::string& model_name, const int64_t model_version,
      const std::string& request_header_str,
      std::shared_ptr<HTTPInferRequestProvider>* infer_provider);

  tensorflow::Status GetNextInputContent(
      const std::string& name, const void** content, size_t* content_byte_size,
      bool force_contiguous) override;

 private:
  HTTPInferRequestProvider(const std::string& model_name, const int64_t version)
      : InferRequestProvider(model_name, version)
  {
  }

  using Block = std::pair<const char*, size_t>;
  std::vector<std::vector<Block>> contents_;
  std::vector<size_t> contents_idx_;
  std::vector<std::vector<char>> contiguous_buffers_;

  // Map from input name to the index in contents_ that contains the
  // data for the input.
  std::unordered_map<std::string, size_t> input_map_;
};

//
// Inference input provider for an internal inference request
//
class SystemMemoryInferRequestProvider : public InferRequestProvider {
 public:
  // Create a SystemMemoryInferRequestProvider object. The 'blocks' object is
  // a mapping from input name to memory block that stores data for the input.
  static tensorflow::Status Create(
      const InferenceBackend& is, const std::string& model_name,
      const int64_t model_version, const InferRequestHeader& request_header,
      const std::unordered_map<std::string, std::shared_ptr<SystemMemoryBlock>>&
          blocks,
      std::shared_ptr<SystemMemoryInferRequestProvider>* infer_provider);

  tensorflow::Status GetNextInputContent(
      const std::string& name, const void** content, size_t* content_byte_size,
      bool force_contiguous) override;

 private:
  SystemMemoryInferRequestProvider(
      const std::string& model_name, const int64_t version)
      : InferRequestProvider(model_name, version)
  {
  }

  std::unordered_map<
      std::string, std::pair<std::shared_ptr<SystemMemoryBlock>, bool>>
      input_map_;
};

//
// Provide support for reporting inference response outputs and
// response meta-data
//
class InferResponseProvider {
 public:
  explicit InferResponseProvider(const InferRequestHeader& request_header);

  // Get the full response header for this inference request.
  virtual const InferResponseHeader& ResponseHeader() const = 0;

  // Get a mutuable full response header for this inference request.
  virtual InferResponseHeader* MutableResponseHeader() = 0;

  // Get a buffer to store results for a named output. The output must
  // be listed in the request header.
  virtual tensorflow::Status GetOutputBuffer(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape) = 0;

  // Return true if this provider requires a named output.
  bool RequiresOutput(const std::string& name);

  // Finialize response based on a servable.
  tensorflow::Status FinalizeResponse(const InferenceBackend& is);

 protected:
  struct Output;

  // Check that 'name' is a valid output. If output is to be buffered,
  // allocate space for it and point to that space with 'content'
  tensorflow::Status CheckAndSetIfBufferedOutput(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape, Output** output);

 protected:
  const InferRequestHeader& request_header_;

  // Map from output name to the InferRequestHeader output information
  // for that output.
  std::unordered_map<std::string, const InferRequestHeader::Output*>
      output_map_;

  // Information about each output.
  struct Output {
    std::string name_;
    std::vector<int64_t> shape_;
    size_t byte_size_;

    // Created buffer for non-RAW results
    std::unique_ptr<char[]> buffer_;
  };

  // Ordered list of outputs as they "added" by GetOutputBuffer().
  std::vector<Output> outputs_;
};

//
// Inference response provider for a GRPC request
//
class GRPCInferResponseProvider : public InferResponseProvider {
 public:
  // Initialize based on gRPC request
  static tensorflow::Status Create(
      const InferRequestHeader& request_header, InferResponse* response,
      std::shared_ptr<GRPCInferResponseProvider>* infer_provider);

  const InferResponseHeader& ResponseHeader() const override;
  InferResponseHeader* MutableResponseHeader() override;
  tensorflow::Status GetOutputBuffer(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape) override;

 private:
  GRPCInferResponseProvider(
      const InferRequestHeader& request_header, InferResponse* response)
      : InferResponseProvider(request_header), response_(response)
  {
  }

  InferResponse* response_;
};

//
// Inference response provider for an HTTP request
//
class HTTPInferResponseProvider : public InferResponseProvider {
 public:
  static tensorflow::Status Create(
      evbuffer* output_buffer, const InferenceBackend& is,
      const InferRequestHeader& request_header,
      std::shared_ptr<HTTPInferResponseProvider>* infer_provider);

  const InferResponseHeader& ResponseHeader() const override;
  InferResponseHeader* MutableResponseHeader() override;
  tensorflow::Status GetOutputBuffer(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape) override;

 private:
  HTTPInferResponseProvider(
      evbuffer* output_buffer, const InferRequestHeader& request_header);

  InferResponseHeader response_header_;
  evbuffer* output_buffer_;
};

//
// Inference response provider for an internal request
//
class SystemMemoryInferResponseProvider : public InferResponseProvider {
 public:
  // Create a SystemMemoryInferResponseProvider object. The 'blocks' object is
  // a mapping from output name to memory block that will store the data for
  // the output. The size and the content of the memory block may change during
  // the use of the provider, so the memory block assigned to the provider
  // shouldn't be assigned for other use until we are done with the provider
  // to avoid race condition.
  static tensorflow::Status Create(
      const InferenceBackend& is, const InferRequestHeader& request_header,
      std::unordered_map<std::string, std::shared_ptr<SystemMemoryBlock>>&
          blocks,
      std::shared_ptr<SystemMemoryInferResponseProvider>* infer_provider);

  const InferResponseHeader& ResponseHeader() const override;
  InferResponseHeader* MutableResponseHeader() override;
  tensorflow::Status GetOutputBuffer(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape) override;

 private:
  SystemMemoryInferResponseProvider(const InferRequestHeader& request_header);

  InferResponseHeader response_header_;
  std::unordered_map<std::string, std::shared_ptr<SystemMemoryBlock>>
      output_block_;
};

}}  // namespace nvidia::inferenceserver
