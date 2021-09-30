/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/examples/label_image_secda/label_image.h"

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/examples/label_image_secda/bitmap_helpers.h"
#include "tensorflow/lite/examples/label_image_secda/get_top_n.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

#include "arm_neon.h"


#include <sys/mman.h>
#include <errno.h>
#include <stdio.h>
#include <termios.h>
#include <chrono>
#include <typeinfo>
#include "tensorflow/lite/examples/label_image_secda/gemm_driver.h"
#define LOG(x) std::cerr

namespace tflite {
namespace label_image_secda {

using namespace std;
using namespace std::chrono;

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

TfLiteDelegatePtr CreateGPUDelegate(Settings* s) {
#if defined(__ANDROID__)
  TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
  gpu_opts.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  gpu_opts.inference_priority1 =
      s->allow_fp16 ? TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY
                    : TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  return evaluation::CreateGPUDelegate(s->model, &gpu_opts);
#else
  return evaluation::CreateGPUDelegate(s->model);
#endif
}

TfLiteDelegatePtrMap GetDelegates(Settings* s) {
  TfLiteDelegatePtrMap delegates;
  if (s->gl_backend) {
    auto delegate = CreateGPUDelegate(s);
    if (!delegate) {
      LOG(INFO) << "GPU acceleration is unsupported on this platform.";
    } else {
      delegates.emplace("GPU", std::move(delegate));
    }
  }

  if (s->accel) {
    auto delegate = evaluation::CreateNNAPIDelegate();
    if (!delegate) {
      LOG(INFO) << "NNAPI acceleration is unsupported on this platform.";
    } else {
      delegates.emplace("NNAPI", evaluation::CreateNNAPIDelegate());
    }
  }
  return delegates;
}

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(FATAL) << "Labels file " << file_name << " not found\n";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

void PrintProfilingInfo(const profiling::ProfileEvent* e,
                        uint32_t subgraph_index, uint32_t op_index,
                        TfLiteRegistration registration) {
  // output something like
  // time (ms) , Node xxx, OpCode xxx, symblic name
  //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D

  LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
            << (e->end_timestamp_us - e->begin_timestamp_us) / 1000.0
            << ", Subgraph " << std::setw(3) << std::setprecision(3)
            << subgraph_index << ", Node " << std::setw(3)
            << std::setprecision(3) << op_index << ", OpCode " << std::setw(3)
            << std::setprecision(3) << registration.builtin_code << ", "
            << EnumNameBuiltinOperator(
                   static_cast<BuiltinOperator>(registration.builtin_code))
            << "\n";
}

void RunInference(Settings* s) {
  if (!s->model_name.c_str()) {
    LOG(ERROR) << "no model file name\n";
    exit(-1);
  }

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
    exit(-1);
  }
  s->model = model.get();
  LOG(INFO) << "Loaded model " << s->model_name << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }

  interpreter->UseNNAPI(s->old_accel);
  interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);

  if (s->verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
    LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }
  }

  if (s->number_of_threads != -1) {
    interpreter->SetNumThreads(s->number_of_threads);
  }

  int image_width = 224;
  int image_height = 224;
  int image_channels = 3;
  std::vector<uint8_t> in = read_bmp(s->input_bmp_name, &image_width,
                                     &image_height, &image_channels, s);

  int input = interpreter->inputs()[0];
  if (s->verbose) LOG(INFO) << "input: " << input << "\n";

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  if (s->verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
  }

  auto delegates_ = GetDelegates(s);
  for (const auto& delegate : delegates_) {
    if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) !=
        kTfLiteOk) {
      LOG(FATAL) << "Failed to apply " << delegate.first << " delegate.";
    } else {
      LOG(INFO) << "Applied " << delegate.first << " delegate.";
    }
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  if (s->verbose) PrintInterpreterState(interpreter.get());

  // get input dimension from the input tensor metadata
  // assuming one input only
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];

  if(wanted_channels!=3){
    wanted_height = dims->data[3];
    wanted_width = dims->data[2];
    wanted_channels = dims->data[1];
  }

  switch (interpreter->tensor(input)->type) {
    case kTfLiteFloat32:
      s->input_floating = true;
      resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels, s);
      break;
    case kTfLiteUInt8:
      resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
                      image_height, image_width, image_channels, wanted_height,
                      wanted_width, wanted_channels, s);
      break;
    default:
      LOG(FATAL) << "cannot handle input type "
                 << interpreter->tensor(input)->type << " yet";
      exit(-1);
  }

  auto profiler =
      absl::make_unique<profiling::Profiler>(s->max_profiling_buffer_entries);
  interpreter->SetProfiler(profiler.get());

    if (s->profiling) profiler->StartProfiling();
  if (s->loop_count > 1)
    for (int i = 0; i < s->number_of_warmup_runs; i++) {
      if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!\n";
      }
    }

//SECDA <-------------------------------->
  int* acc = getArray<int>(acc_address,page_size);
  int dh = open("/dev/mem", O_RDWR | O_SYNC);

  void *dma_mm0 = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_SHARED, dh, dma_addr0); // Memory map AXI Lite register block
  void *dma_mm1 = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_SHARED, dh, dma_addr1); // Memory map AXI Lite register block
  void *dma_mm2 = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_SHARED, dh, dma_addr2); // Memory map AXI Lite register block
  void *dma_mm3 = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_SHARED, dh, dma_addr3); // Memory map AXI Lite register block
  void *dma_in_mm0  = mmap(NULL, dma_buffer_len, PROT_READ | PROT_WRITE, MAP_SHARED, dh, dma_addr_in0); // Memory map source address
  void *dma_in_mm1  = mmap(NULL, dma_buffer_len, PROT_READ | PROT_WRITE, MAP_SHARED, dh, dma_addr_in1); // Memory map source address
  void *dma_in_mm2  = mmap(NULL, dma_buffer_len, PROT_READ | PROT_WRITE, MAP_SHARED, dh, dma_addr_in2); // Memory map source address
  void *dma_in_mm3  = mmap(NULL, dma_buffer_len, PROT_READ | PROT_WRITE, MAP_SHARED, dh, dma_addr_in3); // Memory map source address  
  void *dma_out_mm0 = mmap(NULL, dma_buffer_len, PROT_READ, MAP_SHARED, dh, dma_addr_out0); // Memory map destination address
  void *dma_out_mm1 = mmap(NULL, dma_buffer_len, PROT_READ, MAP_SHARED, dh, dma_addr_out1); // Memory map destination address
  void *dma_out_mm2 = mmap(NULL, dma_buffer_len, PROT_READ, MAP_SHARED, dh, dma_addr_out2); // Memory map destination address
  void *dma_out_mm3 = mmap(NULL, dma_buffer_len, PROT_READ, MAP_SHARED, dh, dma_addr_out3); // Memory map destination address

  unsigned int* dma0 =reinterpret_cast<unsigned int*> (dma_mm0);
  unsigned int* dma1 =reinterpret_cast<unsigned int*> (dma_mm1);
  unsigned int* dma2 =reinterpret_cast<unsigned int*> (dma_mm2);
  unsigned int* dma3 =reinterpret_cast<unsigned int*> (dma_mm3);
  unsigned int* dma_in0 =reinterpret_cast<unsigned int*> (dma_in_mm0);
  unsigned int* dma_in1 =reinterpret_cast<unsigned int*> (dma_in_mm1);
  unsigned int* dma_in2 =reinterpret_cast<unsigned int*> (dma_in_mm2);
  unsigned int* dma_in3 =reinterpret_cast<unsigned int*> (dma_in_mm3);
  int* dma_out0 =reinterpret_cast<int*> (dma_out_mm0);
  int* dma_out1 =reinterpret_cast<int*> (dma_out_mm1);
  int* dma_out2 =reinterpret_cast<int*> (dma_out_mm2);
  int* dma_out3 =reinterpret_cast<int*> (dma_out_mm3);

  initDMA<int>(dma0,dma_addr_in0,dma_addr_out0);
  initDMA<int>(dma1,dma_addr_in1,dma_addr_out1);
  initDMA<int>(dma2,dma_addr_in2,dma_addr_out2);
  initDMA<int>(dma3,dma_addr_in3,dma_addr_out3);

  //Weights
  vector<uint8_t> wb0;
  vector<uint8_t> wb1;
  vector<uint8_t> wb2;
  vector<uint8_t> wb3;
  vector<int> wb_dex;

  //Weight Sums
  vector<int> wt_sum1;
  vector<int> wt_sum2;
  vector<int> wt_sum3;
  vector<int> wt_sum4;
  vector<int> dt_sum_dex;

  //Pre-Loads Weight Data into Temporary Buffers
  //Temp Weight vars
  int w_c = 0;
  int sums_curr=0;
  dt_sum_dex.push_back(0);
  wb_dex.push_back(0);
  for (int l=0 ; l <interpreter->primary_subgraph().nodes_size();l++){
    if (interpreter->primary_subgraph().node_and_registration(l)->second.builtin_code==3){
      int weight_tensor_dex= interpreter->primary_subgraph().node_and_registration(l)->first.inputs->data[1];
      auto tensor = interpreter->tensor(weight_tensor_dex);
      uint8_t* weight_data = tensor->data.uint8;
      int* dims = tensor->dims->data;
      preload_weights<int>(weight_data,dims,wb0,wb1,wb2,wb3,wb_dex,wt_sum1,wt_sum2,wt_sum3,wt_sum4,dt_sum_dex,w_c,sums_curr);
    }
  }

  unsigned int* wb_0 = reinterpret_cast<unsigned int*> (&wb0[0]);
  unsigned int* wb_1 = reinterpret_cast<unsigned int*> (&wb1[0]);
  unsigned int* wb_2 = reinterpret_cast<unsigned int*> (&wb2[0]);
  unsigned int* wb_3 = reinterpret_cast<unsigned int*> (&wb3[0]);

  pthread_t  tid = pthread_self();
  int runs = s->trys;
  vector<int> overalls;
  vector<int> convs;
  vector<int> other_layers;
  vector<int> gemm_times;
  vector<int> miscs;

  cout << "Press Enter to Go";
  cin.ignore();
  for(int i =0;i<runs;i++){
    int bufflen = 10;
    struct dma_in_buffer dinb[bufflen];
    struct store_params st_params[bufflen];
    for(int i=0;i<bufflen;i++) dinb[i].offset = 200000*i;

    struct gemm_driver gd(acc,s->accon,
      dma0,dma1,dma2,dma3,
      dma_in0,dma_in1,dma_in2,dma_in3,
      dma_out0,dma_out1,dma_out2,dma_out3,
      wb_0,wb_1,wb_2,wb_3,wb_dex,st_params,
      wt_sum1,wt_sum2,wt_sum3,wt_sum4,dt_sum_dex,
      dinb,bufflen,tid);
    gd.t.profile = s->acc_prof;

    gd.t.layer_print = 5;
    gd.t.layer_ww = 0;
    gd.t.layer_iw = 0;

    struct timeval start_time, stop_time;
    gettimeofday(&start_time, nullptr);
    if (interpreter->Invoke2(gd) != kTfLiteOk) LOG(FATAL) << "Failed to invoke tflite!\n";
    gettimeofday(&stop_time, nullptr);
    cout << "Run " << i << " Complete" << endl;

    overalls.push_back((get_us(stop_time) - get_us(start_time)) / (1000));
    convs.push_back(chrono::duration_cast<chrono::milliseconds>(gd.t.convtime).count());
    other_layers.push_back(overalls[i]-convs[i]);
    gemm_times.push_back(chrono::duration_cast<chrono::milliseconds>(gd.t.acctime).count());	
    if(s->accon)miscs.push_back(convs[i] - gemm_times[i]);
    else miscs.push_back(0);
  }

  int overall=0;
  int conv=0;	
  int other_layer=0;
  int gemm_time=0;
  int misc=0;

  if(s->acc_prof || s->acc_store){
    string accname ="sa_uint8_v1_01";
#ifdef VM_ACC
    accname = "vm_uint8_v1_01";
#endif

    std::string mname = s->model_name;
    std::string delimiter = "/";
    size_t pos = 0;
    std::string token;
    while ((pos = mname.find(delimiter)) != std::string::npos) {
      token = mname.substr(0, pos);
      mname.erase(0, pos + delimiter.length());
    }

    delimiter = ".";
    while ((pos = mname.find(delimiter)) != std::string::npos) {
      token = mname.substr(0, pos);
      mname.erase(pos,mname.length());
    }

    for(int i =0;i<runs;i++){
      overall += overalls[i];
      conv += convs[i];
      other_layer += other_layers[i];
      gemm_time += gemm_times[i];
      misc += miscs[i];
    }
    overall = overall/runs;
    conv =  conv/runs;
    other_layer =  other_layer/runs;
    gemm_time =  gemm_time/runs;
    misc = misc/runs;

    cout << "Overall: "<< overall << " ms \n";
    cout << "Convolution time : " << conv	<< " ms" << endl;
    cout << "Other layers' time : " <<  other_layer	<< " ms" << endl;
    cout << "GEMM time: " << gemm_time	<< " ms" << endl;
    cout << "Misc time: " << misc	<< " ms" << endl;
    cout << "-----------------------------" << endl;
    cout << "Model: " << mname << endl;
    cout << "Threads: " << s->number_of_threads << endl;
    cout << "Accelerated: " << s->accon << endl;
    cout << "Accelerator: " << accname << endl;
    cout << "Driver: " << accname + "_d1" << endl;
    cout << "-----------------------------" << endl;
    if(s->acc_store){
      ofstream profile_file;
      string acceled = s->accon?"_acc":"_cpu";
      string outfilename =  mname+ "_t" + to_string(s->number_of_threads) + acceled + "_" + accname + "_d1";
      profile_file.open(outfilename +".csv");
      profile_file << "run" << ",";
      profile_file << "gemm_times" << ",";
      profile_file << "miscs" << ",";
      profile_file << "convs" << ",";
      profile_file << "other_layers" << ",";
      profile_file << "overalls" << ",";
      profile_file << "model" << ",";
      profile_file << "threads" << ",";
      profile_file << "accelerated" << ",";
      profile_file << "accelerator" << ",";
      profile_file << "driver" << ",";
      profile_file << endl;
      for(int i =0;i<runs;i++){
        overall += overalls[i];
        conv += convs[i];
        other_layer += other_layers[i];
        gemm_time += gemm_times[i];
        misc += miscs[i];
        profile_file << i << ",";
        profile_file << gemm_times[i] << ",";
        profile_file << miscs[i] << ",";
        profile_file << convs[i] << ",";
        profile_file << other_layers[i] << ",";
        profile_file << overalls[i] << ",";
        profile_file << mname << ",";
        profile_file << s->number_of_threads << ",";
        profile_file << s->accon << ",";
        profile_file << accname << ",";
        profile_file << accname + "_d1" << ",";
        profile_file << endl;
      }
      profile_file.close();
    }
  }
  //SECDA <-------------------------------->



  if (s->profiling) {
    profiler->StopProfiling();
    auto profile_events = profiler->GetProfileEvents();
    for (int i = 0; i < profile_events.size(); i++) {
      auto subgraph_index = profile_events[i]->event_subgraph_index;
      auto op_index = profile_events[i]->event_metadata;
      const auto subgraph = interpreter->subgraph(subgraph_index);
      const auto node_and_registration =
          subgraph->node_and_registration(op_index);
      const TfLiteRegistration registration = node_and_registration->second;
      PrintProfilingInfo(profile_events[i], subgraph_index, op_index,
                         registration);
    }
  }

  const float threshold = 0.001f;

  std::vector<std::pair<float, int>> top_results;



  int output = interpreter->outputs()[0];
  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];
  switch (interpreter->tensor(output)->type) {
    case kTfLiteFloat32:
      get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                       s->number_of_results, threshold, &top_results, true);
      break;
    case kTfLiteUInt8:
      get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                         output_size, s->number_of_results, threshold,
                         &top_results, false);
      break;
    default:
      LOG(FATAL) << "cannot handle output type "
                 << interpreter->tensor(input)->type << " yet";
      exit(-1);
  }
  std::vector<string> labels;
  size_t label_count;


  if (ReadLabelsFile(s->labels_file_name, &labels, &label_count) != kTfLiteOk)exit(-1);
  
  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;
    LOG(INFO) << confidence << ": " << index << " " << labels[index] << "\n";
  }
}

void display_usage() {
  LOG(INFO)
      << "label_image\n"
      << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
      << "--old_accelerated, -d: [0|1], use old Android NNAPI delegate or not\n"
      << "--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not\n"
      << "--count, -c: loop interpreter->Invoke() for certain times\n"
      << "--gl_backend, -g: use GL GPU Delegate on Android\n"
      << "--input_mean, -b: input mean\n"
      << "--input_std, -s: input standard deviation\n"
      << "--image, -i: image_name.bmp\n"
      << "--labels, -l: labels for the model\n"
      << "--tflite_model, -m: model_name.tflite\n"
      << "--profiling, -p: [0|1], profiling or not\n"
      << "--num_results, -r: number of results to show\n"
      << "--threads, -t: number of threads\n"
      << "--verbose, -v: [0|1] print more information\n"
      << "--warmup_runs, -w: number of warmup runs\n"
      << "\n";
}

int Main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"accelerated", required_argument, nullptr, 'a'},
        {"old_accelerated", required_argument, nullptr, 'd'},
        {"allow_fp16", required_argument, nullptr, 'f'},
        {"count", required_argument, nullptr, 'c'},
        {"verbose", required_argument, nullptr, 'v'},
        {"image", required_argument, nullptr, 'i'},
        {"labels", required_argument, nullptr, 'l'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"profiling", required_argument, nullptr, 'p'},
        {"threads", required_argument, nullptr, 't'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"num_results", required_argument, nullptr, 'r'},
        {"max_profiling_buffer_entries", required_argument, nullptr, 'e'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {"gl_backend", required_argument, nullptr, 'g'},
        {"accon", required_argument, nullptr, 'z'},
        {"cmd_var", required_argument, nullptr, 'y'},
        {"verb", required_argument, nullptr, 'x'},
        {"output", required_argument, nullptr, 'o'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:b:c:d:e:f:g:i:l:m:p:r:s:t:v:w:z:o:n:x:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.accel = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'd':
        s.old_accel =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'e':
        s.max_profiling_buffer_entries =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'f':
        s.allow_fp16 =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'g':
        s.gl_backend =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_bmp_name = optarg;
        break;
      case 'l':
        s.labels_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'p':
        s.profiling =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'r':
        s.number_of_results =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'z':
        s.accon =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'o':
        s.acc_prof = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn);
        break;
      case 'x':
        s.acc_store = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn);
        break;
      case 'n':
        s.trys =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
      default:
        exit(-1);
    }
  }
  RunInference(&s);
  return 0;
}

}  // namespace label_image_secda
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::label_image_secda::Main(argc, argv);
}
