#ifndef MM2IM_DRIVER
#define MM2IM_DRIVER

#include "acc_container.h"
#include "mm2im_util.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
#include <cstring>
#include <iostream>

#include "arm_neon.h"

#define TOG(X)

// #define TOG(X)                                                                 \
//   if (drv.verb) X;

int data_transfered = 0;
int colmap_data_sent = 0;
int write_data_recv = 0;
namespace mm2im_driver {
using namespace std;

void LoadConfig(acc_container &drv, int padded_depth) {
  prf_start(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  int opcode = 1;
  in0[inl0++] = 1;
  in0[inl0++] = padded_depth / 16;
  in0[inl0++] = drv.o2;
  in0[inl0++] = (drv.rows / drv.f);
  in0[inl0++] = drv.cols;
  in0[inl0++] = drv.ra;
  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->multi_dma_wait_send();
  prf_end(0, drv.p_t.load_config);
}

void PreloadWeights(acc_container &drv, int starting_row, int number_of_rows,
                    int padded_depth, int filter_step, int starting_filter) {
  prf_start(0);
  int offset = 204800;
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer() + (offset / 4);
  int inl0 = 0;
  int padded_depth_4 = padded_depth / 4;
  int opcode = 2;
  int wgt_packet_a = number_of_rows;
  int wgt_packet_b = padded_depth_4 / 4;

  in0[inl0++] = opcode;
  in0[inl0++] = wgt_packet_a;
  in0[inl0++] = wgt_packet_b;
  in0[inl0++] = filter_step;
  int *lweights = drv.loaded_weights;
  for (int i = 0; i < number_of_rows; i++) {
    int src_addr = (starting_row + i) * padded_depth_4;
    memcpy(&in0[inl0], &lweights[src_addr], padded_depth_4 * 4);
    inl0 += padded_depth_4;
    in0[inl0++] = drv.acc_wt_sum[starting_row + i] * drv.rhs_offset;
  }

  // Send Bias
  for (int i = 0; i < filter_step; i++) {
    in0[inl0++] = drv.bias[starting_filter + i];
    in0[inl0++] = drv.crf[starting_filter + i];
    in0[inl0++] = (int)drv.crx[starting_filter + i];
  }
  drv.wgt_inl0 = inl0;
  prf_end(0, drv.p_t.load_wgt);
}

void SendWeights(acc_container &drv) {
  prf_start(0);
  drv.mdma->multi_dma_change_start(204800);
  drv.mdma->dmas[0].dma_start_send(drv.wgt_inl0);
  data_transfered += drv.wgt_inl0;
  drv.mdma->multi_dma_wait_send();
  prf_end(0, drv.p_t.load_wgt);
}

void LoadWeight(acc_container &drv, int starting_row, int number_of_rows,
                int padded_depth, int filter_step, int starting_filter) {
  prf_start(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  int padded_depth_4 = padded_depth / 4;
  int opcode = 2;
  int wgt_packet_a = number_of_rows;
  int wgt_packet_b = padded_depth_4 / 4;

  in0[inl0++] = opcode;
  in0[inl0++] = wgt_packet_a;
  in0[inl0++] = wgt_packet_b;
  in0[inl0++] = filter_step;
  int *lweights = drv.loaded_weights;
  for (int i = 0; i < number_of_rows; i++) {
    int src_addr = (starting_row + i) * padded_depth_4;
    memcpy(&in0[inl0], &lweights[src_addr], padded_depth_4 * 4);
    inl0 += padded_depth_4;
    in0[inl0++] = drv.acc_wt_sum[starting_row + i] * drv.rhs_offset;
  }
  for (int i = 0; i < filter_step; i++) {
    in0[inl0++] = drv.bias[starting_filter + i];
    in0[inl0++] = drv.crf[starting_filter + i];
    in0[inl0++] = (int)drv.crx[starting_filter + i];
  }
  drv.mdma->dmas[0].dma_start_send(inl0);
  data_transfered += inl0;
  drv.mdma->multi_dma_wait_send();
  prf_end(0, drv.p_t.load_wgt);
}

void StartSchedule(acc_container &drv) {
  prf_start(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  int opcode = 16;
  in0[inl0++] = opcode;
  drv.mdma->dmas[0].dma_start_send(inl0);
  data_transfered += inl0;
  drv.mdma->multi_dma_wait_send();
  prf_end(0, drv.p_t.start_sched);
}

void LoadInput(acc_container &drv, int starting_row, int number_of_rows,
               int padded_depth) {
  prf_start(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  int padded_depth_4 = padded_depth / 4;
  int inp_packet_a = number_of_rows;
  int inp_packet_b = padded_depth_4 / 4;
  int opcode = 4 + 16;
  in0[inl0++] = opcode;
  in0[inl0++] = inp_packet_a;
  in0[inl0++] = inp_packet_b;
  for (int i = 0; i < number_of_rows; i++) {
    int src_addr = (starting_row + i) * padded_depth_4;
    memcpy(&in0[inl0], &drv.loaded_inputs[src_addr], padded_depth_4 * 4);
    inl0 += padded_depth_4;
  }

  drv.mdma->dmas[0].dma_start_send(inl0);
  data_transfered += inl0;
  drv.mdma->multi_dma_wait_send();
  prf_end(0, drv.p_t.load_inp);
}

void LoadColMap(acc_container &drv, int starting_row, int number_of_rows) {
  prf_start(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  int opcode = 32 + 16;
  in0[inl0++] = opcode;
  in0[inl0++] = number_of_rows;
  // vector<vector<int>> &col_dexs = *drv.col_dexs;
  // vector<vector<int>> &out_dexs = *drv.out_dexs;
  // for (int i = starting_row; i < starting_row + number_of_rows; i++) {
  //   vector<int> col_dex_of_row = col_dexs[i];
  //   vector<int> out_dex_of_row = out_dexs[i];
  //   int col_dex_size = col_dex_of_row.size();
  //   in0[inl0++] = col_dex_size;
  //   for (int j = 0; j < col_dex_size; j++) {
  //     in0[inl0++] = col_dex_of_row[j];
  //     in0[inl0++] = out_dex_of_row[j];
  //   }
  // }

  for (int i = starting_row; i < starting_row + number_of_rows; i++) {
    vector<int> *col_dex_of_row = &(drv.col_dexs)->at(i);
    vector<int> *out_dex_of_row = &(drv.out_dexs)->at(i);
    int col_dex_size = col_dex_of_row->size();
    in0[inl0++] = col_dex_size;
    for (int j = 0; j < col_dex_size; j++) {
      in0[inl0++] = col_dex_of_row->at(j);
      in0[inl0++] = out_dex_of_row->at(j);
    }
  }

  drv.mdma->dmas[0].dma_start_send(inl0);
  data_transfered += inl0;
  colmap_data_sent += inl0;
  drv.mdma->multi_dma_wait_send();
  prf_end(0, drv.p_t.load_colmap);
}

void StoreOutTileRow(acc_container &drv, int o_1, int o_3, int filter_step,
                     bool last) {

  
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  int schedule = last ? 0 : 16;
  int opcode = 64 + schedule;
  in0[inl0++] = opcode;
  in0[inl0++] = o_1 * drv.o2;
  in0[inl0++] = drv.o2;
  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->multi_dma_wait_send();

  prf_start(0);
  drv.mdma->dmas[0].dma_start_recv(drv.o2 * filter_step + PE_COUNT + 1);
  drv.mdma->multi_dma_wait_recv();
  prf_end(0, drv.p_t.store);
  
  int *out0 = drv.mdma->dmas[0].dma_get_outbuffer();
  int outl0 = 0;
  for (int o_2 = 0; o_2 < drv.o2; o_2++) {
    int o_dex = ((o_1 * drv.o2) + o_2) * drv.o3 + o_3;
    memcpy(&drv.output_data[o_dex], &out0[outl0], filter_step);
    outl0 += filter_step / 4;
  }
  write_data_recv += outl0;
  
}

// Current Driver assumes filter_step is divisible by x  (x = 2 currently)
// We need to handle the case where it is not
// By processing the last filter_step % x filters separately
void TileMM2IM(acc_container &drv, int padded_depth) {
  // prf_start(1);
  int output_rows = drv.o1 * drv.o2;
  int output_cols = drv.o3;

  // weight params
  int data_per_filter = drv.ks * drv.ks * padded_depth;
  int total_weights = data_per_filter * output_cols;
  int rows_per_filter = drv.ks * drv.ks;
  int acc_max_weight_rows = WGT_BUF_LEN * UF / padded_depth;
  int total_weight_rows = total_weights / padded_depth;
  int max_weight_rows = min(total_weight_rows, acc_max_weight_rows);
  int filter_step = min(max_weight_rows / rows_per_filter, PE_COUNT);
  if (rows_per_filter >= max_weight_rows) {
    cerr << "Warning: rows_per_filter: " << rows_per_filter
         << " >= max_weight_rows: " << max_weight_rows << endl;
    filter_step = PE_COUNT;
  }

  // input params
  int max_input_rows_per_output = (drv.ks * drv.ks) / (drv.sx * drv.sy);
  int total_inputs = drv.ih * drv.iw * padded_depth;
  int acc_max_input_rows = INP_BUF_LEN * UF / padded_depth;
  int total_input_rows = total_inputs / padded_depth;
  int max_input_rows = min(total_input_rows, acc_max_input_rows);
  int input_steps = min(max_input_rows, max_input_rows_per_output);

  int padded_out_width = drv.o2 + drv.pl + drv.pr;
  int padded_out_height = drv.o1 + drv.pt + drv.pb;
  int noOfStepsX = nofSteps(padded_out_width, drv.sx, drv.ks);
  int noOfStepsY = nofSteps(padded_out_height, drv.sy, drv.ks);
  int max_input_rows_per_o1 = noOfStepsX * ceiling(drv.ks, drv.sy);
  if (max_input_rows != total_input_rows &&
      max_input_rows_per_output > max_input_rows)
    cerr << "Warning: max_input_rows_per_output: " << max_input_rows_per_output
         << " > max_input_rows: " << max_input_rows << endl;

  if (max_input_rows != total_input_rows &&
      max_input_rows_per_o1 > max_input_rows)
    cerr << "Warning: max_input_rows_per_o1: " << max_input_rows_per_o1
         << " > max_input_rows: " << max_input_rows << endl;
  int input_o1_steps = min(max_input_rows, max_input_rows_per_o1);

  // filter params
  int remaining_filters = output_cols % filter_step;
  int acc_filters = output_cols - remaining_filters;

  // ==============================================
  drv.validate();
  TOG(cerr << "Starting  MM2IM" << endl;);
  if (acc_filters > 0) LoadConfig(drv, padded_depth);
  int o_3 = 0;
  for (; o_3 < acc_filters; o_3 += filter_step) {
    // Send filter_step * rows_per_filter  rows of weights to accelerator
    TOG(cerr << "Sending weights: " << o_3 * rows_per_filter << " to "
             << (o_3 * rows_per_filter + filter_step * rows_per_filter)
             << endl;);
    LoadWeight(drv, o_3 * rows_per_filter, filter_step * rows_per_filter,
               padded_depth, filter_step, o_3);
    int starting = 0;
    StartSchedule(drv);
    for (int o_1 = 0; o_1 < drv.o1; o_1++) {
      TOG(cerr << "Sending rows: " << starting << " to " << drv.o1_ends[o_1] + 1
               << endl;);
      int rows_to_send = drv.o1_ends[o_1] + 1 - starting;
      if (drv.o1_ends[o_1] != starting - 1) {
        LoadInput(drv, starting, rows_to_send, padded_depth);
        TOG(cerr << "Input loaded" << endl;);
        LoadColMap(drv, starting, rows_to_send);
        TOG(cerr << "ColMap loaded" << endl;);
      }
      // Last ejects acc from schedule mode
      bool last = (o_1 == drv.o1 - 1);
      StoreOutTileRow(drv, o_1, o_3, filter_step, last);
      TOG(cerr << "Store loaded" << endl;);
      starting = drv.o1_ends[o_1] + 1;
    }
  }

  // Handle remaining filters
  prf_start(0);
  vector<vector<int>> &mm2im_map = *drv.mm2im_map;
  for (; o_3 < output_cols; o_3++) {
    for (int o_1 = 0; o_1 < drv.o1; o_1++) {
      for (int o_2 = 0; o_2 < drv.o2; o_2++) {
        int o_dex = ((o_1 * drv.o2) + o_2) * output_cols + o_3;
        int32_t sum = 0;
        for (int i = 0; i < mm2im_map[o_dex].size(); i++) {
          int orow = mm2im_map[o_dex][i] % drv.rows;
          int ocol = mm2im_map[o_dex][i] / drv.rows;
          for (int d = 0; d < drv.depth; d++) {
            int weight_index = orow * drv.depth + d;
            int input_index = ocol * drv.depth + d;
            int weight = drv.weights[weight_index];
            int input = drv.inputs[input_index];
            sum += weight * input;
          }
          // int offset = 0;
          int offset = drv.acc_wt_sum[orow] * drv.rhs_offset;
          sum += offset;
        }
        int bias = drv.bias[o_3];
        int crf_data = drv.crf[o_3];
        int crx_data = drv.crx[o_3];
        int qm_ret =
            drv.ra + CPU_Quantised_Multiplier(sum + bias, crf_data, crx_data);
        if (qm_ret > MAX8) qm_ret = MAX8;
        else if (qm_ret < MIN8) qm_ret = MIN8;
        drv.output_data[o_dex] = qm_ret;
      }
    }
  }
  prf_end(0, drv.p_t.handle_rest);
  // prf_end(1, drv.p_t.driver_total);
};

void Entry(acc_container &drv) {
  prf_start(1);
  int rrows = roundUp(drv.rows, 4);
  int rcols = roundUp(drv.cols, 4);
  int padded_depth = roundUp(drv.depth, 16);
  int output_stride = drv.cols;
  TileMM2IM(drv, padded_depth);
  cerr << "Data transfered: " << data_transfered * 4 << endl;
  cerr << "Colmap data sent: " << colmap_data_sent * 4 << endl;
  cerr << "Write data recv: " << write_data_recv * 4 << endl;
  prf_end(1, drv.p_t.driver_total);
}

} // namespace mm2im_driver

#endif // MM2IM_DRIVER