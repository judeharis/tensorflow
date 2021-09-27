// Created by Nicolas Agostini
#include <systemc/systemc.h>
#include "tensorflow/lite/kernels/modeling/comm_channels.sc.h"

// File adapted from:
// https://sclive.wordpress.com/2008/01/11/systemc-tutorial-interfaces-and-channels-2/

namespace tflite_soc {

class test_bench : public sc_module {
 public:
  sc_port<dma_interface> master_port;

  void stimuli() {
    sc_lv<8> data_sent[10] = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
    sc_lv<8> data_rcv[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    master_port->BurstWrite(100, 10, data_sent);
    wait(100, SC_NS);
    master_port->BurstRead(100, 10, data_rcv);
    for (int i = 0; i < 10; i++) {
      if (data_sent[i] != data_rcv[i]) {
        std::cout << data_sent[i] << " " << data_rcv[i] << std::endl;
        std::cout << "data missmatch" << std::endl;
      }
    }
  }

    SC_HAS_PROCESS(test_bench);

    test_bench(sc_module_name nm) : sc_module(nm) { SC_THREAD(stimuli); }
  };

  class rtl_memory : public sc_module {
   public:
    sc_in_rv<16> address_p;
    sc_inout_rv<8> data_p;
    sc_in_resolved rw_p;
    sc_lv<8>* mem_arr;
    void run()  // sensitive rw_p
    {
      while (true) {
        // read cycle
        if (rw_p->read() == SC_LOGIC_1) {
          data_p->write(*(mem_arr + (sc_uint<16>(address_p->read()))));
          // write cycle
        } else if (rw_p->read() == SC_LOGIC_0) {
          *(mem_arr + (sc_uint<16>(address_p->read()))) = data_p->read();
        }
        wait();
      }
    }

    SC_HAS_PROCESS(rtl_memory);
    rtl_memory(sc_module_name nm, int mem_size = 100) : sc_module(nm) {
      mem_arr = new sc_lv<8>[mem_size];
      for (int i = 0; i < mem_size; i++) {
        mem_arr[i] = sc_lv<8>(0);
      }
      SC_THREAD(run);
      sensitive << rw_p;
    }

    ~rtl_memory() { delete[] mem_arr; }
  };

}  // namespace tflite_soc

  // Main program
  int sc_main(int argc, char* argv[]) {
    sc_set_time_resolution(1, SC_NS);

    sc_signal_rv<16> address_s;
    sc_signal_rv<8> data_s;
    sc_signal_resolved rw_s;

    tflite_soc::test_bench tb("tb");
    tflite_soc::dma_channel transactor("transactor");
    tflite_soc::rtl_memory uut("uut", 1000);

    tb.master_port(transactor);
    transactor.data_p(data_s);
    transactor.rw_p(rw_s);
    transactor.address_p(address_s);
    uut.address_p(address_s);
    uut.data_p(data_s);
    uut.rw_p(rw_s);

    sc_start();
    return 0;
  }
