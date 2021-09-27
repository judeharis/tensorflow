// Created by Nicolas Agostini
#include "tensorflow/lite/kernels/modeling/comm_channels.sc.h"

namespace tflite_soc {


// Implementing methods of dma_channel
void dma_channel::BurstWrite(int destAddress, int numBytes, sc_lv<8>* data) {
  sc_lv<8>* ite = data;
  for (int i = 0; i < numBytes; i++) {
    address_p->write(destAddress++);
    data_p->write(*(ite++));
    wait(10, SC_NS);
    if (debug) std::cout << "Write out " << data_p->read() << std::endl;
    rw_p->write(SC_LOGIC_0);  // Write pulse
    wait(50, SC_NS);
    rw_p->write(SC_LOGIC_Z);
    address_p->write("ZZZZZZZZZZZZZZZZ");
    data_p->write("ZZZZZZZZ");
    wait(10, SC_NS);
  }
}

void dma_channel::BurstRead(int sourceAddress, int numBytes, sc_lv<8>* data) {
   for (int i=0; i<numBytes; i++) {
      address_p->write(sourceAddress++);
      wait(10, SC_NS);
      rw_p->write(SC_LOGIC_1); // Read pulse
      wait(10, SC_NS);
      *(data++) = data_p->read();
      if (debug) std::cout << "Data read " << data_p->read() << std::endl;
      wait(40, SC_NS);
      rw_p->write(SC_LOGIC_Z);
      address_p->write("ZZZZZZZZZZZZZZZZ");
      data_p->write("ZZZZZZZZ");
      wait(10, SC_NS);
   }
}

void dma_channel::StreamWrite(int destAddress, int numBytes, sc_lv<8>* data) {
  sc_lv<8>* ite = data;
  for (int i = 0; i < numBytes; i++) {
    address_p->write(destAddress++);
    data_p->write(*(ite++));
    //wait(10, SC_NS);
    if (debug) std::cout << "Write out " << data_p->read() << std::endl;
    rw_p->write(SC_LOGIC_0);  // Write pulse
    wait(latency/2, SC_NS);
    rw_p->write(SC_LOGIC_Z);
    address_p->write("ZZZZZZZZZZZZZZZZ");
    data_p->write("ZZZZZZZZ");
    wait(latency/2, SC_NS);
  }
}

void dma_channel::StreamRead(int sourceAddress, int numBytes, sc_lv<8>* data) {
  for (int i = 0; i < numBytes; i++) {
    address_p->write(sourceAddress++);
    //wait(10, SC_NS);
    rw_p->write(SC_LOGIC_1);  // Read pulse
    //wait(10, SC_NS);
    *(data++) = data_p->read();
    if (debug) std::cout << "Data read " << data_p->read() << std::endl;
    wait(latency/2, SC_NS);
    rw_p->write(SC_LOGIC_Z);
    address_p->write("ZZZZZZZZZZZZZZZZ");
    data_p->write("ZZZZZZZZ");
    wait(latency/2, SC_NS);
  }
}

}  // namespace tflite_soc