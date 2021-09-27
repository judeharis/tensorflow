// Created by Nicolas Agostini
#ifndef TENSORFLOW_LITE_KERNELS_MODELING_COMM_CHANNELS_H_
#define TENSORFLOW_LITE_KERNELS_MODELING_COMM_CHANNELS_H_

#include <systemc/systemc.h>

namespace tflite_soc {

// Interface adapted from:
// https://sclive.wordpress.com/2008/01/11/systemc-tutorial-interfaces-and-channels-2/
class dma_interface : virtual public sc_interface {
 public:
  virtual void BurstWrite(int destAddress, int numBytes, sc_lv<8>* data) = 0;
  virtual void BurstRead(int sourceAddress, int numBytes, sc_lv<8>* data) = 0;
  virtual void StreamWrite(int destAddress, int numBytes, sc_lv<8>* data) = 0;
  virtual void StreamRead(int sourceAddress, int numBytes, sc_lv<8>* data) = 0;

  // The simulated DRAM module operates at 533MHz frequency
  // Which roughtly translates to 1/533MHz = 1.87ns of latency

  /// Value in nanoseconds
  const unsigned latency = 2;
  const unsigned debug = false;
};

class dma_channel: public dma_interface, public sc_channel {
public:
   sc_out_rv<16> address_p;
   sc_inout_rv<8> data_p;
   sc_out_resolved rw_p;
   dma_channel(sc_module_name nm): sc_channel(nm)
      ,address_p("address_p")
      ,data_p("data_p")
      ,rw_p("rw_p")
   { }
   virtual void BurstWrite( int destAddress, int numBytes, sc_lv<8> *data );
   virtual void BurstRead(int sourceAddress, int numBytes, sc_lv<8>* data);

   virtual void StreamWrite(int destAddress, int numBytes, sc_lv<8>* data);
   virtual void StreamRead(int sourceAddress, int numBytes, sc_lv<8>* data);
};


}  // namespace tflite_soc

#endif  // TENSORFLOW_LITE_KERNELS_MODELING_COMM_CHANNELS_H_