#ifndef SYSC_TYPES_H
#define SYSC_TYPES_H

#include <systemc.h>

typedef struct _DATA {
  sc_uint<32> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _DATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
} DATA;

typedef struct _SDATA {
  sc_int<32> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _SDATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
} SDATA;

template <int W>
struct _FDATA {
  sc_uint<W> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _FDATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
};

template <int W>
using FDATA = _FDATA<W>;

#endif