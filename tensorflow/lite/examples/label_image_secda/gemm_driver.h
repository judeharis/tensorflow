#ifndef ACCEL
#define ACCEL

#include <chrono>
#include <typeinfo>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <fcntl.h>
#include <fstream>

using namespace std;
using namespace std::chrono;

#define MM2S_CONTROL_REGISTER 0x00
#define MM2S_STATUS_REGISTER 0x04
#define MM2S_START_ADDRESS 0x18
#define MM2S_LENGTH 0x28

#define S2MM_CONTROL_REGISTER 0x30
#define S2MM_STATUS_REGISTER 0x34
#define S2MM_DESTINATION_ADDRESS 0x48
#define S2MM_LENGTH 0x58

#ifdef ACC_PROFILE
#define prf_start(N) auto start##N = chrono::steady_clock::now();
#define prf_end(N,X) auto end##N = chrono::steady_clock::now(); X += end##N-start##N;
#else
#define prf_start(N) 
#define prf_end(N,X)
#endif

#define acc_address 0x43C00000
#define page_size 65536

#define dma_buffer_len 4194304
#define dma_buffer_size 200000
#define dma_addr0 0x40400000
#define dma_addr1 0x40410000
#define dma_addr2 0x40420000
#define dma_addr3 0x40430000

#define dma_addr_in0 0x16000000
#define dma_addr_in1 0x18000000
#define dma_addr_in2 0x1a000000
#define dma_addr_in3 0x1c000000
#define dma_addr_out0 0x16400000
#define dma_addr_out1 0x18400000
#define dma_addr_out2 0x1a400000
#define dma_addr_out3 0x1c400000



// Basic DMA API
template <typename Integer>
void dma_set(unsigned int* dma_virtual_address, int offset, unsigned int value) {
    dma_virtual_address[offset>>2] = value;
}

template <typename Integer>
unsigned int dma_get(unsigned int* dma_virtual_address, int offset) {
    return dma_virtual_address[offset>>2];
}


template <typename Integer>
void dma_s2mm_status(unsigned int* dma_virtual_address) {
    msync(dma_virtual_address,88,MS_SYNC);
    unsigned int status = dma_get<int>(dma_virtual_address, S2MM_STATUS_REGISTER);
}

template <typename Integer>
void dma_mm2s_status(unsigned int* dma_virtual_address) {
    msync(dma_virtual_address,88,MS_SYNC);
    unsigned int status = dma_get<int>(dma_virtual_address, MM2S_STATUS_REGISTER);
}

template <typename Integer>
void dma_mm2s_sync(unsigned int* dma_virtual_address) {
    msync(dma_virtual_address,88,MS_SYNC);
    unsigned int mm2s_status =  dma_get<int>(dma_virtual_address, MM2S_STATUS_REGISTER);
    while(!(mm2s_status & 1<<12) || !(mm2s_status & 1<<1) ){
        dma_s2mm_status<int>(dma_virtual_address);
        dma_mm2s_status<int>(dma_virtual_address);
        mm2s_status =  dma_get<int>(dma_virtual_address, MM2S_STATUS_REGISTER);

    }
}

template <typename Integer>
void dma_s2mm_sync(unsigned int* dma_virtual_address) {
    msync(dma_virtual_address,88,MS_SYNC);
    unsigned int s2mm_status = dma_get<int>(dma_virtual_address, S2MM_STATUS_REGISTER);
    while(!(s2mm_status & 1<<12) || !(s2mm_status & 1<<1)){
        dma_s2mm_status<int>(dma_virtual_address);
        dma_mm2s_status<int>(dma_virtual_address);      
        s2mm_status = dma_get<int>(dma_virtual_address, S2MM_STATUS_REGISTER);
    }
}

template <typename Integer>
void dma_change_start(unsigned int* dma0, unsigned int* dma1, unsigned int* dma2, unsigned int* dma3,int offset) {
    dma_set<int>(dma0, MM2S_START_ADDRESS, dma_addr_in0+offset); // Write source address
    dma_set<int>(dma1, MM2S_START_ADDRESS, dma_addr_in1+offset); // Write source address
    dma_set<int>(dma2, MM2S_START_ADDRESS, dma_addr_in2+offset); // Write source address
    dma_set<int>(dma3, MM2S_START_ADDRESS, dma_addr_in3+offset); // Write source address
    // printf("");
    // msync(dma0,88,MS_SYNC);
    // msync(dma1,88,MS_SYNC);
    // msync(dma2,88,MS_SYNC);
    // msync(dma3,88,MS_SYNC);
}

template <typename Integer>
void dma_change_end(unsigned int* dma0, unsigned int* dma1, unsigned int* dma2, unsigned int* dma3,int offset) {
    dma_set<int>(dma0, S2MM_DESTINATION_ADDRESS, dma_addr_out0+offset); // Write destination address
    dma_set<int>(dma1, S2MM_DESTINATION_ADDRESS, dma_addr_out1+offset); // Write destination address
    dma_set<int>(dma2, S2MM_DESTINATION_ADDRESS, dma_addr_out2+offset); // Write destination address
    dma_set<int>(dma3, S2MM_DESTINATION_ADDRESS, dma_addr_out3+offset); // Write destination address
    // printf("");
    // msync(dma0,88,MS_SYNC);
    // msync(dma1,88,MS_SYNC);
    // msync(dma2,88,MS_SYNC);
    // msync(dma3,88,MS_SYNC);
}

template <typename Integer>
void initDMA(unsigned int* dma, int src, int dst){
    dma_set<int>(dma, S2MM_CONTROL_REGISTER, 4);
    dma_set<int>(dma, MM2S_CONTROL_REGISTER, 4);
    dma_set<int>(dma, S2MM_CONTROL_REGISTER, 0);
    dma_set<int>(dma, MM2S_CONTROL_REGISTER, 0);
    dma_set<int>(dma, S2MM_DESTINATION_ADDRESS, dst);
    dma_set<int>(dma, MM2S_START_ADDRESS, src);
    dma_set<int>(dma, S2MM_CONTROL_REGISTER, 0xf001);
    dma_set<int>(dma, MM2S_CONTROL_REGISTER, 0xf001);
}
// End of DMA API

template <typename Integer>
int* getArray(size_t base_addr,size_t length ){
    fstream myfile;
    size_t virt_base = base_addr & ~(getpagesize()- 1);
    size_t virt_offset = base_addr - virt_base;
    int fd = open ("/dev/mem", O_RDWR | O_SYNC);
    void *addr =mmap(NULL,length+virt_offset,PROT_READ | PROT_WRITE,MAP_SHARED,fd,virt_base);
    close(fd);
    if (addr == (void*) -1 ) exit (EXIT_FAILURE);
    int* array =reinterpret_cast<int*> (addr);
    return array;
}

template <typename Integer>
void preload_weights(uint8_t* weight_data, int* dims,
    vector<uint8_t> &wb0,vector<uint8_t> &wb1,
    vector<uint8_t> &wb2,vector<uint8_t> &wb3,
    vector<int> &wb_dex,vector<int> &wt_sum1,
    vector<int> &wt_sum2,vector<int> &wt_sum3,
    vector<int> &wt_sum4, vector<int> &dt_sum_dex,
    int &w_c,int &sums_curr){

    int width = dims[0];
    int w = ((width + 4 - 1) - ((width + 4 - 1) % 4));
    int depth = dims[1]*dims[2]*dims[3];
    int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
    int max = width*depth;
    for(int i=0; i<w/4;i++){
    int  s0 =0;
    int  s1 =0;
    int  s2 =0;
    int  s3 =0;

    for(int j=0; j<d;j++){
    if (j<depth){
        unsigned char w0 = (i*(depth*4)+j>=max)?0:weight_data[i*(depth*4)+j];
        unsigned char w1 = (i*(depth*4)+j+ depth*1>=max)?0:weight_data[i*(depth*4)+j+ depth*1];
        unsigned char w2 = (i*(depth*4)+j+ depth*2>=max)?0:weight_data[i*(depth*4)+j+ depth*2];
        unsigned char w3 = (i*(depth*4)+j+ depth*3>=max)?0:weight_data[i*(depth*4)+j+ depth*3];
        unsigned char weights[] = {w3,w2,w1,w0};
        s0+=w0;
        s1+=w1;
        s2+=w2;
        s3+=w3;
        wb0.push_back(w0);
        wb1.push_back(w1);
        wb2.push_back(w2);
        wb3.push_back(w3);
        w_c++;
    }else{
        wb0.push_back(0);
        wb1.push_back(0);
        wb2.push_back(0);
        wb3.push_back(0);
        w_c++;
    }
    }
    wt_sum1.push_back(s0);
    wt_sum2.push_back(s1);
    wt_sum3.push_back(s2);
    wt_sum4.push_back(s3);
    sums_curr++;
    }  
    wb_dex.push_back(w_c);
    dt_sum_dex.push_back(sums_curr);
}

struct dma_in_buffer{ 
    int offset=0;
    int inl0=0;
    int inl1=0;
    int inl2=0;
    int inl3=0;
    bool in_use=false;
    int id=-1;
};

//Used for tracking output locations
struct store_params{ 
    int* dst;
    int dcs; 
    int rows; 
    int cols; 
    int rcols;
    int rrows; 
};

//Used for profiling
struct gemm_profilier{ 
    std::chrono::duration<long long int, std::ratio<1, 1000000000>> acctime= std::chrono::duration<long long int, std::ratio<1, 1000000000>>(0);
    std::chrono::duration<long long int, std::ratio<1, 1000000000>> convtime = std::chrono::duration<long long int, std::ratio<1, 1000000000>>(0);
    int layer=0;
    int layer_weight_tile = 0;
    int layer_input_tile = 0;
    int layer_print = -1;
    int layer_ww = -1;
    int layer_iw = -1;
    bool profile = false;
};


struct gemm_driver { 
    //Accelerator "Software" variables
    int* acc;
    bool on;

    //DMA MM Pointers
    unsigned int* dma0;
    unsigned int* dma1;
    unsigned int* dma2;
    unsigned int* dma3;

    //DMA Input MMapped Buffers
    unsigned int* in0;
    unsigned int* in1;
    unsigned int* in2;
    unsigned int* in3;

    // //DMA Output MMapped Buffers
    int* out0;
    int* out1;
    int* out2;
    int* out3;

    //Temporary Weight non-MMapped Padded Buffers
    unsigned int* wb_0;
    unsigned int* wb_1;
    unsigned int* wb_2;
    unsigned int* wb_3;
    std::vector<int> wb_dex;

    //Temporary Input non-MMapped Padded Buffers
    unsigned int* inb_0;
    unsigned int* inb_1;
    unsigned int* inb_2;
    unsigned int* inb_3;
    int in_id=0;

    //Driver variables
    struct store_params* st_params; 
    int w_c=0;
    bool lhs_start=false;

    // Output Pipeline Metadata
    std::vector<int> wt_sum1;
    std::vector<int> wt_sum2;
    std::vector<int> wt_sum3;
    std::vector<int> wt_sum4;
    std::vector<int> dt_sum_dex;

    int* in_sum1;
    int* in_sum2;
    int* in_sum3;
    int* in_sum4;
    int in_sum_len=0;

    std::vector<int> bias;
    int rf=0;
    int ra=0;
    int re=0;
    int rhs_offset=0;
    int lhs_offset=0;


    //Pipeline vars
    struct dma_in_buffer* dinb; 
    int bufflen;
    int dID=0;
    int sID=0;
    int rID=0;

    //Threading variables
    pthread_t  mtid;
    int t2_row_jump=0;

    //Profiling varaiable
    struct gemm_profilier t;

    gemm_driver(
        int* _acc, bool _on,
        unsigned int* _dma0, unsigned int* _dma1,unsigned int* _dma2,unsigned int* _dma3,
        unsigned int* _in0, unsigned int* _in1, unsigned int* _in2,unsigned int* _in3,
        int* _out0,int* _out1,int* _out2,int* _out3,
        unsigned int* _wb_0, unsigned int* _wb_1, unsigned int* _wb_2, unsigned int* _wb_3, 
        std::vector<int> _wb_dex,store_params* _st_params, 
        std::vector<int> _wt_sum1, std::vector<int> _wt_sum2, std::vector<int> _wt_sum3, std::vector<int> _wt_sum4,
        std::vector<int> _dt_sum_dex,dma_in_buffer* _dinb, int _bufflen, pthread_t _mtid
    ){
        acc = _acc;
        on = _on;
        dma0 = _dma0;
        dma1 = _dma1;
        dma2 = _dma2;
        dma3 = _dma3;
        in0 = _in0;
        in1 = _in1;
        in2 = _in2;
        in3 = _in3;
        out0 = _out0;
        out1 = _out1;
        out2 = _out2;
        out3 = _out3;
        wb_0 = _wb_0;
        wb_1 = _wb_1;
        wb_2 = _wb_2;
        wb_3 = _wb_3;
        wb_dex = _wb_dex;
        st_params = _st_params;
        wt_sum1 = _wt_sum1;
        wt_sum2 = _wt_sum2;
        wt_sum3 = _wt_sum3;
        wt_sum4 = _wt_sum4;
        dt_sum_dex = _dt_sum_dex;
        dinb = _dinb;
        bufflen = _bufflen;
        mtid = _mtid;
    }

};

#endif
