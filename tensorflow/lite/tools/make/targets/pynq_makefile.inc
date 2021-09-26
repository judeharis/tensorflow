# Settings for PYNQ Z1.
ifeq ($(TARGET),pynq_sa)
  TARGET_ARCH := armv7l
  TARGET_TOOLCHAIN_PREFIX := arm-linux-gnueabihf-

  ifeq ($(TARGET_ARCH), armv7l)
    CXXFLAGS += \
      -march=armv7-a \
      -mfpu=neon \
      -funsafe-math-optimizations \
      -ftree-vectorize \
      -fPIC \
      -DACC_PROFILE \
      -DACC_NEON

    CFLAGS += \
      -march=armv7-a \
      -mfpu=neon \
      -funsafe-math-optimizations \
      -ftree-vectorize \
      -fPIC

    LDFLAGS := \

  endif

  LIBS := \
    -lstdc++ \
    -lpthread \
    -lm \
    -ldl \
    -lrt

endif


ifeq ($(TARGET),pynq_vm)
  TARGET_ARCH := armv7l
  TARGET_TOOLCHAIN_PREFIX := arm-linux-gnueabihf-

  ifeq ($(TARGET_ARCH), armv7l)
    CXXFLAGS += \
      -march=armv7-a \
      -mfpu=neon \
      -funsafe-math-optimizations \
      -ftree-vectorize \
      -fPIC \
      -DACC_PROFILE \
      -DACC_NEON \
      -DVM_ACC

    CFLAGS += \
      -march=armv7-a \
      -mfpu=neon \
      -funsafe-math-optimizations \
      -ftree-vectorize \
      -fPIC

    LDFLAGS := \

  endif

  LIBS := \
    -lstdc++ \
    -lpthread \
    -lm \
    -ldl \
    -lrt

endif