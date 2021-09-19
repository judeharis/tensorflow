# Settings for PYNQ Z1.
ifeq ($(TARGET),pynq)
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
