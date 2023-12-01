#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2

struct FileSystem {
  uchar *volume;
  int SUPERBLOCK_SIZE;
  int FCB_SIZE;
  int FCB_ENTRIES;
  int STORAGE_SIZE;
  int STORAGE_BLOCK_SIZE;
  int MAX_FILENAME_SIZE;
  int MAX_FILE_NUM;
  int MAX_FILE_SIZE;
  int FILE_BASE_ADDRESS;
};

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
                        int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
                        int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
                        int MAX_FILE_NUM, int MAX_FILE_SIZE,
                        int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);

struct FCB {
  static const uint32_t filename_offset = 0;
  static const uint32_t start_block_offset = 20;
  static const uint32_t file_size_offset = 22;
  static const uint32_t creation_time_offset = 24;
  static const uint32_t modified_time_offset = 26;

  uchar *data;

  uint32_t index;

  FileSystem *fs;

  __device__ FCB() : data(NULL), index(-1) {}
  __device__ FCB(FileSystem *fs, u32 index) : index(index), fs(fs) {
    if (index >= fs->FCB_ENTRIES) {
      this->data = NULL;
      return;
    }
    this->data = fs->volume + fs->SUPERBLOCK_SIZE + index * fs->FCB_SIZE;
    set_start_block_index(index * (fs->MAX_FILE_SIZE / fs->MAX_FILE_NUM /
                                   fs->STORAGE_BLOCK_SIZE));
  }

  __device__ uint8_t is_null() { return data == NULL ? 1 : 0; }

  __device__ const char *get_filename() const {
    return (char *)&(data[filename_offset]);
  }

  __device__ void set_filename(const char *name) {
    uchar *name_ptr = data + filename_offset;
    for (int i = 0; i < 19; i++) {
      name_ptr[i] = name[i];
      if (name[i] == '\0')
        break;
    }
    name_ptr[19] = '\0';
  }

  __device__ void clean_filename() {
    (data + filename_offset)[0] = '\0';
  }

  __device__ uint8_t is_unused() {
    if (*(data + filename_offset) == '\0') {
      return 1;
    }
    return 0;
  }

  __device__ uint16_t get_start_block_index() const {
    return *(uint16_t *)(data + start_block_offset);
  }

  __device__ uint16_t set_start_block_index(uint16_t new_index) {
    uint16_t *ptr = (uint16_t *)(data + start_block_offset);
    *ptr = new_index;
  }

  __device__ uint16_t get_ending_block_index() const {
    uint16_t size = get_file_size();
    uint16_t carry = !!(size % fs->STORAGE_BLOCK_SIZE);
    uint16_t starting_index = get_start_block_index();
    uint16_t ending_index =
        starting_index + size / fs->STORAGE_BLOCK_SIZE + carry;
    return ending_index;
  }

  __device__ uchar *get_start_content_ptr() const {
    return fs->volume + fs->FILE_BASE_ADDRESS + index * 32;
  }

  __device__ uint16_t get_file_size() const {
    return *(uint16_t *)(data + file_size_offset);
  }

  __device__ uint16_t set_file_size(uint16_t new_size) {
    uint16_t *ptr = (uint16_t *)(data + file_size_offset);
    *ptr = new_size;
  }

  __device__ void set_creation_timestamp(uint16_t timestamp) {
    uint16_t *p = (uint16_t *)(data + creation_time_offset);
    *p = timestamp;
  }

    __device__ uint16_t get_creation_timestamp() const {
      return *(uint16_t *)(data + creation_time_offset);
    }

    __device__ void set_modified_timestamp(uint16_t timestamp) {
      uint16_t *p = (uint16_t *)(data + modified_time_offset);
      *p = timestamp;
    }

  __device__ uint16_t get_modified_timestamp() const {
    return *(uint16_t *)(data + modified_time_offset);
  }
};

struct SuperBlock {
  uchar *data;
  FileSystem *fs;

  __device__ SuperBlock(FileSystem *fs) : fs(fs) { data = fs->volume; }

  __device__ void set_free(u32 index) {
    if (index >= fs->MAX_FILE_SIZE / 32) {
      return;
    }
    data[index / 8] = data[index / 8] & (~(0x80 >> (index % 8)));
  }

  __device__ void set_used(u32 index) {
    if (index >= fs->MAX_FILE_SIZE / 32) {
      return;
    }
    data[index / 8] = data[index / 8] | (0x80 >> (index % 8));
  }
};

#endif
