#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdio.h>

typedef unsigned char uchar;
typedef uint32_t u32;

#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2
#define CD 3
#define CD_P 4
#define RM_RF 5
#define PWD 6
#define MKDIR 7

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

__device__ static uint8_t mystrcmp(const char* a, const char* b)
{
  for (u32 i = 0; a[i] == b[i]; i++) {
    if (a[i] == '\0') {
      return 1;
    }
  }
  return 0;
}

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

struct FCB {
  static const uint32_t filename_offset = 0;
  static const uint32_t start_block_offset = 20;
  static const uint32_t file_size_offset = 22;
  static const uint32_t creation_time_offset = 24;
  static const uint32_t modified_time_offset = 26;

  static const uint32_t dir_struct_offset = 28;

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

  __device__ FCB(FileSystem *fs, u32 index, uint16_t gtime, const char* name) : FCB(fs, index) {
    set_creation_timestamp(gtime);
    set_modified_timestamp(gtime);
    set_filename(name);
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

  __device__ uint16_t get_parent_index() const {
    return (uint16_t)((*(uint32_t*)(data + dir_struct_offset) & 0x7FE00000) >> 21);
  }

  __device__ FCB get_parent() const {
    return FCB(fs, get_parent_index());
  }

  __device__ uint16_t set_parent_index(uint16_t new_index) {
    uint32_t* ptr = (uint32_t*)(data + dir_struct_offset);
    *ptr = (*ptr & ~0x7FE00000) | ((new_index & 0x3FF) << 21) ;
  }

  __device__ uint16_t get_sibiling_index() const {
    return (uint16_t)((*(uint32_t*)(data + dir_struct_offset) & 0x1FF800) >> 11);
  }

  __device__ FCB get_next_sibiling() const {
    return FCB(fs, get_sibiling_index());
  }

  __device__ uint16_t set_sibiling_index(uint16_t new_index) {
    uint32_t tmp = new_index;
    uint32_t* ptr = (uint32_t*)(data + dir_struct_offset);
    *ptr = (*ptr & ~0x1FF800) | ((tmp & 0x3FF) << 11);
  }

  __device__ uint16_t get_sub_file_index() const {
    return (uint16_t)((*(uint32_t*)(data + dir_struct_offset) & 0x7FE) >> 1);
  }

  __device__ FCB get_next_sub_file() const {
    return FCB(fs, get_sub_file_index());
  }

  __device__ void set_sub_file_index(uint16_t new_index) const {
    uint32_t tmp = new_index;
    uint32_t* ptr = (uint32_t *)(data + dir_struct_offset);
    *ptr = (*ptr & ~0x7FE) | ((tmp & 0x3FF) << 1);
  }

  __device__ uint8_t is_directory() const {
    return (*(uint32_t*)(data + dir_struct_offset) & 0x80000000) >> 31;
  }

  __device__ uint8_t is_file() const {
    return !is_directory();
  }

  __device__ void set_to_directory() {
    uint32_t* ptr = (uint32_t*)(data + dir_struct_offset);
    *ptr = *ptr | 0x80000000;
  }

  __device__ void set_to_file() {
    uint32_t* ptr = (uint32_t*)(data + dir_struct_offset);
    *ptr = *ptr & 0x7FFFFFFF;
  }

  __device__ uint32_t get_name_len() const {
    const char* name = get_filename();
    uint32_t len = 0;
    for (int i = 0; i < 20; i++) {
      if (name[i] != '\0') {
        len++;
      } else {
        return len;
      }
    }
  }

  __device__ void add_file_to_directory(FCB& new_file) {
    if (is_directory()) {
      uint16_t prev_head = get_sub_file_index();
      set_sub_file_index(new_file.index);

      new_file.set_sibiling_index(prev_head);
      new_file.set_parent_index(index);

      set_file_size(get_file_size() + new_file.get_name_len());
    } else {
      printf("not a dir! %s\n", get_filename());
    }
  }

  __device__ void remove_file_from_directory(FCB& file) {
    if (is_directory() && index == file.get_parent_index()) {
      SuperBlock sb(fs);

      FCB prev;
      FCB current = get_next_sub_file();
  
      set_file_size(get_file_size() - file.get_name_len());
      while(current.index != 0 && current.index != file.index) {
        prev = current;
        current = current.get_next_sibiling();
      }

      for (u32 i = file.get_start_block_index();
           i < file.get_ending_block_index(); i++) {
        sb.set_free(i);
      }

      file.set_file_size(0);
      file.clean_filename();

      if (prev.is_null()) {
        set_sub_file_index(file.get_sibiling_index());
      } else {
        prev.set_sibiling_index(file.get_sibiling_index());
      }
    } else {
      printf("not a subfile!\n");
    }
  }

  __device__ void remove_directory() {
    if (is_directory()) {
      SuperBlock sb(fs);
      FCB current = get_next_sub_file();
      while (current.index != 0) {
        if (current.is_directory()) {
          current.remove_directory();
        } else {
          for (u32 i = current.get_start_block_index();
            i < current.get_ending_block_index(); i++) {
            sb.set_free(i);
          }

          current.set_file_size(0);
          current.clean_filename();
        }
        current = current.get_next_sibiling();
      }

      get_parent().remove_file_from_directory(*this);
    }
  }

  __device__ uint8_t is_root_dir() const {
    return mystrcmp(get_filename(), "root");
  }

  __device__ uint16_t get_current_dir() const {
    if (is_root_dir()) {
      return *(uint16_t*)(data + filename_offset + 18);
    } else {
      printf("not root\n");
      return 0xFFFF;
    }
  }

  __device__ void set_current_dir(uint16_t new_index) {
    if (is_root_dir()) {
      uint16_t* ptr = (uint16_t*)(data + filename_offset + 18);
      *ptr = new_index;
    }
  }

};

#endif
