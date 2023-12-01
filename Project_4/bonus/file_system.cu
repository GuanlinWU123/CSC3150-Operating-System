#include "file_system.h"
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;

__device__ void make_root_directory(FileSystem* fs) {
  FCB root(fs, 0, gtime++, "root");
  root.set_to_directory();
  root.set_current_dir(0);
}

__device__ FCB get_cd(FileSystem* fs) {
  FCB root(fs, 0);
  return FCB(fs, root.get_current_dir());
}

__device__ void set_cd(FileSystem* fs, uint16_t index) {
  FCB root(fs, 0);
  root.set_current_dir(index);
}

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
                        int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
                        int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
                        int MAX_FILE_NUM, int MAX_FILE_SIZE,
                        int FILE_BASE_ADDRESS) {
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
  make_root_directory(fs);
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op) {
  /* Implement open operation here */
  FCB first_unused_entry;

  for (u32 i = 0; i < fs->MAX_FILE_NUM; i++) {

    FCB current(fs, i);

    if (current.is_unused() && first_unused_entry.is_null()) {
      first_unused_entry = current;
    } else if (mystrcmp(s, current.get_filename()) && current.get_parent_index() == get_cd(fs).index) {
      current.set_modified_timestamp(gtime++);
      return i;
    }
  }

  if (first_unused_entry.is_null()) {
    printf("storage is full.\n");
    return 0xFFFFFFFF;
  }
  first_unused_entry.set_filename(s);
  first_unused_entry.set_file_size(0);
  uint16_t current_time = gtime++;
  first_unused_entry.set_creation_timestamp(current_time);
  first_unused_entry.set_modified_timestamp(current_time);

  get_cd(fs).add_file_to_directory(first_unused_entry);
  return first_unused_entry.index;
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp) {
  /* Implement read operation here */
  if (fp > fs->MAX_FILE_NUM) {
    return;
  }
  FCB file_to_read(fs, fp);
  size = (file_to_read.get_file_size() < size) ? file_to_read.get_file_size() : size;

  uchar *start = file_to_read.get_start_content_ptr();
  for (u32 i = 0; i < size; i++) {
    output[i] = start[i];
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp) {
  /* Implement write operation here */
  if (fp > fs->MAX_FILE_NUM) {
    return 0xFFFFFFFF;
  }

  FCB file(fs, fp);
  SuperBlock sb(fs);
  if (size > fs->MAX_FILE_SIZE / fs->MAX_FILE_NUM) {
    return 0xFFFFFFFF;
  }

  FCB parent_dir = file.get_parent();
  file.set_file_size(size);

  uint16_t current_time = gtime++;
  file.set_modified_timestamp(current_time);

  for (u32 i = file.get_start_block_index(); i < file.get_ending_block_index();
       i++) {
    sb.set_used(i);
  }

  uchar *start = file.get_start_content_ptr();
  for (u32 i = 0; i < size; i++) {
    start[i] = input[i];
  }
  return 0;
}

__device__ void sorting(FCB files_to_sort[], u32 size, int op)
{
  u32 greatest = 0;
  for (u32 i = 0; i < size; i++) {
    greatest = i;
    for (u32 j = i + 1; j < size; j++) {
      FCB current = files_to_sort[greatest];
      FCB next = files_to_sort[j];
      if (op == LS_D) {
        if (next.get_modified_timestamp() > current.get_modified_timestamp()) {
          greatest = j;
        }
      } else if (op == LS_S) {
        if (next.get_file_size() > current.get_file_size()) {
          greatest = j;
        } else if (next.get_file_size() == current.get_file_size() && next.get_creation_timestamp() < current.get_creation_timestamp()) {
          greatest = j;
        }
      }
    }

    if (greatest != i) {
      FCB tmp = files_to_sort[i];
      files_to_sort[i] = files_to_sort[greatest];
      files_to_sort[greatest] = tmp;
    }
  }
}

__device__ FCB find_fcb(FileSystem *fs, const char* name) {
  FCB current = get_cd(fs).get_next_sub_file();
  while (current.index != 0) {
    if (mystrcmp(current.get_filename(), name)) {
        return current;
      }
      current = current.get_next_sibiling();
  }
  return FCB();
}

__device__ void print_pwd(FCB current) {
  if (!current.is_root_dir()) {
    print_pwd(current.get_parent());
    printf("%s/", current.get_filename());
  } else {
    printf("/");
  }
}

__device__ void fs_gsys(FileSystem *fs, int op) {
  /* Implement LS_D and LS_S operation here */
  if (op == CD_P) {
    set_cd(fs, get_cd(fs).get_parent_index());
    return;
  } else if (op == PWD) {
    FCB cd = get_cd(fs);
    print_pwd(cd);
    printf("\n");
    return;
  } else {
    uint32_t size = 0;
    FCB* files_to_sort = NULL;
    FCB cd = get_cd(fs);
    cudaMalloc(&files_to_sort, fs->MAX_FILE_NUM*sizeof(FCB));
    FCB current = cd.get_next_sub_file();

    while (current.index != 0 && !current.is_unused()) {
      files_to_sort[size++] = current;
      current = current.get_next_sibiling();
    }
    sorting(files_to_sort, size, op);
    if (op == LS_D) {
      printf("=== sort by modification time ===\n");
      for (uint32_t i = 0; i < size; i++) {
        FCB& c = files_to_sort[i];
        printf("%s %c\n", c.get_filename(), c.is_directory() ? 'd' : ' ');
      }
    } else if (op == LS_S) {
      printf("=== sort by size ===\n");
      for (uint32_t i = 0; i < size; i++) {
        FCB& c = files_to_sort[i];
        printf("%s %d %c\n", c.get_filename(), c.get_file_size(), c.is_directory() ? 'd' : ' ');
      }
    }
    return;
  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s) {
  /* Implement rm operation here */
  SuperBlock sb(fs);
  FCB current_dir = get_cd(fs);

  if (op == RM) {
    FCB file = find_fcb(fs, s);
    if (file.is_file()) {
      for (u32 i = file.get_start_block_index(); i < file.get_ending_block_index(); i++) {
        sb.set_free(i);
      }
      file.set_file_size(0);
      file.clean_filename();
    }
    return;
  } else if (op == MKDIR) {
    for (u32 i = 0; i < fs->MAX_FILE_NUM; i++) {
      FCB current(fs, i);
      if (current.is_unused()) {
        current = FCB(fs, i, gtime++, s);
        current.set_to_directory();
        current_dir.add_file_to_directory(current);
        break;
      }
    }
    return;
  } else if (op == CD) {
    FCB dir = find_fcb(fs, s);
    if (!dir.is_null()) {
      set_cd(fs, dir.index);
    } else {
      printf("dir not found!\n");
    }
    return;
  } else if (op == RM_RF) {
    FCB dir = find_fcb(fs, s);
    dir.remove_directory();
    return;
  }
}
