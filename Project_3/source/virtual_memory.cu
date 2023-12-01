#include "virtual_memory.h"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/types.h>

#define INVALID_BITS(var) decltype(var)(-1)

#define INVALID_U32 0xFFFFFFFF
#define INVALID_U16 0xFFFF

const uint32_t invalid_bit_offset = 31;
const uint32_t invalid_bit_mask = 0x1 << invalid_bit_offset;

const uint32_t virtual_page_num_offset = 18;
const uint32_t virtual_page_num_mask = 0x1FFF << virtual_page_num_offset;

const uint32_t lru_chain_next_offset = 8;
const uint32_t lru_chain_next_mask = 0x3FF << lru_chain_next_offset;

const uint32_t _placeholder_offset = 0;
const uint32_t _placeholder_mask = 0xFF << _placeholder_offset;

class memory_page_entry {
public:
  uint32_t invalid_bit;
  uint32_t virtual_page_num;
  uint32_t lru_chain_next;
  uint32_t _placeholder;

  uint32_t page_table_index;

  __device__ memory_page_entry(u32 entry, uint32_t index) {
    invalid_bit = (entry & invalid_bit_mask) >> invalid_bit_offset;
    virtual_page_num =
        (entry & virtual_page_num_mask) >> virtual_page_num_offset;
    lru_chain_next = (entry & lru_chain_next_mask) >> lru_chain_next_offset;
    _placeholder = (entry & _placeholder_mask) >> _placeholder_offset;

    page_table_index = index;
  }

  __device__ u32 to_u32() {
    u32 out = 0;
    out |= ((invalid_bit << invalid_bit_offset) & invalid_bit_mask);
    out |=
        ((virtual_page_num << virtual_page_num_offset) & virtual_page_num_mask);
    out |= ((lru_chain_next << lru_chain_next_offset) & lru_chain_next_mask);
    out |= ((_placeholder << _placeholder_offset) & _placeholder_mask);
    return out;
  }

  __device__ void update_virtual_page_num(VirtualMemory *vm,
                                          uint32_t new_page_num) {
    virtual_page_num = new_page_num;
    update_to_table(vm);
  }

  __device__ void update_to_table(VirtualMemory *vm) {
    if (page_table_index < vm->PAGE_ENTRIES) {
      vm->invert_page_table[page_table_index] = this->to_u32();
    } else {
      printf("[ERROR]: index out of range: %d, in %s\n", page_table_index,
             __func__);
    }
  }

};

__device__ memory_page_entry get_memory_page_entry(const VirtualMemory *vm,
                                                   uint32_t index) {
  if (index < vm->PAGE_ENTRIES) {
    return memory_page_entry(vm->invert_page_table[index], index);
  } else {
    printf("[ERROR]: index out of range: %d, in %s\n", index, __func__);
    return memory_page_entry(INVALID_U32, INVALID_U32);
  }
}

__device__ uint16_t get_storage_page_entry(const VirtualMemory *vm,
                                           uint32_t index) {
  if (index < (vm->STORAGE_SIZE / vm->PAGESIZE)) {
    uint16_t *start = (uint16_t *)(vm->invert_page_table + vm->PAGE_ENTRIES);
    return start[index];
  } else {
    printf("[ERROR]: index out of range: %d, in %s\n", index, __func__);
    return uint16_t(-1);
  }
}

__device__ void set_storage_page_entry(VirtualMemory *vm, uint32_t index,
                                       uint16_t page_num) {
  if (index < (vm->STORAGE_SIZE / vm->PAGESIZE)) {
    uint16_t *start = (uint16_t *)(vm->invert_page_table + vm->PAGE_ENTRIES);
    start[index] = page_num;
  } else {
    printf("[ERROR]: index out of range: %d, in %s\n", index, __func__);
  }
}

__device__ uint32_t get_lru_index(const VirtualMemory *vm) {
  u32 entry = vm->invert_page_table[vm->INVERT_PAGE_TABLE_SIZE / 4 - 1];
  return entry & 0x3FF;
}

__device__ uint32_t get_mru_index(const VirtualMemory *vm) {
  u32 entry = vm->invert_page_table[vm->INVERT_PAGE_TABLE_SIZE / 4 - 1];
  return (entry >> 10) & 0x3FF;
}

__device__ void set_lru_index(VirtualMemory *vm, uint32_t new_index) {
  if (new_index < vm->PAGE_ENTRIES) {
    vm->invert_page_table[vm->INVERT_PAGE_TABLE_SIZE / 4 - 1] &= (~0x3FF);
    vm->invert_page_table[vm->INVERT_PAGE_TABLE_SIZE / 4 - 1] |=
        (new_index & 0x3FF);
  } else {
    printf("[ERROR]: index out of range: %d, in %s\n", new_index, __func__);
  }
}

__device__ void set_mru_index(VirtualMemory *vm, uint32_t new_index) {
  if (new_index < vm->PAGE_ENTRIES) {
    vm->invert_page_table[vm->INVERT_PAGE_TABLE_SIZE / 4 - 1] &=
        (~((0x3FF) << 10));
    vm->invert_page_table[vm->INVERT_PAGE_TABLE_SIZE / 4 - 1] |=
        ((new_index & 0x3FF) << 10);
  } else {
    printf("[ERROR]: index out of range: %d, in %s\n", new_index, __func__);
  }
}

__device__ void move_index_to_mru(VirtualMemory *vm, uint32_t index,
                                  uint32_t prev_index) {
  uint32_t mru_index = get_mru_index(vm);
  if (index == mru_index) {
    return;
  }

  memory_page_entry prev_entry = get_memory_page_entry(vm, prev_index);
  memory_page_entry entry = get_memory_page_entry(vm, index);

  prev_entry.lru_chain_next = entry.lru_chain_next;
  prev_entry.update_to_table(vm);

  entry.lru_chain_next = mru_index;
  entry.update_to_table(vm);

  // set_memory_page_entry(vm, prev_index, prev_entry);
  // set_memory_page_entry(vm, index, entry);

  set_mru_index(vm, index);
  if (get_lru_index(vm) == index) {
    set_lru_index(vm, prev_index);
  }
}

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (uint32_t i = 0; i < vm->PAGE_ENTRIES; i++) {
    memory_page_entry entry = memory_page_entry(0, i);
    entry.invalid_bit = 1;
    entry.lru_chain_next = (i + 1);
    // vm->invert_page_table[i] = entry.to_u32();
    entry.update_to_table(vm);
  }

  for (uint32_t i = 0; i < vm->STORAGE_SIZE / vm->PAGESIZE; i++) {
    uint16_t *start = (uint16_t *)(vm->invert_page_table + vm->PAGE_ENTRIES);
    start[i] = uint16_t(-1);
  }

  set_lru_index(vm, vm->PAGE_ENTRIES - 1);
  set_mru_index(vm, 0);
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uint32_t is_invalid_storage_page_num(uint32_t page_num) {
  return (page_num & 0xFFFF) == 0xFFFF ? 1 : 0;
}

__device__ uint32_t find_page_in_storage(VirtualMemory *vm, uint32_t page_num) {
  const uint32_t storage_index_max = vm->STORAGE_SIZE / vm->PAGESIZE;

  uint32_t matched_storage_index = INVALID_U32;
  uint32_t storage_index_first_free = INVALID_U32;

  for (uint32_t storage_index = 0; storage_index < storage_index_max;
       storage_index++) {
    uint16_t current_page_num = get_storage_page_entry(vm, storage_index);

    if (current_page_num == page_num) {
      matched_storage_index = storage_index;
      break;
    } else if (is_invalid_storage_page_num(current_page_num) &&
               storage_index_first_free ==
                   INVALID_U32) {
      storage_index_first_free = storage_index;
    }
  }

  if (matched_storage_index != INVALID_U32) {
    return matched_storage_index;
  } else if (storage_index_first_free !=
             INVALID_U32) {
    set_storage_page_entry(vm, storage_index_first_free, page_num);
    return storage_index_first_free;
  } else {
    printf("[ERROR]: storage is full when trying to access page number: %d, in "
           "%s\n",
           page_num, __func__);
    return INVALID_U32;
  }
}

__device__ void swap(VirtualMemory *vm, uint32_t memory_page_index,
                     uint32_t storage_page_index,
                     uint32_t prev_memory_page_index) {
  uint32_t physical_memory_addr = memory_page_index * vm->PAGESIZE;
  uint32_t storage_addr = storage_page_index * vm->PAGESIZE;

  uchar temp_buffer[32];

  memcpy(&temp_buffer, &vm->buffer[physical_memory_addr], vm->PAGESIZE);
  memcpy(&vm->buffer[physical_memory_addr], &vm->storage[storage_addr],
         vm->PAGESIZE);
  memcpy(&vm->storage[storage_addr], &temp_buffer, vm->PAGESIZE);

  memory_page_entry memory_entry = get_memory_page_entry(vm, memory_page_index);
  uint16_t storage_page_num = get_storage_page_entry(vm, storage_page_index);

  set_storage_page_entry(vm, storage_page_index, memory_entry.virtual_page_num);
  memory_entry.virtual_page_num = storage_page_num;
  memory_entry.update_to_table(vm);
  // set_memory_page_entry(vm, memory_page_index, memory_entry);

  move_index_to_mru(vm, memory_page_index, prev_memory_page_index);
}

__device__ uint32_t get_page_table_index(VirtualMemory *vm, uint32_t page_num) {
  uint32_t mru_index = get_mru_index(vm);
  uint32_t lru_index = get_lru_index(vm);

  uint32_t prev_lru_index = INVALID_U32;
  uint32_t prev_index = INVALID_U32;
  uint32_t current_index = mru_index;

  while (prev_index != lru_index) {
    memory_page_entry entry = get_memory_page_entry(vm, current_index);

    if (entry.invalid_bit) {
      entry.invalid_bit = 0;
      entry.virtual_page_num = page_num;
      entry.update_to_table(vm);
      // set_memory_page_entry(vm, current_index, entry);
      move_index_to_mru(vm, current_index, prev_index);
      ++*(vm->pagefault_num_ptr);
      return current_index;
    }

    if (entry.virtual_page_num == page_num) {
      move_index_to_mru(vm, current_index, prev_index);
      return current_index;
    }

    prev_lru_index = prev_index;
    prev_index = current_index;
    current_index = entry.lru_chain_next;
  }

  ++*(vm->pagefault_num_ptr);
  uint32_t storage_index = find_page_in_storage(vm, page_num);
  if (storage_index != INVALID_U32) {
    swap(vm, lru_index, storage_index, prev_lru_index);
    return lru_index;
  } else {
    printf("[ERROR]: storage is full when trying to access page number: %d, in "
           "%s\n",
           page_num, __func__);
    return INVALID_U32;
  }
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  uint32_t write_virtual_page_num = addr / vm->PAGESIZE;
  uint32_t page_index = get_page_table_index(vm, write_virtual_page_num);
  if (page_index != INVALID_U32) {
    return vm->buffer[page_index * vm->PAGESIZE + (addr % vm->PAGESIZE)];
  } else {
    printf("[ERROR]: invalid address: %d, in "
           "%s\n",
           addr, __func__);
    return 0; // TODO
  }
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  uint32_t write_virtual_page_num = addr / vm->PAGESIZE;
  uint32_t page_index = get_page_table_index(vm, write_virtual_page_num);
  if (page_index != INVALID_U32) {
    vm->buffer[page_index * vm->PAGESIZE + (addr % vm->PAGESIZE)] = value;
  } else {
    printf("[ERROR]: storage is full when trying to write page number: %d, in "
           "%s\n",
           write_virtual_page_num, __func__);
  }
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for (uint32_t i = offset; i < offset + input_size; i++) {
    results[i - offset] = vm_read(vm, i);
  }
}
