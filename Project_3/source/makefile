all:
		nvcc --relocatable-device-code=true main.cu user_program.cu virtual_memory.cu -o test

clean:
		rm result.out test snapshot.bin -rf
