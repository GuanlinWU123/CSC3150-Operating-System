#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define PATH "/proc"
#define BUFFSIZE_INFO 60

struct process_info {
	int pid;
	int ppid;
	char *comm;
};

struct process_info process_info_list[500];
int number_process = 0;

void set_process_info();
void print_child_process(int ppid, int pos, int pname_length);

void print_child_process(int ppid, int pos, int pname_length)
{
	int i = 0;
	int j;
	int temp_pos = 0;
	int length = pname_length;
	while (pos < number_process) {
		if (process_info_list[pos].ppid == ppid) {
			// printf("Enter if");
			if (i == 0) {
				temp_pos = pos;
			}
			i++;
		}
		pos++;
	}

	if (i > 1) {
		printf("——|——%s\n", process_info_list[temp_pos].comm);
	} else if (i == 1) {
		printf("——%s", process_info_list[temp_pos].comm);
		print_child_process(
			process_info_list[temp_pos].pid, temp_pos,
			length + strlen(process_info_list[temp_pos].comm));
	}
	i--;
	temp_pos++;

	while (i > 0) {
		if (process_info_list[temp_pos].ppid == ppid) {
			if (i <= 2) {
				j = length + 18;
			} else {
				j = length;
			}
			for (j; j--; j > 0) {
				printf(" ");
			}
			printf("|——%s", process_info_list[temp_pos].comm);
			print_child_process(
				process_info_list[temp_pos].pid, temp_pos,
				length + strlen(process_info_list[temp_pos]
							.comm));
			printf("\n");
			i--;
		}
		temp_pos++;
	}
}

void set_process_info()
{
	DIR *dir_ptr;
	struct dirent *direntp;
	int pid;
	int ppid;
	char process_path[51] = "/proc/";
	char stat[6] = "/stat";
	char pidStr[20];
	char process_info[BUFFSIZE_INFO + 1];

	dir_ptr = opendir(PATH);

	if (dir_ptr == NULL) {
		fprintf(stderr, "can not open /proc\n");
		exit(0);
	}

	while (direntp = readdir(dir_ptr)) {
		pid = atoi(direntp->d_name);
		if (pid != 0) {
			process_info_list[number_process].pid = pid;
			sprintf(pidStr, "%d", pid);
			strcat(process_path, pidStr);
			strcat(process_path, stat);

			FILE *fp = fopen(process_path, "r");
			process_info[BUFFSIZE_INFO] = '\0';
			char *right_ptr;
			char *left_ptr;
			int signal_name = 0;
			int start_pos, end_pos;
			if (fp == NULL) {
				printf(stderr, "open file error!\n");

			} else {
				fgets(process_info, BUFFSIZE_INFO, fp);
				for (int i = 0; i < BUFFSIZE_INFO; i++) {
					if (process_info[i] == '(') {
						start_pos = i + 1;

					} else if (process_info[i] == ')') {
						end_pos = i;
					}
				}
				int needed_length = end_pos - start_pos;
				char *comm = (char *)malloc(sizeof(char) *
							    needed_length * 4);

				strncpy(comm, process_info + start_pos,
					needed_length);
				comm[needed_length] = '\0';
				right_ptr = strrchr(process_info, ')');
				right_ptr += 3;
				sscanf(right_ptr, "%d", &ppid);
				process_info_list[number_process].comm = comm;
				process_info_list[number_process].ppid = ppid;
				number_process++;
			}
			process_path[6] = 0;
		}
	}
}

int main()
{
	set_process_info();
	int out_time = 0;
	for (int i = 0; i < number_process; i++) {
		if (process_info_list[i].pid == 1) {
			printf("%s——", process_info_list[i].comm);
		}

		if (process_info_list[i].ppid == 1) {
			if (out_time == 0) {
				printf("|——%s", process_info_list[i].comm);
				print_child_process(
					process_info_list[i].pid, i,
					strlen(process_info_list[i].comm) + 8);
				printf("\n");
				out_time = 1;
				continue;
			} else {
				printf("         |——%s",
				       process_info_list[i].comm);
				print_child_process(
					process_info_list[i].pid, i,
					strlen(process_info_list[i].comm) + 8);
				printf("\n");
			}
		}
	}
	return 0;
}