#include <unistd.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>

int main(int argc,char* argv[]){
	int i=0;

	printf("--------USER PROGRAM--------\n");

	// abort();
	// alarm(2);
	// raise(SIGBUS);
	// raise(SIGFPE);
	// raise(SIGHUP);
	// raise(SIGILL);	
	// raise(SIGINT);
	// raise(SIGKILL);
	// raise(SIGPIPE);
	// raise(SIGQUIT);
	// raise(SIGSEGV);
	// raise(SIGSTOP);
	// raise(SIGTERM);
	raise(SIGTRAP);

	sleep(5);
	printf("user process success!!\n");
	printf("--------USER PROGRAM--------\n");
	return 100;
}

