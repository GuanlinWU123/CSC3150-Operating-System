#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
	/* fork a child process */

	/* execute test program */

	/* wait for child process terminates */

	/* check child process'  termination status */

	pid_t pid;
	int status;
	const char **term_signal;

	printf("Process start to fork\n");
	pid = fork();

	if (pid == -1) {
		perror("fork");
		exit(1);
	} else {
		if (pid == 0) {
			sleep(1);
			int i;
			char *arg[argc];

			for (i = 0; i < argc - 1; i++) {
				arg[i] = argv[i + 1];
			}
			arg[argc - 1] = NULL;
			printf("I'm the Child Process, my pid = %d \n",
			       getpid());
			printf("Child process start to execute test program: \n");
			execve(arg[0], arg, NULL);
			// raise(SIGCHLD);

		} else {
			printf("I'm the Parent Process, my pid = %d \n",
			       getpid());

			waitpid(pid, &status, WUNTRACED);
			printf("Parent process receives SIGCHLD signal\n");

			if (WIFEXITED(status)) {
				printf("Normal termination with EXIT STATUS = %d \n",
				       WEXITSTATUS(status));

			} else if (WIFSIGNALED(status)) {
				switch (WTERMSIG(status)) {
				case SIGABRT:
					printf("Child process get SIGABRT signal\n");
					break;

				case SIGALRM:
					printf("Child process get SIGALRM signal\n");
					break;

				case SIGBUS:
					printf("Child process get SIGBUS signal\n");
					break;

				case SIGCONT:
					printf("Child process get SIGCONT signal\n");
					break;

				case SIGFPE:
					printf("Child process get SIGFPE signal\n");
					break;

				case SIGHUP:
					printf("Child process get SIGHUP signal\n");
					break;

				case SIGILL:
					printf("Child process get SIGILL signal\n");
					break;

				case SIGINT:
					printf("Child process get SIGINT signal\n");
					break;

				case SIGIO:
					printf("Child process get SIGIO signal\n");
					break;

				case SIGKILL:
					printf("Child process get SIGKILL signal\n");
					break;

				case SIGPIPE:
					printf("Child process get SIGPIPE signal\n");
					break;

				case SIGPROF:
					printf("Child process get SIGPROF signal\n");
					break;

				case SIGPWR:
					printf("Child process get SIGPWR signal\n");
					break;

				case SIGQUIT:
					printf("Child process get SIGQUIT signal\n");
					break;

				case SIGSEGV:
					printf("Child process get SIGSEGV signal\n");
					break;

				case SIGSTKFLT:
					printf("Child process get SIGSTKFLT signal\n");
					break;

				case SIGTSTP:
					printf("Child process get SIGTSTP signal\n");
					break;

				case SIGSYS:
					printf("Child process get SIGSYS signal\n");
					break;

				case SIGTERM:
					printf("Child process get SIGTERM signal\n");
					break;

				case SIGTRAP:
					printf("Child process get SIGTRAP signal\n");
					break;

				case SIGTTIN:
					printf("Child process get SIGTTIN signal\n");
					break;

				case SIGTTOU:
					printf("Child process get SIGTTOU signal\n");
					break;

				case SIGURG:
					printf("Child process get SIGURG signal\n");
					break;

				case SIGVTALRM:
					printf("Child process get SIGVTALRM signal\n");
					break;

				case SIGXCPU:
					printf("Child process get SIGXCPU signal\n");
					break;

				case SIGXFSZ:
					printf("Child process get SIGXFSZ signal\n");
					break;

				default:
					printf("get UNKNOWN signal\n");
					break;
				}

			} else if (WIFSTOPPED(status)) {
				printf("Child process get SIGSTOP signal\n");

			} else {
				printf("Child process get CONTINUE signal\n");
			}

			exit(0);
		}
	}

	return 0;
}
