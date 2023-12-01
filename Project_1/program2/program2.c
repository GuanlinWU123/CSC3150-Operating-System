#include <linux/delay.h>
#include <linux/err.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/pid.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/sched/task.h>
#include <linux/slab.h>
#include <linux/threads.h>
#include <linux/wait.h>

#define __WAIT_INT(status) (status)
#define __WEXITSTATUS(status) (((status)&0xff00) >> 8)
#define __WTERMSIG(status) ((status)&0x7f)
#define __WSTOPSIG(status) __WEXITSTATUS(status)
#define __WIFEXITED(status) (__WTERMSIG(status) == 0)
#define __WIFSIGNALED(status) (((signed char)(((status)&0x7f) + 1) >> 1) > 0)
#define __WIFSTOPPED(status) (((status)&0xff) == 0x7f)

#define WEXITSTATUS(status) __WEXITSTATUS(__WAIT_INT(status))
#define WTERMSIG(status) __WTERMSIG(__WAIT_INT(status))
#define WSTOPSIG(status) __WSTOPSIG(__WAIT_INT(status))
#define WIFEXITED(status) __WIFEXITED(__WAIT_INT(status))
#define WIFSIGNALED(status) __WIFSIGNALED(__WAIT_INT(status))
#define WIFSTOPPED(status) __WIFSTOPPED(__WAIT_INT(status))

struct wait_opts {
	enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;

	struct waitid_info *wo_info;
	int wo_stat;
	struct rusage *wo_rusage;

	wait_queue_entry_t child_wait;
	int notask_error;
};

extern pid_t kernel_clone(struct kernel_clone_args *args);
extern struct filename *getname_kernel(const char *filename);
extern int do_execve(struct filename *filename,
		     const char __user *const __user *__argv,
		     const char __user *const __user *__envp);

extern long do_wait(struct wait_opts *wo);

MODULE_LICENSE("GPL");

static struct task_struct *task;

int my_exec(void)
{
	int result;
	const char path[] = "/tmp/test";
	// const char path[] = "/home/seed/work/Assignment1/source/program2/test";
	const char *const argv[] = { path, NULL, NULL };
	const char *const envp[] = { "HOME=/",
				     "PATH=/sbin:/user/sbin:/bin:/usr/bin",
				     NULL };

	struct filename *my_filename = getname_kernel(path);
	printk("[program2] : child process");
	result = do_execve(my_filename, NULL, NULL);
	// printk("[program2] : [do_execve] : the returned is %d\n", result);
	if (!result) {
		return 0;
	}

	do_exit(result);
}

int my_wait(pid_t pid, int *status)
{
	int retval;
	struct wait_opts wo;
	struct pid *wo_pid = NULL;
	enum pid_type type;
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);
	// printk("[program2] : [my_wait] : pid is %d and get_pid ptr is %p\n", pid,
	// wo_pid);

	wo.wo_type = type;
	wo.wo_pid = wo_pid;
	wo.wo_flags = WEXITED | WUNTRACED;
	wo.wo_info = NULL;
	wo.wo_rusage = NULL;

	retval = do_wait(&wo);

	put_pid(wo_pid);
	*status = wo.wo_stat;
	// printk("[program2] : [my_wait] : exit wait, status %d", status);
	return retval;
}

// implement fork function
int my_fork(void *argc)
{
	pid_t pid;

	// set default sigaction for current process
	int i;

	struct k_sigaction *k_action = &current->sighand->action[0];
	for (i = 0; i < _NSIG; i++) {
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}

	/* fork a process using kernel_clone or kernel_thread */

	struct kernel_clone_args args = {
		.flags = SIGCHLD,
		.stack = &my_exec,
		.child_tid = NULL,
		.parent_tid = NULL,
	};

	pid = kernel_clone(&args);
	printk("[program2] : The Child process has pid = %d\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n",
	       (int)current->pid);

	/* execute a test program in child process */
	msleep(50);
	int status = 0;
	int retval = my_wait(pid, &status);
	/* wait until child process terminates */
	if (WIFSTOPPED(status)) {
		printk("[program2] : get SIGSTOP signal\n");
		// printk("[program2] : child process stopped\n");
		printk("[program2] : the return signal is %d\n",
		       WTERMSIG(status));

	} else if (WIFSIGNALED(status)) {
		switch (WTERMSIG(status)) {
		case SIGABRT:
			printk("[program2] : get SIGABRT signal\n");
			break;

		case SIGALRM:
			printk("[program2] : get SIGALRM signal\n");
			break;

		case SIGBUS:
			printk("[program2] : get SIGBUS signal\n");
			break;

		case SIGCONT:
			printk("[program2] : get SIGCONT signal\n");
			break;

		case SIGFPE:
			printk("[program2] : get SIGFPE signal\n");
			break;

		case SIGHUP:
			printk("[program2] : get SIGHUP signal\n");
			break;

		case SIGILL:
			printk("[program2] : get SIGILL signal\n");
			break;

		case SIGINT:
			printk("[program2] : get SIGINT signal\n");
			break;

		case SIGIO:
			printk("[program2] : get SIGIO signal\n");
			break;

		case SIGKILL:
			printk("[program2] : get SIGKILL signal\n");
			break;

		case SIGPIPE:
			printk("[program2] : get SIGPIPE signal\n");
			break;

		case SIGPROF:
			printk("[program2] : get SIGPROF signal\n");
			break;

		case SIGPWR:
			printk("[program2] : get SIGPWR signal\n");
			break;

		case SIGQUIT:
			printk("[program2] : get SIGQUIT signal\n");
			break;

		case SIGSEGV:
			printk("[program2] : get SIGSEGV signal\n");
			break;

		case SIGSTKFLT:
			printk("[program2] : get SIGSTKFLT signal\n");
			break;

		case SIGTSTP:
			printk("[program2] : get SIGTSTP signal\n");
			break;

		case SIGSYS:
			printk("[program2] : get SIGSYS signal\n");
			break;

		case SIGTERM:
			printk("[program2] : get SIGTERM signal\n");
			break;

		case SIGTRAP:
			printk("[program2] : get SIGTRAP signal\n");
			break;

		case SIGTTIN:
			printk("[program2] : get SIGTTIN signal\n");
			break;

		case SIGTTOU:
			printk("[program2] : get SIGTTOU signal\n");
			break;

		case SIGURG:
			printk("[program2] : get SIGURG signal\n");
			break;

		case SIGVTALRM:
			printk("[program2] : get SIGVTALRM signal\n");
			break;

		case SIGXCPU:
			printk("[program2] : get SIGXCPU signal\n");
			break;

		case SIGXFSZ:
			printk("[program2] : get SIGXFSZ signal\n");
			break;

		default:
			printk("[program2] : get UNKNOWN signal\n");
			break;
		}
		printk("[program2] : child process terminated\n");
		printk("[program2] : the return signal is %d\n",
		       WTERMSIG(status));

	} else if (WIFEXITED(status)) {
		printk("[program2] : child process exited\n");
		printk("[program2] : the return signal is %d\n",
		       WIFEXITED(status));

	} else {
		printk("this is the unreachable else branch\n");
	}

	// printk("[program2] : Module_exit\n");
	// do_exit(0);
	return 0;
}

static int __init program2_init(void)
{
	printk("[program2] : module_init Guanlin Wu 120010048\n");

	/* write your code here */

	/* create a kernel thread to run my_fork */

	printk("[program2] : module_init create kthread start\n");

	task = kthread_create(&my_fork, NULL, "newThread");

	if (!IS_ERR(task)) {
		printk("[program2] : module_init kthread start\n");
		wake_up_process(task);
	}

	return 0;
}

static void __exit program2_exit(void)
{
	printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);