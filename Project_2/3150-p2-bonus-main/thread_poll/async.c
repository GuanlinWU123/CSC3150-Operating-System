
#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"

pthread_mutex_t count_mutex;
// pthread_cond_t *count_threshold_cv_t;
pthread_cond_t *count_threshold_cvs;
struct my_queue task_queue = {0, NULL};
pthread_t *threads;

int count = 0;
int *threads_status;
long thread_num = 0;
// int count = 0;

void* thread_do(void *idp){
    int thread_id = (int) idp;

    while (1){
        pthread_mutex_lock(&count_mutex);

        // while (task_queue.size != 0){
        pthread_cond_wait(&count_threshold_cvs[thread_id], &count_mutex);
        // }

        // pthread_mutex_lock(&count_mutex);
        printf("after wait thread: %d..........\n", thread_id);
        
        if (task_queue.size == 0){
            continue;
        }

        printf("execution thread: %d..........\n", thread_id);
        // pthread_mutex_lock(&count_mutex);
        void (*temp_func)(int) = task_queue.head->function;
        int temp_arg = task_queue.head->args;
        DL_DELETE(task_queue.head, task_queue.head);
        task_queue.size--;
        pthread_mutex_unlock(&count_mutex);

        temp_func(temp_arg);
        // pthread_mutex_lock(&count_mutex);
        pthread_mutex_lock(&count_mutex);
        threads_status[thread_id] = 0;
        pthread_mutex_unlock(&count_mutex);
    }

    // pthread_exit(NULL);
}

void async_init(int num_threads) {
    /** TODO: create num_threads threads and initialize the thread pool **/

    pthread_attr_t attr;
    // task_queue->size = 0;
    // pthread_cond_t count_threshold_cvs[num_threads];

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    pthread_mutex_init(&count_mutex, NULL);
    count_threshold_cvs = (pthread_cond_t*) malloc (sizeof(pthread_cond_t)*num_threads);
    threads = (pthread_t*) malloc (sizeof(pthread_t)*num_threads);
    threads_status = (int*) malloc (sizeof(int)*num_threads);

    for (int i = 0; i < num_threads; i++){
        pthread_cond_init(&count_threshold_cvs[i], NULL);
    }

    for (int m = 0; m < num_threads; m++){
        pthread_create(&threads[m], &attr, thread_do, (void*) thread_num);
        // pthread_mutex_lock(&count_mutex);
        threads_status[m] = 0;
        thread_num++; 
        // pthread_mutex_unlock(&count_mutex);
    }

    return;
}

void async_run(void (*hanlder)(int), int args) {
    /** TODO: rewrite it to support thread pool **/
    count++;
    struct my_item *item_t;
    item_t = (struct my_item*) malloc (sizeof(struct my_item));

    item_t->function = hanlder; 
    item_t->args = args;   

    pthread_mutex_lock(&count_mutex);
    DL_APPEND(task_queue.head, item_t);
    task_queue.size++;
    pthread_mutex_unlock(&count_mutex);

    // hanlder(args);
    // printf("COUNT: %d\n", count);
    // pthread_mutex_lock(&count_mutex);
    while (task_queue.size != 0){
        // pthread_mutex_unlock(&count_mutex);
        for (int i=0; i < thread_num; i++){
            // pthread_mutex_unlock(&count_mutex);
            pthread_mutex_lock(&count_mutex);
            if (threads_status[i] == 0){
                pthread_mutex_unlock(&count_mutex);

                pthread_mutex_lock(&count_mutex);
                threads_status[i] = 1;
                pthread_mutex_unlock(&count_mutex);
                pthread_cond_signal(&count_threshold_cvs[i]);
                // pthread_mutex_unlock(&count_mutex);
                return;
            }
            pthread_mutex_unlock(&count_mutex); 
        }
    }
}