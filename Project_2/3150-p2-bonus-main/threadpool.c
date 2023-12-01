#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cvs[5];

int thread_ids[5] = {0, 1, 2, 3, 4};

void *thread_task(void *idp){
    int *my_id = (int*)idp;

    // pthread_mutex_lock(&count_mutex);
    while (1){
        if (*my_id == 0){
            pthread_cond_wait(&count_threshold_cvs[0], &count_mutex);
            pthread_mutex_lock(&count_mutex);
            printf("count_threshold_cvs number: 0\n");
            pthread_mutex_unlock(&count_mutex);
            break;

        } else if (*my_id == 1){
            pthread_cond_wait(&count_threshold_cvs[1], &count_mutex);
            pthread_mutex_lock(&count_mutex);
            printf("count_threshold_cvs number: 1\n");
            pthread_mutex_unlock(&count_mutex);
            break;

        } else if (*my_id == 2){
            pthread_cond_wait(&count_threshold_cvs[2], &count_mutex);
            pthread_mutex_lock(&count_mutex);
            printf("count_threshold_cvs number: 2\n");
            pthread_mutex_unlock(&count_mutex);
            break;

        } else if (*my_id == 3){
            pthread_cond_wait(&count_threshold_cvs[3], &count_mutex);
            pthread_mutex_lock(&count_mutex);
            printf("count_threshold_cvs number: 3\n");
            pthread_mutex_unlock(&count_mutex);
            break;

        } else if (*my_id == 4){
            pthread_cond_wait(&count_threshold_cvs[4], &count_mutex);
            pthread_mutex_lock(&count_mutex);
            printf("count_threshold_cvs number: 4\n");
            pthread_mutex_unlock(&count_mutex);
            break;
        }
    }
    // pthread_mutex_unlock(&count_mutex);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]){
    pthread_t threads[5];
    pthread_attr_t attr;

    pthread_mutex_init(&count_mutex, NULL);
    for (int i = 0; i < 5; i++){
        pthread_cond_init(&count_threshold_cvs[i], NULL);
    }

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    for (int j = 0; j < 5; j++){
        // pthread_create(&threads[i], &attr, thread_task, (void *)&thread_ids[i]);
        pthread_create(&threads[j], &attr, thread_task, (void *)&j);
    }

    for (int i = 0; i < 5; i++){
        // pthread_mutex_lock(&count_mutex);
        pthread_cond_signal(&count_threshold_cvs[i]);
        // sleep(1);
        // pthread_mutex_unlock(&count_mutex);
    }

    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&count_mutex);
    for (int i = 0; i < 5; i++){
        pthread_cond_destroy(&count_threshold_cvs[i]);
    }
    pthread_exit(NULL);
    return 0;
}