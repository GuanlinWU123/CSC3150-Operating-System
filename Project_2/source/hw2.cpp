#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 

int kbhit(void);
void *check_quit(void *t);
void *logs_move( void *t );

struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 


char map[ROW+10][COLUMN];
int column_pos[9];
int sign_map[9] = {-1, 1, -1, 1, -1, 1, -1, 1, -1};
int counter_nums[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
int add_sign[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
int frog_sign = 1;
int end_sign = 0;
int quit_sign = 0;
pthread_mutex_t mutex;
// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 

void clean_console() {
	printf("\033[0;0H\033[2J");
}

void update_console() {
	clean_console();
	// pthread_mutex_lock(&mutex);
	for(int i = 0; i <= ROW; ++i)	
		puts( map[i] );
	usleep(70000);
	// pthread_mutex_unlock(&mutex);
}

void *check_quit(void *t){
	pthread_mutex_lock(&mutex);
	if (frog.x != 10){
		int temp_col_pos = column_pos[frog.x - 1];
        // if (temp_col_pos = 48 - counter_nums[frog.x - 1]){}
		if (frog.x == 0 || frog.y == 0 || frog.y == 48){
			end_sign = 1;
		
		} else if (temp_col_pos == 0 && frog.y > 15-counter_nums[frog.x - 1]){
			if (frog.y < 20){
				end_sign = 1;
			} else {
				end_sign = 0;
			}

		// } else if (temp_col_pos == 0 && frog.y > 15-counter_nums[frog.x - 1])

        } else if (frog.y < temp_col_pos || frog.y > temp_col_pos+15){
            if (temp_col_pos == 0 && frog.y > 48-counter_nums[frog.x - 1]){
                end_sign = 0;
            } else if (frog.y < temp_col_pos && temp_col_pos > 35 && frog.y < 15){
				end_sign = 0;
			} else {
            end_sign = 1;
            }
                
		} else {
			end_sign = 0;
		}

	} else if (frog.y == 0|| frog.y == 48){
		end_sign = 1;
	}

	pthread_mutex_unlock(&mutex);
	pthread_exit(NULL);
	return 0;
}

int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

void *logs_move( void *t ){
	pthread_mutex_lock(&mutex);

	long log_id;
	int count_int, row_int, column_int, temp_sign;
	int move_sign; //1:right, -1:left;
	count_int = 0;
	log_id = (long) t;

	row_int = (int) log_id;
	column_int = column_pos[row_int];
	// printf("row_int: %d, column_int: %d\n", row_int, column_int);

	/*  Move the logs  */
    int temp_num = counter_nums[row_int];
    move_sign = sign_map[row_int];
		 
    if (move_sign == 1){

        if (frog.x == row_int+1){
			map[row_int+1][frog.y] = '=';
			frog.y++;
		}

        if (add_sign[row_int] == 1){
            map[row_int+1][column_int] = ' ';
            map[row_int+1][temp_num] = '=';
            column_int++;
            counter_nums[row_int]++;
            // printf("counter_nums[%d]: %d", row_int, counter_nums[row_int]);
            if (temp_num == 14){
                column_int = 0;
            }

        } else {
            map[row_int+1][column_int] = ' ';
		    column_int++;
		    map[row_int+1][column_int+14] = '=';
        }

	} else{
		if (frog.x == row_int+1){
			map[row_int+1][frog.y] = '=';
			frog.y--;
		}

        if (add_sign[row_int] == 1){
            map[row_int+1][14-temp_num] = ' ';
            map[row_int+1][48-temp_num] = '=';
            // column_int = 48-temp_num;
            counter_nums[row_int]++;
            if (temp_num == 14){
                column_int = 34;
            }

        } else {
            map[row_int+1][column_int+14] = ' ';
		    column_int--;
		    map[row_int+1][column_int] = '=';
        }
	}

	column_pos[row_int] = column_int;

    if (column_int == 0 || column_int+14 == 48){
		add_sign[row_int] = 1;

	} 

    if (counter_nums[row_int] == 15){
        add_sign[row_int] = 0;
        counter_nums[row_int] = 0;
    }



	/*  Check keyboard hits, to change frog's position or quit the game. */
	if (kbhit()){
			char dir = getchar();
			if (dir == 'w' || dir == 'W'){
				if (frog.x == 10){
					map[frog.x][frog.y] = ' ';
				} else {
					map[frog.x][frog.y] = '=';
				}
				frog.x -= 1;

			} else if (dir == 's' || dir == 'S'){
				if (frog.x == 10){
					frog.x = frog.x;
				} else {
					map[frog.x][frog.y] = '=';
					frog.x += 1;
				}

			} else if (dir == 'a' || dir == 'A'){
				if (frog.x == 10){
					map[frog.x][frog.y] = '|';
				} else {
					map[frog.x][frog.y] = '=';
				}
				frog.y -= 1;

			} else if (dir == 'd' || dir == 'D'){
				if (frog.x == 10){
					map[frog.x][frog.y] = '|';
				} else {
					map[frog.x][frog.y] = '=';
				}
				frog.y += 1;

			} else if (dir == 'q' || dir == 'Q'){
				end_sign = 1;
				quit_sign = 1;
			} else {
				NULL;
			}
		}

	pthread_mutex_unlock(&mutex);

	if (frog.x == row_int+1){
		// if (column_int < frog.y <= column_int+14){
		// 	map[row_int+1][frog.y] = '0';
		// }

		if (column_int <= frog.y || frog.y <= column_int+14){
			map[row_int+1][frog.y] = '0';
		}
	}

	/*  Check game's status  */


	/*  Print the map on the screen  */

	pthread_exit(NULL);
	return 0;
}

int main( int argc, char *argv[] ){

	pthread_t log_threads[9];
	pthread_t check_thread;
	pthread_mutex_init(&mutex, NULL);
	int log_return, check_return;
	long thread_num = 0;
	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;

	int i, j, column_int, print_int;
	column_int = 5; 

	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ; 

	int pos_i = 0;

	while (pos_i < 9){
		int rand_num = rand()%33;
		if (rand_num <= 3){
			continue;
		} else {
			column_pos[pos_i] = rand_num;
			pos_i++;
		}
	}

	for (i=1; i<10; i++){
		column_int = column_pos[i-1];
		for (int j=column_int; j<15+column_int; j++){
			map[i][j] = '=';
		}
	}

	/*  Create pthreads for wood move and frog control.  */


	while (!end_sign){

		for (i=0; i<9; i++){
			if (i == 0){
				thread_num = 0;
			}
			log_return = pthread_create(&log_threads[i], NULL, logs_move, (void*)thread_num);
			if (log_return){
				printf("ERROR: LOG thread creation fails! thread number: %d", i);
			}
			thread_num++;
		}
        
		// pthread_join(check_thread, NULL);
		for (i=0; i<9; i++){
			pthread_join(log_threads[i], NULL);
		}

        check_return = pthread_create(&check_thread, NULL, check_quit, NULL);
		if (check_return){
				printf("ERROR: CHECK thread creation fails! thread number: %d", i);
			}

		if (frog.x == 10){
			map[frog.x][frog.y] = '0';
		}

		if (frog.x == 0){
			map[frog.x][frog.y] = '0';
		}
		update_console();
	}

	pthread_mutex_destroy(&mutex);

	clean_console();
	if (frog.x == 0){
		printf("You win! \n");
	} else if (quit_sign == 1) {
		printf("You quit \n");
	}
	else {
		printf("You lose! \n");
	}

	pthread_exit(NULL);
	return 0;
}
