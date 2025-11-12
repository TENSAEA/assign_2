/* **********************************************************
 * Pthreads Parallel Code : Conways' game of life
 * 
 *  Author : Urvashi R.V. [04/06/2004]
 *     Modified by Scott Baden [10/8/06]
 *     Modified by Pietro Cicotti [10/8/08]
 *     Modified by Didem Unat [03/06/15]
 *     Parallelized with Pthreads
 *************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>

#define MATCH(s) (!strcmp(argv[ac], (s)))

int MeshPlot(int t, int m, int n, char **mesh);

double real_rand();
int seed_rand(long sd);

static char **currWorld=NULL, **nextWorld=NULL, **tmesh=NULL;
static int maxiter = 200; /* number of iteration timesteps */
static int population[2] = {0,0}; /* number of live cells */

int nx = 100;      /* number of mesh points in the x dimension */
int ny = 100;      /* number of mesh points in the y dimension */

static int w_update = 0;
static int w_plot = 1;

double getTime();
extern FILE *gnu;

/* Thread synchronization variables */
static pthread_mutex_t sync_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t comp_done = PTHREAD_COND_INITIALIZER;
static pthread_cond_t ready_for_next = PTHREAD_COND_INITIALIZER;
static int computation_complete = 0;
static int current_iteration = 0;
static int ready_to_compute = 1;  /* Start ready to compute iteration 0 */
static int has_plotting_thread = 0;  /* Whether plotting thread exists */

/* Thread data structure for computation threads */
typedef struct {
    int thread_id;
    int start_row;
    int end_row;
    int iteration;
} thread_data_t;

/* Global thread data */
static thread_data_t *thread_data_array = NULL;
static pthread_t *comp_threads = NULL;
static pthread_t plot_thread;
static int num_comp_threads = 0;
static int disable_display_flag = 0;
static int s_step_flag = 0;

/* Computation thread function */
void* compute_thread(void* arg)
{
    thread_data_t *data = (thread_data_t*)arg;
    int i, j;
    int local_pop = 0;
    int t;
    
    for (t = 0; t < maxiter && population[w_plot]; t++)
    {
        /* Wait until ready to compute this iteration */
        pthread_mutex_lock(&sync_mutex);
        while (!ready_to_compute || current_iteration != t) {
            pthread_cond_wait(&ready_for_next, &sync_mutex);
        }
        pthread_mutex_unlock(&sync_mutex);
        
        /* Compute updates for assigned rows */
        local_pop = 0;
        for (i = data->start_row; i < data->end_row; i++)
        {
            for (j = 1; j < ny-1; j++)
            {
                int nn = currWorld[i+1][j] + currWorld[i-1][j] + 
                    currWorld[i][j+1] + currWorld[i][j-1] + 
                    currWorld[i+1][j+1] + currWorld[i-1][j-1] + 
                    currWorld[i-1][j+1] + currWorld[i+1][j-1];
                
                nextWorld[i][j] = currWorld[i][j] ? (nn == 2 || nn == 3) : (nn == 3);
                local_pop += nextWorld[i][j];
            }
        }
        
        /* Update global population counter and check if last thread */
        pthread_mutex_lock(&sync_mutex);
        population[w_update] += local_pop;
        
        computation_complete++;
        if (computation_complete == num_comp_threads)
        {
            /* All computation threads finished - do pointer swap */
            tmesh = nextWorld;
            nextWorld = currWorld;
            currWorld = tmesh;
            
            /* Swap population counters */
            int temp = w_update;
            w_update = w_plot;
            w_plot = temp;
            
            /* Reset for next iteration */
            population[w_update] = 0;
            computation_complete = 0;
            current_iteration = t + 1;
            
            if (has_plotting_thread) {
                ready_to_compute = 0;  /* Not ready until plotting done */
                /* Signal plotting thread that computation is done */
                pthread_cond_broadcast(&comp_done);
            } else {
                ready_to_compute = 1;  /* No plotting thread, ready immediately */
                /* Signal master thread that computation is done */
                pthread_cond_broadcast(&comp_done);
                /* Also wake up computation threads for next iteration */
                pthread_cond_broadcast(&ready_for_next);
            }
        }
        pthread_mutex_unlock(&sync_mutex);
    }
    
    return NULL;
}

/* Plotting thread function */
void* plot_thread_func(void* arg)
{
    int t;
    
    for (t = 0; t < maxiter && population[w_plot]; t++)
    {
        if (!disable_display_flag)
        {
            /* Wait for computation to complete for this iteration */
            pthread_mutex_lock(&sync_mutex);
            while (current_iteration <= t) {
                pthread_cond_wait(&comp_done, &sync_mutex);
            }
            pthread_mutex_unlock(&sync_mutex);
            
            /* Plot the current generation */
            MeshPlot(t+1, nx, ny, currWorld);
        }
        
        if (s_step_flag) {
            printf("Finished with step %d\n", t);
            printf("Press enter to continue.\n");
            getchar();
        }
        
        /* Signal that plotting is done, ready for next iteration */
        pthread_mutex_lock(&sync_mutex);
        ready_to_compute = 1;
        pthread_cond_broadcast(&ready_for_next);
        pthread_mutex_unlock(&sync_mutex);
    }
    
    return NULL;
}

/* Helper function for when there's no plotting thread - master handles plotting */
void handle_plotting_no_thread(int t)
{
    if (!disable_display_flag) {
        MeshPlot(t+1, nx, ny, currWorld);
    }
    
    if (s_step_flag) {
        printf("Finished with step %d\n", t);
        printf("Press enter to continue.\n");
        getchar();
    }
}

int main(int argc,char **argv)
{
    int i,j,ac;

    /* Set default input parameters */
    
    float prob = 0.5;   /* Probability of placing a cell */
    long seedVal = 0;
    int game = 0;
    int s_step = 0;
    int numthreads = 1;
    int disable_display= 0; 

    /* Over-ride with command-line input parameters (if any) */
    
    for(ac=1;ac<argc;ac++)
    {
        if(MATCH("-n")) {nx = atoi(argv[++ac]);}
        else if(MATCH("-i")) {maxiter = atoi(argv[++ac]);}
        else if(MATCH("-t"))  {numthreads = atoi(argv[++ac]);}
        else if(MATCH("-p"))  {prob = atof(argv[++ac]);}
        else if(MATCH("-s"))  {seedVal = atol(argv[++ac]);}
        else if(MATCH("-step"))  {s_step = 1;}
        else if(MATCH("-d"))  {disable_display = 1;}
        else if(MATCH("-g"))  {game = atoi(argv[++ac]);}
        else {
            printf("Usage: %s [-n < meshpoints>] [-i <iterations>] [-s seed] [-p prob] [-t numthreads] [-step] [-g <game #>] [-d]\n",argv[0]);
            return(-1);
        }
    }

    int rs = seed_rand(seedVal);
    
    /* Store flags for threads */
    disable_display_flag = disable_display;
    s_step_flag = s_step;
    
    /* Determine number of computation threads */
    num_comp_threads = (numthreads > 1 && !disable_display) ? numthreads - 1 : numthreads;
    if (num_comp_threads < 1) num_comp_threads = 1;
    
    /* Increment sizes to account for boundary ghost cells */
    
    nx = nx+2;
    ny = nx; 
    
    /* Allocate contiguous memory for two 2D arrays of size nx*ny.
     * Two arrays are required because in-place updates are not
     * possible with the simple iterative scheme listed below */
    
    currWorld = (char**)malloc(sizeof(char*)*nx + sizeof(char)*nx*ny);
    for(i=0;i<nx;i++) 
      currWorld[i] = (char*)(currWorld+nx) + i*ny;
    
    nextWorld = (char**)malloc(sizeof(char*)*nx + sizeof(char)*nx*ny);
    for(i=0;i<nx;i++) 
      nextWorld[i] = (char*)(nextWorld+nx) + i*ny;
    
    /* Set the boundary ghost cells to hold 'zero' */
    for(i=0;i<nx;i++)
    {
        currWorld[i][0]=0;
        currWorld[i][ny-1]=0;
        nextWorld[i][0]=0;
        nextWorld[i][ny-1]=0;
    }
    for(i=0;i<ny;i++)
    {
        currWorld[0][i]=0;
        currWorld[nx-1][i]=0;
        nextWorld[0][i]=0;
        nextWorld[nx-1][i]=0;
    }

    // Generate a world - master thread initializes
    if (game == 0){ // Use Random input
        for(i=1;i<nx-1;i++)
            for(j=1;j<ny-1;j++) {
                currWorld[i][j] = (real_rand() < prob);
                population[w_plot] += currWorld[i][j];
            }
    }
    else if (game == 1){ //  Block, still life
        printf("2x2 Block, still life\n");
        int nx2 = nx/2;
        int ny2 = ny/2;
        currWorld[nx2+1][ny2+1] = currWorld[nx2][ny2+1] = currWorld[nx2+1][ny2] = currWorld[nx2][ny2] = 1;
        population[w_plot] = 4;
    }
    else if (game == 2){ //  Glider (spaceship)
        printf("Glider (spaceship)\n");
        int nx2 = nx/2;
        int ny2 = ny/2;
        // Glider pattern: 5 cells forming a moving pattern
        currWorld[nx2][ny2] = 1;
        currWorld[nx2+1][ny2+1] = 1;
        currWorld[nx2+2][ny2-1] = 1;
        currWorld[nx2+2][ny2] = 1;
        currWorld[nx2+2][ny2+1] = 1;
        population[w_plot] = 5;
    }
    else{
        printf("Unknown game %d\n",game);
        exit(-1);
    }
    
    printf("probability: %f\n",prob);
    printf("Random # generator seed: %d\n", rs);
    printf("Number of threads: %d (computation: %d, plotting: %d)\n", 
           numthreads, num_comp_threads, (numthreads > 1 && !disable_display) ? 1 : 0);

    /* Plot the initial data */
    if(!disable_display)
      MeshPlot(0,nx,ny,currWorld);
    
    /* Initialize population counter for updates */
    population[w_update] = 0;
    
    /* Calculate row decomposition for computation threads */
    int rows_per_thread = (nx - 2) / num_comp_threads;
    int extra_rows = (nx - 2) % num_comp_threads;
    
    /* Allocate thread data structures */
    thread_data_array = (thread_data_t*)malloc(sizeof(thread_data_t) * num_comp_threads);
    comp_threads = (pthread_t*)malloc(sizeof(pthread_t) * num_comp_threads);
    
    /* Initialize thread data and create computation threads */
    for (i = 0; i < num_comp_threads; i++)
    {
        thread_data_array[i].thread_id = i;
        thread_data_array[i].start_row = 1 + i * rows_per_thread + (i < extra_rows ? i : extra_rows);
        thread_data_array[i].end_row = thread_data_array[i].start_row + rows_per_thread + (i < extra_rows ? 1 : 0);
        thread_data_array[i].iteration = 0;
        
        pthread_create(&comp_threads[i], NULL, compute_thread, &thread_data_array[i]);
    }
    
    /* Create plotting thread if display is enabled and multiple threads */
    has_plotting_thread = (!disable_display && numthreads > 1);
    if (has_plotting_thread)
    {
        pthread_create(&plot_thread, NULL, plot_thread_func, NULL);
    }
    
    /* Perform updates for maxiter iterations */
    double t0 = getTime();
    
    /* If no plotting thread, master thread handles plotting */
    if (!has_plotting_thread)
    {
        int t;
        for (t = 0; t < maxiter && population[w_plot]; t++)
        {
            /* Wait for computation to complete */
            pthread_mutex_lock(&sync_mutex);
            while (current_iteration <= t) {
                pthread_cond_wait(&comp_done, &sync_mutex);
            }
            pthread_mutex_unlock(&sync_mutex);
            
            /* Master handles plotting */
            handle_plotting_no_thread(t);
            
            /* Signal ready for next iteration */
            pthread_mutex_lock(&sync_mutex);
            ready_to_compute = 1;
            pthread_cond_broadcast(&ready_for_next);
            pthread_mutex_unlock(&sync_mutex);
        }
    }
    
    /* Wait for all computation threads to complete */
    for (i = 0; i < num_comp_threads; i++)
    {
        pthread_join(comp_threads[i], NULL);
    }
    
    /* Wait for plotting thread to complete */
    if (has_plotting_thread)
    {
        pthread_join(plot_thread, NULL);
    }
    
    double t1 = getTime(); 
    printf("Running time for the iterations: %f sec.\n",t1-t0);
    printf("Press enter to end.\n");
    getchar();
    
    if(gnu != NULL)
      pclose(gnu);
    
    /* Clean up synchronization objects */
    pthread_mutex_destroy(&sync_mutex);
    pthread_cond_destroy(&comp_done);
    pthread_cond_destroy(&ready_for_next);
        
    /* Free resources */
    free(thread_data_array);
    free(comp_threads);
    free(nextWorld);
    free(currWorld);

    return(0);
}

