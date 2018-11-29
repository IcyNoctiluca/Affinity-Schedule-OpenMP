#include <stdio.h>
#include <math.h>
// B138101

#define N 729
#define reps 1000
#include <omp.h>

double a[N][N], b[N][N], c[N];
int jmax[N];

// number of threads available for OpenMP
int totalThreads;

// struct to hold variables relating to each thread
struct Thread {
   int startItNum;      // the starting iteration number of the local set which the thread is assigned
   int finishItNum;     // the finishing iteration number of the local set which the thread is assigned
   int itsCompleted;    // how many iterations of the local set the thread has completed
};

// data structure to represent a pair of bounds executing a chunk
// needed when determining the busiest thread and the bounds of the underlying largest chunk
struct HighLow {
  int low;
  int high;
  int busiestThreadID;
};

void init1(void);
void init2(void);
void runloop(int);
void loop1chunk(int, int);
void loop2chunk(int, int);
void valid1(void);
void valid2(void);

struct HighLow getMostLoadedThreadsNextChunk(struct Thread[]);


int main(int argc, char *argv[]) {


  // initialise the value of total thread number
  #pragma omp parallel default(none) shared(totalThreads)
  {
    #pragma omp single
    {
      totalThreads = omp_get_max_threads();
    }
  }

  double start1,start2,end1,end2;
  int r;

  init1();

  start1 = omp_get_wtime();

  for (r=0; r<reps; r++){
    runloop(1);
  }

  end1  = omp_get_wtime();

  valid1();

  printf("Total time for %d reps of loop 1 = %f\n",reps, (float)(end1-start1));


  init2();


  start2 = omp_get_wtime();

  for (r=0; r<reps; r++){
    runloop(2);
  }

  end2  = omp_get_wtime();

  valid2();

  printf("Total time for %d reps of loop 2 = %f\n",reps, (float)(end2-start2));

}

void init1(void){
  int i,j;

  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      a[i][j] = 0.0;
      b[i][j] = 3.142*(i+j);
    }
  }

}

void init2(void){
  int i,j, expr;

  for (i=0; i<N; i++){
    expr =  i%( 3*(i/30) + 1);
    if ( expr == 0) {
      jmax[i] = N;
    }
    else {
      jmax[i] = 1;
    }
    c[i] = 0.0;
  }

  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      b[i][j] = (double) (i*j+1) / (double) (N*N);
    }
  }

}


void runloop(int loopid) {

  // initialise array of thread structs
  struct Thread threadProgress[totalThreads];

  #pragma omp parallel default(none) shared(loopid, threadProgress, totalThreads)
  {
    // set up per-thread variables
    int thisThreadNumber  = omp_get_thread_num();
    int iterationsPerThread = (int)ceil((double)N / (double)totalThreads);

    // compute iteration bounds of this thread's local set
    int startIterationNumber = thisThreadNumber * iterationsPerThread;
    int finishIterationNumber = (thisThreadNumber + 1) * iterationsPerThread;

    // upper bound on the finishing thread number
    if (finishIterationNumber > N) {
        finishIterationNumber = N;
    }

    // each thread updates its Thread struct variables
    threadProgress[thisThreadNumber].startItNum = startIterationNumber;
    threadProgress[thisThreadNumber].finishItNum = finishIterationNumber;
    threadProgress[thisThreadNumber].itsCompleted = 0;

    // require each thread to have updated its respective Thread struct before continuing
    #pragma omp barrier

    // init boundaries of upcoming chunk in the local set
    int low;
    int high;

    // only allow a single thread to access the shared array variable at a time
    #pragma omp critical
    {
      // get chunk boundaries based on remaining threads in local set
      int remainingIterations = threadProgress[thisThreadNumber].finishItNum - threadProgress[thisThreadNumber].startItNum - threadProgress[thisThreadNumber].itsCompleted;
      low = threadProgress[thisThreadNumber].startItNum + threadProgress[thisThreadNumber].itsCompleted;
      high = low + ceil((double)remainingIterations / (double)totalThreads);

      // upper bound on the finishing thread number
      if (high > N) {
          high = N;
      }

      // update the iterations completed before execution so other threads don't complete them also
      threadProgress[thisThreadNumber].itsCompleted += high - low;
    }

    // do the local set of iterations while there are still some to do
    do {

      switch (loopid) {
         case 1: loop1chunk(low, high); break;
         case 2: loop2chunk(low, high); break;
      }

      // get next chunk boundaries in the local set
      // only allow a single thread to access the shared array variable
      #pragma omp critical
      {
        int remainingIterations = threadProgress[thisThreadNumber].finishItNum - threadProgress[thisThreadNumber].startItNum - threadProgress[thisThreadNumber].itsCompleted;
        low = threadProgress[thisThreadNumber].startItNum + threadProgress[thisThreadNumber].itsCompleted;
        high = low + ceil((double)remainingIterations / (double)totalThreads);

        // upper bound on the finishing thread number
        if (high > N) {
            high = N;
        }

        // update the iterations completed before execution so other threads don't complete them also
        threadProgress[thisThreadNumber].itsCompleted += high - low;
      }


    } while(high != low);    // if high = low, then number of remaining iterations = 0, so break the do loop


    // once finished the local set, help out other threads

    // get next chunk boundaries of all remaining working threads
    // only allow a single thread to access the shared array variable
    struct HighLow hl;      // init data structure to hold info on progress of all threads
    #pragma omp critical
    {
      // iteration boundaries of upcoming chunk of busiest thread
      hl = getMostLoadedThreadsNextChunk(threadProgress);

      // update the iterations completed before execution so other threads don't complete them also
      threadProgress[hl.busiestThreadID].itsCompleted += hl.high - hl.low;
    }

    do {

      switch (loopid) {
         case 1: loop1chunk(hl.low, hl.high); break;
         case 2: loop2chunk(hl.low, hl.high); break;
      }

      // get next chunk boundaries
      // only allow a single thread to access the shared variable
      #pragma omp critical
      {
        // iterations of upcoming chunk of busiest thread
        hl = getMostLoadedThreadsNextChunk(threadProgress);

        // update the iterations completed before execution so other threads don't complete them also
        threadProgress[hl.busiestThreadID].itsCompleted += hl.high - hl.low;
      }

    } while(hl.high != hl.low);

    // all iterations are complete

  }
}


// returns the chunk boundaries of the busiest thread's local set
struct HighLow getMostLoadedThreadsNextChunk(struct Thread threadProgress[]) {

  int busiestThreadID = 0;

  // initialise difference to be worst possible value
  int largestRemainingIterations = 0;

  // initialise returning variable
  struct HighLow hl = {0, 0, 0};

  // find remaining iterations for each thread
  for (int i = 0; i < totalThreads; i++) {

    // calculate number remaining iterations for the given thread
    int remainingIts = threadProgress[i].finishItNum - threadProgress[i].startItNum - threadProgress[i].itsCompleted;

    // if remaining greater than current largest, update
    if (remainingIts >= largestRemainingIterations) {

      largestRemainingIterations = remainingIts;
      hl.busiestThreadID = i;
      hl.low = threadProgress[i].startItNum + threadProgress[i].itsCompleted;
      hl.high = hl.low + ceil((double)remainingIts / (double)totalThreads);

      // upper bound on the finishing thread number
      if (hl.high > N) {
          hl.high = N;
      }
    }
  }

  return hl;
}


void loop1chunk(int lo, int hi) {
  int i,j;

  for (i=lo; i<hi; i++){
    for (j=N-1; j>i; j--){
      a[i][j] += cos(b[i][j]);
    }
  }

}



void loop2chunk(int lo, int hi) {
  int i,j,k;
  double rN2;

  rN2 = 1.0 / (double) (N*N);

  for (i=lo; i<hi; i++){
    for (j=0; j < jmax[i]; j++){
      for (k=0; k<j; k++){
	       c[i] += (k+1) * log (b[i][j]) * rN2;
      }
    }
  }

}

void valid1(void) {
  int i,j;
  double suma;

  suma= 0.0;
  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      suma += a[i][j];
    }
  }
  printf("Loop 1 check: Sum of a is %lf\n", suma);

}


void valid2(void) {
  int i;
  double sumc;

  sumc= 0.0;
  for (i=0; i<N; i++){
    sumc += c[i];
  }
  printf("Loop 2 check: Sum of c is %f\n", sumc);
}
