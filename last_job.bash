#!/bin/sh                                                                                 
LAST_JOBID=$(sacct -X -o jobid | tail -n 1)                                               
srun --pty --jobid $LAST_JOBID /bin/bash