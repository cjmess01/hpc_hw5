module load cuda/11.7.0
nvcc conway.cu -o conway 

./conway 5000 5000  ./scratch/5000test1.txt
./conway 5000 5000  ./scratch/5000test2.txt
./conway 5000 5000  ./scratch/5000test3.txt

./conway 10000 5000 ./scratch/10000test1.txt
./conway 10000 5000 ./scratch/10000test2.txt
./conway 10000 5000 ./scratch/10000test3.txt

diff ./scratch/5000test1.txt ./scratch/5000test2.txt
diff ./scratch/5000test1.txt ./scratch/5000test3.txt
diff ./scratch/10000test1.txt ./scratch/10000test2.txt
diff ./scratch/10000test1.txt ./scratch/10000test3.txt

diff ./scratch/5000test1.txt ./scratch/mpi.txt
diff ./scratch/5000test1.txt ./scratch/omp.txt

