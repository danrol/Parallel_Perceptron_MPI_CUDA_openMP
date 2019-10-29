Parallel Implementation of Perceptron (Binary Classification) Algorithm
Using MPI + CUDA + OPENMP



As part of Parallel and Distributed Computation course (10324) 
Afeka 2019




Abstract
Perceptron is one of the basic algorithms in the world of artificial neural networks. Perceptron is a single layer network and a multi-layer perceptron is called a neural network. Our motive is to fit a line that separates all a samples from the b samples. Because this algorithm is so widely used sequential solution is well studied. But what’s about parallel implementation? In this paper, we’ll discuss the parallel implementation of linear perceptron using MPI, NVIDIA CUDA , and OpenMP.

Sequential Implementation of Simplified Binary Classification algorithm
1.	Set α = α0
2.	Choose initial value of all components of the vector of weights W to be equal to zero.
3.	Cycle through all given points Xi in the order as it is defined in the input file
4.	For each point Xi define a sign of discriminant function f(Xi) = WT Xi. If the values of vector W is chosen properly then all points belonging to set A will have positive value of f(Xi) and all points belonging to set B will have negative value of f(Xi). The first point P that does not satisfies this criterion will cause to stop the check and immediate redefinition of the vector W:
W = W + [α*sign(f(P))] P
5.	 Loop through stages 3-4 till one of following satisfies:
a.	All given points are classified properly
b.	The number maximum iterations LIMIT is reached

6.	Find Nmis - the number of points that are wrongly classified, meaning that the value of f(Xi) is not correct for those points. Calculate a Quality of Classifier q according the formula

q = Nmis / N

7.	Check if the Quality of Classifier is reached (q is less than a given value QC). 
8.	Stop if q < QC.
9.	Increment the value of α:    α = α + α0.    Stop if α > αMAX
10.	Loop through stages 2- 9


Parallel Implementation of Simplified Binary Classification algorithm
When we are trying to parallelize algorithm, we must always ask: “What can be parallelized?”
Next tasks may be parallelized:
1.	Calculation for every alpha is independent one from another.  
2.	vectors addition in section 4 (weights update) complexity of sequential implementation O(n)
3.	f function (section 4 of sequential implementation) relatively heavy for computation (vectors multiplication):
Having 2 vectors like  a = [a1, a2, … , an] and b = [b1, b2, … , bn] then the dot-product is given by a.b = a1 * b1 + a2  * b2 + … an * bn.
To compute this, we must perform n multiplication and (n – 1) additions. Assuming multiplication and addition are constant-time operations, the time-complexity is therefore   O(n) + O(n) = O(2n) ≈ O(n)
O(2n) slightly bigger than O(n), usually scalar values neglected in time complexity but in this case, this will justify my decision to compute f function using CUDA. CUDA can create about a thousand threads at the same time that may be used for heavy computations. 
Naïve approach would be dividing all alphas between MPI slaves and then to get the smallest alpha solution from every process. The problem is this way if we have a lot of alphas too much time may be wasted in calculations. So smarter way is to share with processes some default number of alphas, get a result from them and then to send a new chunk of alphas if needed.
We will leave weights update to OpenMP same goes for quality check.
The algorithm is based on next data structures:
a.	One-dimension points array which consists of coordinates. To get coordinate j   of the point i will be done next way: points[i*K + j].
b.	The one-dimension array of weights vector
c.	One-dimension array of points groups. To find out to which group point number i belongs will be done next way: pointGroups[i].
Initial data read and jobs share/accumulation will be performed using MPICH.  





Master process algorithm:
1.	Read initialize data from txt file:
a.	N (numOfPoints) – number of points.
b.	K (numOfDimensions) – number of coordinates.
c.	Coordinates for every point.
d.	pointGroups – array with information about the type of group (1 or -1) for every point.
e.	alphaZero – by this value alpha will be raised every iteration.
f.	alphaMax – maximum value of alpha.
g.	LIMIT (limitIter) – the maximum number of iterations 
h.	qc – quality control.
2.	Share initialize data between slaves.
3.	Divide tasks (alphas needed to check) between slaves.
4.	Get results from slaves.
a.	If there is minimal alpha or in the case with q<qc or alphaMax’ve been reached continue to step 5.
b.	If there is no desired answer share next alpha’s chunks to slaves in steps 2-3.
5.	Command slaves to stop.
6.	Print results in “output.txt” file.
Slaves processes algorithm:
1.	Get initial data from the master.
2.	Get tasks chunk from the master.
3.	If the command to stop wasn’t received or first, alpha that slave receives is in allowed range (alpha < alphaMax), the process continues to step 4.
4.	For each point find f(Xi) = WTXi using CUDA and save results in an array of size N (cudaPoints[i]) shows the result of f(Xi) for point with index i).
5.	Loop through all cudaPoints if at some moment cudaPoints[i] != pointsGroups[i] update weights(w) using openMP.
6.	Loop through stages 3-4 until one of the following satisfies:
a.	All given points are classified properly.
b.	The number maximum iterations LIMIT is reached.
7.	Find Nmis the number of points that are wrongly classified, meaning that the value of f(Xi) is not correct for those points. Calculate a Quality of Classifier q according to the formula: q = Nmis/N. The calculation will be performed using openMP.
8.	Check if the Quality of Classifier is reached.
9.	Stop if q < qc.
10.	If q>qc increments the value of α = α + α0. Stop if α > αmax allowed to process.
11.	Loop through stages 4-10
12.	Send minimum alpha, weights , and q to Master.
13.	Return to step 2.
To summarize responsibilities of MPI, OpenMP, CUDA:
MPI responsibilities:
1.	Share of work to processes
2.	Processing of results received from processes
OpenMP responsibilities:
1.	Update weights (complexity of sequential weights update is O(n))
2.	Find Nmis (complexity of sequential Nmis calculation is O(n))
CUDA
Find f(Xi) for every point



