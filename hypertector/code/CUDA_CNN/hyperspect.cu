#include <mat.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <iostream>
#include "cublas_v2.h"
#include "cokus.cpp"
#include "cuda_util.h"
#include <cuda_runtime.h>
using namespace std;

const int KER_NUM = 20;//卷积核数量
const int P_NUM = 3;//每次卷积的层数
const int LEAP = 2;//跳数
const int GP_NUM = 5;//maxpooling每组的个数
const int NEU_NUM1 = 100;
const int NEU_NUM2 = 20;//输出层神经元个数

//CUDA初始化
bool InitCUDA(){
	int count;
	cudaGetDeviceCount(&count);
	if(count==0){
		fprintf(stderr,"There is no device.\n");
		return false;
	}
	int i;
	for (i =0; i<count;i++){
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){
			if(prop.major>=1){																	break;
			}
		}
	}
	if(i==count){
		fprintf(stderr,"There is no device supporting CUDA 1.x.\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}

//copy数据到shared memory
__device__ void copy_data_to_shared(double * data, double * data_tmp, int length){
        for(int i=0; i<length; i++){
		 data_tmp[i] = data[i];
	}

        __syncthreads();
}

//GPU端负责卷积
__global__ static void convol(int iter,double * train,double * kernel,double * re,int x,int y,int z,int re_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;//保存当前线程编号

	//每个线程负责一个卷积核与一个3*3*hight柱状图像的卷积
	if (id < KER_NUM){
		extern __shared__ double train_tmp[];

		copy_data_to_shared(train,train_tmp,x*y*z);//复制train到shared memory中

		/*double * ker = new double [x*y*P_NUM];//载入对应的kernel到寄存器
		for(int i=0; i<x*y*P_NUM; i++){
			ker[i] = kernel[id*x*y*P_NUM + i];
		}*/

		int dim_x = 0, dim_y = 0, dim_z = 0;//初始位置为(0,0,0)

		double mid;
		int i_1=0;
		for(; dim_z+P_NUM-1 < z; dim_z=dim_z+LEAP){//每次跳LEAP层
			mid = 0.0;

			for(int i_0=0;i_0<P_NUM;i_0++){//每次进行3*3*P_NUM的像素块的卷积
				for(int i=0;i<x;i++){
					for(int j=0;j<y;j++){
						mid =mid + train_tmp[dim_x+j + (dim_y+i) * x + (dim_z+i_0)*x*y] * kernel[j + i*x + i_0*x*y + id*x*y*P_NUM];
					}
				}
			}

			re[i_1 + id * re_size] =1/(1 + (1/exp(mid/1000)));//激活函数
			i_1 ++;
		}

	}
}

//GPU端进行下采样
__global__ static void maxpooling(int iter,double * re,double * mre,int re_size,int mre_num){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
       	int id = tid + iter * threadNum; 
	
	//int res = re_size, mres = mre_num;

	if(id < KER_NUM){
		double mid;
		for(int i=0; i<mre_num; i++){
			mid = re[i*GP_NUM + id*re_size];//存放每组第一个值
			for(int j=i*GP_NUM+1; j<(i+1)*GP_NUM; j++){
				if(j >= re_size)
					break;
				if(mid < re[j + id*re_size])
					mid = re[j + id*re_size];
			}
			mre[i + id * mre_num] = mid;
		}
	}
}

//全连接层,每个线程负责一个神经元输出结果的计算
__global__ static void fullconnect(int iter,double * mre,double * omega,double * F1,int mre_size){
	int tid = blockIdx.x * blockDim.x +threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;

	if(id < NEU_NUM1){
		//复制mre数组到共享内存
		//__shared__ double mre_tmp[MAX_MRE_LEN * KER_NUM];
	        extern __shared__ double mre_tmp[];	
		copy_data_to_shared(mre,mre_tmp,mre_size);
		
		//计算神经元的输出
		double mid=0;
		int j; 
		for(int i=id*mre_size; i<(id+1)*mre_size; i++){
			j = 0;
			mid = mid + omega[i] * mre_tmp[j];
			j = j + 1;
		}
		F1[id] = mid;//暂未定义激活函数
	}
}

//输出层，每个线程负责一个神经元输出结果的计算
__global__ static void output(int iter, double * F1, double * omega2, double * O2){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;

	if(id < NEU_NUM2){
		//复制F1到共享内存中
		__shared__ double F1_tmp[NEU_NUM1];
		copy_data_to_shared(F1, F1_tmp, NEU_NUM1);

		//计算神经元的输出
		double mid = 0;
		int j;
		for(int i=id*NEU_NUM1; i<(id+1)*NEU_NUM1; i++){
			j = 0;
			mid = mid + omega2[i] * F1_tmp[j];
			j = j + 1;
		}
		O2[id] = mid;//暂未定义激活函数
	}
}

//反向传播


//训练CNN
int training(double * train, double * labels, int x,int y,int z){
	double * kernel = new double [3*3*P_NUM*KER_NUM];

	//随机生成kernekl数组
	//srand( (unsigned)time( NULL ) );
	for(int i=0; i<9*P_NUM*KER_NUM; i++){
		kernel[i] = 2 * (rand()/(double)(RAND_MAX)) - 1 ;
		//kernel[i] = 0.001;
		if(kernel[i] == 0 || kernel[i] == -1 || kernel[i] == 1)
			kernel[i] = 0.001;
	}
	//fprintf(stdout,"kernel:%lf %lf %lf %lf %lf %lf %lf %lf %lf\n",kernel[0],kernel[1],kernel[2],kernel[3],kernel[4],kernel[5],kernel[6],kernel[7],kernel[8]);

	//计算每次卷积的结果个数
	int re_size = 0;
	for (int i=0; i+P_NUM-1<z; i+=LEAP){
		re_size ++;
	}

	double * re = new double [re_size * KER_NUM];
	/*for(int i=0;i<(x-2)*(y-2)*(z-2);i++){
		re[i] = 0;
	}*/
	fprintf(stdout,"Size of re:%d\n",re_size);
	//fprintf(stdout,"train:%lf %lf\n",train[2 + 2*x + 2*x*y],train[2 + 1*x + 2*x*y]);
	//fprintf(stdout,"kernel:%lf %lf %lf %lf\n",kernel[0],kernel[1],kernel[2],kernel[3]);

	double * gpu_train;
	double * gpu_labels;
	double * gpu_kernel;
	double * gpu_re;//存放卷积结果
	double * gpu_mre;//存放maxpooling结果
	double * gpu_omega1;//第一层网络的输入权重
	double * gpu_F1;//第一层神经元的输出
	double * gpu_omega2;
	double * gpu_O2;

	//复制数据到显存
	SAFE_CALL(cudaMalloc((void**) &gpu_train,sizeof(double) * x * y *z));
	SAFE_CALL(cudaMemcpy(gpu_train,train,sizeof(double) * x * y * z,cudaMemcpyHostToDevice));
	//复制标签
	SAFE_CALL(cudaMalloc((void**) &gpu_labels,sizeof(double) * x * y));
	SAFE_CALL(cudaMemcpy(gpu_labels,labels,sizeof(double) * x * y,cudaMemcpyHostToDevice));
	//复制kernel数组
	SAFE_CALL(cudaMalloc((void**) &gpu_kernel,sizeof(double) * 9 * P_NUM * KER_NUM));
	SAFE_CALL(cudaMemcpy(gpu_kernel,kernel,sizeof(double) * 9 * P_NUM * KER_NUM,cudaMemcpyHostToDevice));
	//卷积结果存入gpu_re，分配显存
	SAFE_CALL(cudaMalloc((void **) &gpu_re,sizeof(double) * re_size * KER_NUM));
//	SAFE_CALL(cudaMemcpy(gpu_re,re,sizeof(double) * (x-2) * (y-2) * (z-2),cudaMemcpyHostToDevice));

	int blocksize = 512;
	int gridsize = 64;
	int threadNum = blocksize * gridsize;

	int mre_num = re_size/GP_NUM + 1;
	if(re_size/GP_NUM == 0){
		mre_num = re_size / GP_NUM;
	}
	fprintf(stdout,"mre_num:%d\n",mre_num);
	int mre_size = mre_num * KER_NUM;
	int ome_num1 = mre_num * KER_NUM * NEU_NUM1;//第一层网络的输入权重个数
	int ome_num2 = NEU_NUM1 * NEU_NUM2;//输出层的权重个数

	double * omega1 = new double [ome_num1];
	double * omega2 = new double [ome_num2];

	//随机生成Omega1
	for(int i=0; i<ome_num1; i++){
		omega1[i] = 2 * (rand()/(double)(RAND_MAX)) - 1 ;
	        if(omega1[i] == 0 || omega1[i] == -1 || omega1[i] == 1)
			omega1[i] = 0.001;
	}

	//随机生成Omega2
	for(int i=0; i<ome_num2; i++){
		omega2[i] = 2 * (rand()/(double)(RAND_MAX)) - 1;
		if(omega2[i] ==0 || omega2[i] == 1 || omega2[i] ==-1)
			omega2[i] = 0.001;
	}

	SAFE_CALL(cudaMalloc((void **) &gpu_mre, sizeof(double) * mre_num * KER_NUM));//maxpooling结果存入gpu_mre，分配显存
        SAFE_CALL(cudaMalloc((void **) &gpu_omega1, sizeof(double) * ome_num1));//第一层网络的输入权重，分配显存
	SAFE_CALL(cudaMalloc((void **) &gpu_omega2, sizeof(double) * ome_num2));//输出层的权重，分配显存
	SAFE_CALL(cudaMalloc((void **) &gpu_F1, sizeof(double) * NEU_NUM1));//第一层网络的输出，分配显存
	SAFE_CALL(cudaMalloc((void **) &gpu_O2, sizeof(double) * NEU_NUM2));//输出层的结果
	
	SAFE_CALL(cudaMemcpy(gpu_omega1, omega1, sizeof(double) * ome_num1, cudaMemcpyHostToDevice));//复制初始权重到GPU端
	SAFE_CALL(cudaMemcpy(gpu_omega2, omega2, sizeof(double) * ome_num2, cudaMemcpyHostToDevice));

	double * mre = new double [mre_num * KER_NUM];//CPU端存放maxpooling结果
	double * F1 = new double [NEU_NUM1];//CPU端存放第一层网络输出结果
	double * O2 = new double [NEU_NUM2];//CPU端存放输出层的结果

	for(int iter=0; iter <= KER_NUM/threadNum; iter++){
		//卷积，每个线程负责一个卷积核和训练数据的卷积
		convol<<<gridsize,blocksize,x*y*z*sizeof(double)>>>(iter,gpu_train,gpu_kernel,gpu_re,x,y,z,re_size);
		
		//下采样，maxpooling方法，每个线程负责re的一列
		maxpooling<<<gridsize,blocksize>>>(iter,gpu_re,gpu_mre,re_size,mre_num);
	}

	for(int iter=0; iter<=NEU_NUM1/threadNum; iter++){
		//全连接层
		fullconnect<<<gridsize,blocksize,mre_size * sizeof(double)>>>(iter,gpu_mre,gpu_omega1,gpu_F1,mre_size);
	}

	for(int iter=0; iter<=NEU_NUM2/threadNum; iter++){
		//输出层
		output<<<gridsize,blocksize>>>(iter,gpu_F1,gpu_omega2,gpu_O2);
	}

	cudaDeviceSynchronize();
	SAFE_CALL(cudaMemcpy(re, gpu_re, sizeof(double) * re_size * KER_NUM, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(mre,gpu_mre,sizeof(double) * mre_num * KER_NUM, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(F1,gpu_F1,sizeof(double) * NEU_NUM1, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(O2,gpu_O2,sizeof(double) * NEU_NUM2, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	fprintf(stdout,"result:%lf %lf %lf\n",re[0],re[98],re[196]);
	fprintf(stdout,"resulr:%lf %lf %lf\n",re[1],re[99],re[197]);
	fprintf(stdout,"mre:%lf %lf %lf\n",mre[0],re[1],re[2]);
	fprintf(stdout,"mre:%lf %lf %lf\n",mre[20],re[21],re[22]);

	fprintf(stdout,"F1 Output:%lf %lf; %lf %lf\n",F1[0],F1[1],F1[98],F1[99]);
	fprintf(stdout,"O2 Output:%lf %lf; %lf %lf\n",O2[0],O2[1],O2[18],O2[19]);

	return 0;
}

int main(int argc, char * argv[])
{
/*	if(!InitCUDA()){
                return 0;
        }
        printf("CUDA initialized.\n");
*/
        clock_t start,end;
       
        double *trainset,*trainlabels;
        if(argc!=2){
                fprintf(stderr, "4 input arguments required!");
        }
        MATFile * datamat = matOpen(argv[1], "r");
        mxArray * train = matGetVariable(datamat,"DataSet");
        mxArray * labels = matGetVariable(datamat,"labels");
		
	trainset = (double*)mxGetData(train);
        trainlabels = (double*)mxGetData(labels);

	const mwSize  * dim;

     	dim = mxGetDimensions(train);//获取trainset每维的元素个数
	//fprintf(stdout,"Dim:%d %d %d\n",dim[0],dim[1],dim[2]);
	//fprintf(stdout,"trainset: %lf %lf %lf... %lf %lf...%lf %lf\n", trainset[0],trainset[1],trainset[2],trainset[145],trainset[146],trainset[21025],trainset[21026]);

	start = clock();
	int tr = training(trainset,trainlabels,dim[0],dim[1],dim[2]);
	end = clock();
	double usetime = double(end - start);
	fprintf(stdout, "Using time of training:%fs \n",usetime/CLOCKS_PER_SEC);

	return 0;
}

