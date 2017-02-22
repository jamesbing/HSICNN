#include <mat.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <matrix.h>
#include <iostream>
#include "cublas_v2.h"
#include "cokus.cpp"
#include "cuda_util.h"
#include <cuda_runtime.h>
using namespace std;

const int KER_NUM = 20;//卷积核数量
const int P_NUM = 8;//每次卷积的层数
const int LEAP = 2;//跳数
const int GP_NUM = 2;//maxpooling每组的个数
const int NEU_NUM1 = 100;
const int NEU_NUM2 = 13;//输出层神经元个数
const int NEIGHBOR = 8;//定义邻居个数
double LEARN_RATE = 0.008;
const double MIN_ERR = 0.001;
const int VALID_BATCH = 10;

//copy数据到shared memory
__device__ void copy_data_to_shared(double * data, double * data_tmp,int head, int length){
	for(int i=0; i<length; i++){
		data_tmp[i] = data[i+head];
	}

	__syncthreads();
}

//GPU端负责卷积
__global__ static void convol(int iter,int i0,double * train,double * kernel,double * re,double * bias,int x,int y,int z,int re_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;//保存当前线程编号

	//每个线程负责一个卷积核与一个3*3*hight柱状图像的卷积
	if (id < KER_NUM){
		extern __shared__ double train_tmp[];
		//__shared__ double train_tmp[9*200];
		int st = i0 * x * y * z;

		copy_data_to_shared(train,train_tmp,st,x*y*z);//复制train到shared memory中

		/*double * ker = new double [x*y*P_NUM];//载入对应的kernel到寄存器
		for(int i=0; i<x*y*P_NUM; i++){
			ker[i] = kernel[id*x*y*P_NUM + i];
		}*/
		double mid;
		//int i_1=0;
		for(int i=0; i<re_size; i++){
			mid = 0;
			int start = i*x*y*LEAP;//训练数据每次卷积的起点
			for(int j=0; j<x*y*P_NUM; j++){
				mid = mid + train_tmp[start + j]*kernel[id*x*y*P_NUM+j];
			}
			mid = mid + bias[id];
			re[i + id*re_size] = 2/(1+(1/exp(2*mid))) - 1;//激活函数tanh
		}
		/*for
		}*/
	}
}

//GPU端进行下采样
__global__ static void maxpooling(int iter,double * re,double * mre,int * mre_index,int re_size,int mre_num){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
       	int id = tid + iter * threadNum; 
	
	//int res = re_size, mres = mre_num;
	//extern __shared__ double re_tmp[];
	//copy_data_to_shared(re, re_tmp, 0, re_size*KER_NUM);

	if(id < KER_NUM){
		double mid;
		int mid_index;
		for(int i=0; i<mre_num; i++){
			mid = re[i*GP_NUM + id*re_size];//存放每组第一个值
			mid_index = i*GP_NUM + id*re_size;
			for(int j=i*GP_NUM+1; j<(i+1)*GP_NUM && j<re_size; j++){
				if(mid < re[j + id*re_size]){
					mid = re[j + id*re_size];
					mid_index = j+id*re_size;
				}
			}
			mre[i + id * mre_num] = mid;
			mre_index[i + id * mre_num] = mid_index;
		}
	}
}

//全连接层,每个线程负责一个神经元输出结果的计算
__global__ static void fullconnect(int iter,double * mre,double * omega,double * bias,double * F1,int mre_size){
	int tid = blockIdx.x * blockDim.x +threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;

	if(id < NEU_NUM1){
		//复制mre数组到共享内存
		//__shared__ double mre_tmp[50 * KER_NUM];
	        extern __shared__ double mre_tmp[];	
		copy_data_to_shared(mre,mre_tmp,0,mre_size);
		
		//计算神经元的输出
		double mid=0;
		for(int i=0; i<mre_size; i++){
			mid = mid + omega[id + i*NEU_NUM1] * mre_tmp[i];
		}
		mid = mid + bias[id];
		F1[id] = 2/(1 + 1/exp(mid * 2)) - 1;//激活函数tanh
	}
}

//输出层，每个线程负责一个神经元输出结果的计算
__global__ static void output(int iter, double * F1, double * omega2, double * bias, double * O2){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;

	if(id < NEU_NUM2){
		//复制F1到共享内存中
		__shared__ double F1_tmp[NEU_NUM1];
		copy_data_to_shared(F1, F1_tmp, 0, NEU_NUM1);
		__shared__ double O2_tmp[NEU_NUM2];

		//计算神经元的输出
		double mid = 0;
		for(int i=0; i<NEU_NUM1; i++){
			mid = mid + omega2[id + i*NEU_NUM2] * F1_tmp[i];
		}
		O2[id] = exp(mid+ bias[id]);
		O2_tmp[id] = O2[id];
		__syncthreads(); //等待所有线程将神经元输出结果加载入SM

		//计算softmax激活函数的输出结果
		int length = NEU_NUM2;//当前需要累加的数组长度
		int offset = (length - 1)/2 +1;//累加的偏移值
		while(length >= 2)
		{
			if(id + offset < length){
				O2_tmp[id] = O2_tmp[id] + O2_tmp[id + offset];
			}
			offset = (offset - 1)/2 + 1;
			length = (length - 1)/2 + 1;
			__syncthreads();//等待所有线程完成当前的累加
		}
		O2[id] = O2[id]/O2_tmp[0];

	}
}

//计算正确率
double count_err(double * test_labels, double * output, int test_idx)
{
	double right=0;
	double max =0;
	int idx = 0;
	for(int i=0; i<NEU_NUM2; i++){
		if(output[i]>max){
			max = output[i];
			idx = i;
		}
	}
	if((idx+1) == int(test_labels[test_idx]))
		right = 1;
	
	return right;
}

double testint(int test_size, int data_size, double * test_data, double * test_labels, double * kernel, double * omega1, double * omega2, double * bias0, double * bias1, double * bias2)
{
		double * gpu_processed_test;
		double * gpu_kernel;
		double * gpu_omega1;
		double * gpu_omega2;
		double * gpu_bias0;
		double * gpu_bias1;
		double * gpu_bias2;
		double * gpu_re;
		double * gpu_mre;
		double * gpu_mre_index;
		double * gpu_F1;
		double * gpu_O2;
		
			//计算每次卷积的结果个数
		int re_size = 0;
		for (int i=0; i+P_NUM-1<z; i+=LEAP){
			re_size ++;
		}
		int mre_num = (re_size-1)/GP_NUM + 1;
		int mre_size = mre_num * KER_NUM;
		int ome_num1 = mre_num * KER_NUM * NEU_NUM1;//第一层网络的输入权重个数
		int ome_num2 = NEU_NUM1 * NEU_NUM2;//输出层的权重个数	
		
		SAFE_CALL(cudaMalloc((void **) &gpu_processed_test, sizeof(double) * data_size));
		SAFE_CALL(cudaMalloc((void **) &gpu_kernel,sizeof(double) * (NEIGHBOR+1) * P_NUM * KER_NUM));
		SAFE_CALL(cudaMalloc((void **) &gpu_omega1, sizeof(double) * ome_num1));//第一层网络的输入权重，分配显存
		SAFE_CALL(cudaMalloc((void **) &gpu_omega2, sizeof(double) * ome_num2));//输出层的权重，分配显存
		SAFE_CALL(cudaMalloc((void **) &gpu_bias0, sizeof(double) * KER_NUM));//卷积层偏置值
		SAFE_CALL(cudaMalloc((void **) &gpu_bias1, sizeof(double) * NEU_NUM1));//全连接层偏置值
		SAFE_CALL(cudaMalloc((void **) &gpu_bias2, sizeof(double) * NEU_NUM2));//输出层偏置
		SAFE_CALL(cudaMalloc((void **) &gpu_re,sizeof(double) * re_size * KER_NUM));
		SAFE_CALL(cudaMalloc((void **) &gpu_mre, sizeof(double) * mre_num * KER_NUM));//maxpooling结果存入gpu_mre，分配显存
		SAFE_CALL(cudaMalloc((void **) &gpu_mre_index, sizeof(int) * mre_num * KER_NUM));//为maxpooling的最大值索引分配显存
		SAFE_CALL(cudaMalloc((void **) &gpu_F1, sizeof(double) * NEU_NUM1));//第一层网络的输出，分配显存
		SAFE_CALL(cudaMalloc((void **) &gpu_O2, sizeof(double) * NEU_NUM2));//输出层的结果
		
		SAFE_CALL(cudaMemcpy(gpu_processed_test,test_data,sizeof(double) * (NEIGHBOR+1) * data_size, cudaMemcpyHostToDevice));
		SAFE_CALL(cudaMemcpy(gpu_kernel,kernel,sizeof(double) * (NEIGHBOR+1) * P_NUM * KER_NUM,cudaMemcpyHostToDevice));
		SAFE_CALL(cudaMemcpy(gpu_omega1, omega1, sizeof(double) * ome_num1, cudaMemcpyHostToDevice));//复制初始权重到GPU端
		SAFE_CALL(cudaMemcpy(gpu_omega2, omega2, sizeof(double) * ome_num2, cudaMemcpyHostToDevice));
		SAFE_CALL(cudaMemcpy(gpu_bias0, bias0, sizeof(double) * KER_NUM, cudaMemcpyHostToDevice));
		SAFE_CALL(cudaMemcpy(gpu_bias1, bias1, sizeof(double) * NEU_NUM1, cudaMemcpyHostToDevice));//复制偏置值到显存
		SAFE_CALL(cudaMemcpy(gpu_bias2, bias2, sizeof(double) * NEU_NUM2, cudaMemcpyHostToDevice));
		
		double right = 0;
		double count0 = 0;
		for (int i1=0; i1<test_size; i1++){
			int iter = 0;
			convol<<<1,KER_NUM,(NEIGHBOR+1)*z*sizeof(double)>>>(iter,i1,gpu_processed_test,gpu_kernel,gpu_re,gpu_bias0,3,3,z,re_size);
			cudaDeviceSynchronize();

			maxpooling<<<1,KER_NUM>>>(iter,gpu_re,gpu_mre,gpu_mre_index,re_size,mre_num);
			cudaDeviceSynchronize();

			fullconnect<<<1,NEU_NUM1,mre_size * sizeof(double)>>>(iter,gpu_mre,gpu_omega1,gpu_bias1,gpu_F1,mre_size);
			cudaDeviceSynchronize();

			output<<<1,NEU_NUM2>>>(iter,gpu_F1,gpu_omega2,gpu_bias2,gpu_O2);
			cudaDeviceSynchronize();

			SAFE_CALL(cudaMemcpy(O2, gpu_O2, sizeof(double) * NEU_NUM2, cudaMemcpyDeviceToHost));
			cudaDeviceSynchronize();

			//fprintf(stdout,"\n");
			right = count_err(test_labels, O2, i1);
			count0 = count0 + right;
		}
		
		return count0/test_size;
}
int main(int argc, char * argv[])
{
	clock_t start,end;

	double * kernel,* omega1, * omega2, * bias0, * bias1, * bias2;
	if(argc!=3){
		fprintf(stderr, "3 input arguments required!");
	}
	MATFile * datamat = matOpen(argv[1], "r");
	mxArray * ker = matGetVariable(datamat,"kernel");
	mxArray * ome1 = matGetVariable(datamat,"omega1");
	mxArray * ome2 = matGetVariable(datamat,"omega2");
	mxArray * b0 = matGetVariable(datamat,"bias0");
	mxArray * b1 = matGetVariable(datamat,"bias1");
	mxArray * b2 = matGetVariable(datamat,"bias2");

	kernel = (double*)mxGetData(ker);
	omega1 = (double*)mxGetData(ome1);
	omega2 = (double*)mxGetData(ome2);
	bias0 = (double*)mxGetData(b0);
	bias1 = (double*)mxGetData(b1);
	bias2 = (double*)mxGetData(b2);
	matClose(datamat);
	
	double * test_data, * test_labels;
	MATFile * testmat = matOpen(argv[2], "r");
	mxArray * data = matGetVariable(testmat,"data");
	mxArray * labels = matGetVariable(testmat,"labels");
	
	test_data = (double*)mxGetData(data);
	test_labels	= (double*)mxGetData(labels);
	const mwSize  * dim0, *dim1;
	dim0 = mxGetDimensions(labels);//获取测试集个数
	dim1 = mxGetDimensions(data);//获取测试集规模
	matClose(testmat);

	double corr = testing(dim0[0],dim1[0] * dim1[1] * dim1[2],test_data,test_labels,kernel,omega1,omega2,bias0,bias1,bias2);	
}