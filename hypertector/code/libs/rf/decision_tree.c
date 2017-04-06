/*************************************************************************
	> File Name: decision_tree.c
    > Author: tadakey 
	> Mail: tadakey@163.com
	> Created Time: Thu 06 Apr 2017 09:23:48 AM CST
 ************************************************************************/

#include<stdio.h>
#include<malloc.h>
#include<string.h>
#include<math.h>
#include<stdlib.h>
//define the bool type
typedef enum {false = 0, true = 1} bool;

//some constant variables to construct a decision tree
#define Max_Samples 100000  //the max size of sampls  the M
#define Max_Attributes 300  //the max number of a certain attribute  the N
#define Len 20
#define dimension 10

//define the structure of a node in decision tree
//split_attribute:表示该节点用于划分数据的特征
//is_leaf_node:用于说明本节点是否为叶子节点
//class_id:用于记录结点中样本的类别，主要是用在是叶子节点用于分类时
struct TreeNode{
    int split_attribute;
    bool is_leaf_node;
    int class_id;
    struct TreeNode* child[Max_Attributes];
};

//define the structure of a sample
struct sample{
    int class;
    float* data[dimension];
};

/*
 *判断是否所有样本都属于同一类
 *
 */
 bool same_class()
 {

 }

/*
 *取得某个数据集中，属于哪个类别的样本数目最多
 */
 int get_most_class(D)
 {
     int class = 0;


     return class;
 }

 /**
  *用于取得划分样本的最佳特征，采用C4.5算法
  */
 int get_split_attribute(D, A)
 {
     int class_number = 2048;

     return class_number;
 }

 /**
  *计算信息增益比
  */
 float entropy_ratio()
 {
     return 1;
 }

/*
 *构造决策树
 ==================
 Parameters:
 D:训练数据集
 A:特征集
 threashold:阈值
 ==================
 Return:
 TreeNode T:学习到的决策树
 *
 */
 struct TreeNode* BuildTree(int D, int A[], int threshold)
 {
     struct TreeNode *T;
     T = (struct TreeNode *)malloc(sizeof(struct TreeNode));
    
     //判断D中的所有实例是否属于同一类C，若是，则置T为单节点树，并将C作为该节点的类，返回T
    if (same_class())
     {
         T->is_leaf_node = true;
         //TODO：这儿应该是将类别取出赋给class_id，但是由于数据格式不确定，暂时先空着
         int temp_class = 10;
         T->class_id = temp_class; 
         return T;
         
     }
    //判断特征集是否为空，若特征集为空，则将T置为单节点树，并将D中包含样本数量最多的类C作为该节点的类，返回T
    if (sizeof(A) == 0)
     {
         T->is_leaf_node = true;
         T->class_id = get_most_class(D);
         return T;
     }
    //上述两条判断都不成立，则进行数据集的分裂
    //挑选数据集中用于划分节点的属性
    int chosen_attribute = get_split_attribute(D, A);
    float information_gain_ratio = entropy_ratio(chosen_attribute);
    if (information_gain_ratio < threshold)
     {
        T->is_leaf_node = true;
        T->class_id = get_most_class(D);
        return T;
     }else{
         printf("重新划分数据....\n");
        
     }
     
     return T;
 }

int main()
{
    //读取输入训练数据集
//    float data1[2][1] = [[2.0],[0.21,0.33,0.87,0.77,0.66]];
    
    
    printf("Construct decision tree...\n");
}
