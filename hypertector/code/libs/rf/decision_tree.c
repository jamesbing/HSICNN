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

//define the bool type
typedef enum {false = 0, true = 1} bool;

//some constant variables to construct a decision tree
#define Max_Samples 100000  //the max size of sampls  the M
#define Max_Attributes 300  //the max number of a certain attribute  the N
#define Len 20

//define the structure of a node in decision tree
struct TreeNode{
    int position;
    bool if_leaf_node;
    int class_id;
    struct TreeNode* child[Max_Attributes];
};

/*
 *判断是否所有样本都属于同一类
 *
 */
 bool same_class()
 {

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
    
     //判断D中的所有实例是否属于同一类C，若是，则置T为但节点数，并将C作为该节点的类，返回T
    if (same_class())
     //


     return T;
 }

int main()
{
    printf("Construct decision tree...\n");
}
