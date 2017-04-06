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
#define Max_Samples 100000  //the max size of sampls
#define Max_Attributes 300  //the max number of a certain attribute
#define Len 20

//define the structure of a node in decision tree
struct TreeNode{
    int position;
    bool if_leaf_node;
    int class_id;
    struct TreeNode* child[Max_Attributes];
};

/*
 *构造决策树
 *
 */
TreeNode* ConstructTree()

int main(){
    printf("Construct decision tree...\n");
}
