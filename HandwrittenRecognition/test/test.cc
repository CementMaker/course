#include <cstdio>
#include <iostream>

using namespace std;

struct node {
	int data;
};

int main()
{
	node a;
	a.data = 1;
	cout << a.data << endl;
	cout << a -> data << endl;
	return 0;
}