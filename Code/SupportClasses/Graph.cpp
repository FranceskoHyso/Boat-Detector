#include <iostream>
#include "Graph.h"

Graph::Graph(int numOfNodes, std::vector<std::vector<int>>& edges) {
	mNumOfNodes = numOfNodes;
	mEdges = edges;
}

int Graph::getNumOfNodes(){
	return mEdges.size();
}

void Graph::DFS(int ithNode, std::vector<int> ithNodeEdges, std::vector<bool>& visited, std::vector<int>& currConnComp){
	//std::cout<<"Inside the DFS() function: ithNode = "<<ithNode<<std::endl;
	visited[ithNode] = true;
	currConnComp.push_back(ithNode);
	for (int i = 0; i < ithNodeEdges.size(); i++) {
		if (!visited[ithNodeEdges[i]])
			DFS(ithNodeEdges[i], mEdges[ithNodeEdges[i]], visited, currConnComp);
	}
}

void Graph::getConnectedComponents(std::vector<std::vector<int>>& connComp, int& actualNumOfConnComp){
	//std::cout<<"Computing the connected components of the graph..."<<std::endl;
	std::vector<bool> visited(mNumOfNodes);
	for (int i = 0; i < visited.size(); i++) {
		visited[i] = false;
	}
	for (int i = 0; i < visited.size(); i++) {
		//std::cout<<"Working to the connected component "<<i<<":"<<std::endl;
		if (!visited[i]) {
			DFS(i, mEdges[i], visited, connComp[i]);
		}
	}
	for (int i = 0; i < connComp.size(); i++) {
		if (!connComp[i].empty()) {
			actualNumOfConnComp++;
		}
	}
	int k = 0;
	std::vector<std::vector<int>> tmpConnComp(actualNumOfConnComp);
	for (int i = 0; i < connComp.size(); i++) {
		if (!connComp[i].empty()) {
			tmpConnComp[k] = connComp[i];
			k++;
		}
	}
	connComp = tmpConnComp;
	//std::cout<<"Connected components computed."<<std::endl;
}

void Graph::printConnComp(std::vector<std::vector<int>>& connComp, int actualNumOfConnComp){
	std::cout<<"Max number of connected componets: "<<connComp.size()<<std::endl;
	std::cout<<"Actual number of connected componets: "<<actualNumOfConnComp<<std::endl;
	for (int i = 0; i < connComp.size(); i++) {
		if (!connComp[i].empty()){
			//std::cout<<"Nodes of connected components number "<<i<<std::endl;
			for (int j = 0; j < connComp[i].size(); j++) {
				std::cout<<"    "<<connComp[i][j]<<std::endl;
			}
		}
	}
}