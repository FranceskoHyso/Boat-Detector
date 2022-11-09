#include <list>
#include <iterator>
#include <highgui.hpp>
#include <imgcodecs.hpp>

class Graph{
public:
	Graph(int numOfNodes, std::vector<std::vector<int>>& edges);
	int getNumOfNodes();
	void getConnectedComponents(std::vector<std::vector<int>>& connComp, int& actualNumOfConnComp);
	void printConnComp(std::vector<std::vector<int>>& connComp, int actualNumOfConnComp);
private:
	int mNumOfNodes;
	std::vector<std::vector<int>> mEdges;
	void DFS(int i, std::vector<int> ithNodeEdges, std::vector<bool>& visited, std::vector<int>& currConnComp);
};
