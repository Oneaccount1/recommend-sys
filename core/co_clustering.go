package core

import (
	"math"

	"gonum.org/v1/gonum/floats"
)

type CoClustering struct {
	globalMean       float64     // 所有评分的全局平均值
	userMeans        []float64   // 每个用户的平均评分
	itemMeans        []float64   // 每个物品的平均评分
	userClusters     []int       // 用户所属的簇编号
	itemClusters     []int       // 物品所属的簇编号
	userClusterMeans []float64   // 每个用户簇的平均评分
	itemClusterMeans []float64   // 每个物品簇的平均评分
	coClusterMeans   [][]float64 // 用户簇-物品簇的平均评分
	trainSet         TrainSet    // 训练数据集
}

func (c *CoClustering) Predict(userId, itemId int) float64 {
	// Convert to inner Id
	innerUserId := c.trainSet.ConvertUserID(userId)
	innerItemId := c.trainSet.ConvertItemID(itemId)
	prediction := 0.0
	if innerUserId != newID && innerItemId != newID {
		// old user - old item
		userCluster := c.userClusters[innerUserId]
		itemCluster := c.itemClusters[innerItemId]
		prediction = c.userMeans[innerUserId] + c.itemMeans[innerItemId] -
			c.userClusterMeans[userCluster] - c.itemClusterMeans[itemCluster] +
			c.coClusterMeans[userCluster][itemCluster]
	} else if innerItemId == newID {
		// old user - new item
		prediction = c.userMeans[innerUserId]
	} else if innerUserId == newID {
		// new user - old item
		prediction = c.itemMeans[innerItemId]
	} else {
		// new user - new item
		prediction = c.globalMean
	}
	return prediction
}

// Fit a co-clustering model.
// Parameters:
//
//	nEpochs		- The number of iteration of the SGD procedure. Default is 20.
//	nUserClusters	- The number of user clusters.
//	nItemClusters	- The number of item clusters.
func (c *CoClustering) Fit(trainSet TrainSet, parameters Parameters) {
	// Setup parameters
	reader := newParameterReader(parameters)
	nUserClusters := reader.getInt("nUserClusters", 3)
	nItemClusters := reader.getInt("nItemClusters", 3)
	nEpochs := reader.getInt("nEpochs", 20)
	// Initialize parameters
	c.trainSet = trainSet
	c.globalMean = trainSet.GlobalMean()
	// 用户对物品评分的平均值
	c.userMeans = means(trainSet.UserRatings())
	// 物品受到评分的平均值
	c.itemMeans = means(trainSet.ItemRatings())
	//
	c.userClusters = newUniformVectorInt(trainSet.UserCount(), 0, nUserClusters)
	c.itemClusters = newUniformVectorInt(trainSet.ItemCount(), 0, nItemClusters)

	c.userClusterMeans = make([]float64, nUserClusters)
	c.itemClusterMeans = make([]float64, nItemClusters)
	c.coClusterMeans = newZeroMatrix(nUserClusters, nItemClusters)

	// 计算平均值
	clusterMean(c.userClusterMeans, c.userClusters, trainSet.userRatings)
	clusterMean(c.itemClusterMeans, c.itemClusters, trainSet.itemRatings)
	// A^{tmp1}_{ij} = A_{ij} - A^R_i - A^C_j
	userRatings := trainSet.UserRatings()
	tmp1 := newNanMatrix(trainSet.UserCount(), trainSet.ItemCount())
	for i := range tmp1 {
		for j := range tmp1[i] {
			if !math.IsNaN(tmp1[i][j]) {
				tmp1[i][j] = userRatings[i][j] - c.userMeans[i] - c.itemMeans[j]
			}
		}
	}
	// 聚类
	for ep := 0; ep <= nEpochs; ep++ {
		// Compute averages A^{COC}, A^{RC}, A^{CC}, A^R, A^C
		clusterMean(c.userClusterMeans, c.userClusters, trainSet.UserRatings())
		clusterMean(c.itemClusterMeans, c.itemClusters, trainSet.ItemRatings())
		coClusterMean(c.coClusterMeans, c.userClusters, c.itemClusters, userRatings)
		// Update row (user) cluster assignments
		for i := range c.userClusters {
			bestCluster, leastCost := -1, math.Inf(1)
			for k := 0; k < nUserClusters; k++ {
				// \sum^n_{j=1}W_{ij}(A^{tmp1}_{ij}-A^{COC)_{gy(j)}+A^{RC}_g+A^{RC}_{y(j)})^2
				cost := 0.0
				for j := range c.itemClusters {
					if !math.IsNaN(userRatings[i][j]) {
						cost += tmp1[i][j] -
							c.coClusterMeans[k][c.itemClusters[j]] +
							c.userClusterMeans[k] +
							c.itemClusterMeans[c.itemClusters[j]]
					}
				}
				if cost < leastCost {
					bestCluster = k
				}
			}
			c.userClusters[i] = bestCluster
		}
		// 更新列 （item） cluster assignments
		for j := range c.itemClusters {
			bestCluster, leastCost := -1, math.Inf(1)
			for k := 0; k < nItemClusters; k++ {
				// \sum^m_{i=1}W_{ij}(A^{tmp1}_{ij}-A^{COC)_{p(i)h}+A^{RC}_{p(i)}+A^{RC}_h)^2
				cost := 0.0
				for i := range c.userClusters {
					if !math.IsNaN(userRatings[i][j]) {
						cost += tmp1[i][j] -
							// todo num文件提供的函数有误
							c.coClusterMeans[c.itemClusters[i]][k] +
							c.userClusterMeans[c.itemClusters[i]] +
							c.itemClusterMeans[k]
					}
				}
				if cost < leastCost {
					bestCluster = k
				}
			}
			c.itemClusters[j] = bestCluster
		}
	}

}

func NewCoClustering() *CoClustering {
	return new(CoClustering)
}

func clusterMean(dst []float64, clusters []int, ratings [][]float64) {
	resetZeroVector(dst)
	// 记录元素值数量
	count := make([]float64, len(dst))
	for id, cluster := range clusters {
		for _, rating := range ratings[id] {
			if !math.IsNaN(rating) {
				dst[cluster] += rating
				count[cluster]++
			}
		}
	}
	// 对应下标元素相除
	floats.Div(dst, count)
}

func coClusterMean(dst [][]float64, userClusters, itemClusters []int, ratings [][]float64) {
	resetZeroMatrix(dst)
	count := newZeroMatrix(len(dst), len(dst[0]))

	for userID, userCluster := range userClusters {
		for itemID, rating := range ratings[userID] {
			if !math.IsNaN(rating) {
				itemCluster := itemClusters[itemID]
				count[userCluster][itemCluster]++
				dst[userCluster][itemCluster] += rating
			}
		}

	}
	for i := range dst {
		for j := range dst {
			dst[i][j] /= count[i][j]
		}
	}
}
