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

	// 计算初始平均值
	userRatings := trainSet.UserRatings()
	itemRatings := trainSet.ItemRatings()
	clusterMean(c.userClusterMeans, c.userClusters, userRatings)
	clusterMean(c.itemClusterMeans, c.itemClusters, itemRatings)
	coClusterMean(c.coClusterMeans, c.userClusters, c.itemClusters, userRatings)

	// 计算tmp1矩阵: A^{tmp1}_{ij} = A_{ij} - A^R_i - A^C_j
	tmp1 := newNanMatrix(trainSet.UserCount(), trainSet.ItemCount())
	for i := range tmp1 {
		for j := range tmp1[i] {
			if !math.IsNaN(userRatings[i][j]) {
				tmp1[i][j] = userRatings[i][j] - c.userMeans[i] - c.itemMeans[j]
			}
		}
	}

	// 聚类
	for ep := 0; ep < nEpochs; ep++ {
		// 1. 更新用户聚类
		for i := range c.userClusters {
			bestCluster, leastCost := 0, math.Inf(1) // 将初始值从-1改为0，确保至少有一个有效的簇索引
			count := 0
			for k := 0; k < nUserClusters; k++ {
				cost := 0.0
				for j, rating := range userRatings[i] {

					if !math.IsNaN(rating) {
						count++
						// 正确公式: (A^{tmp1}_{ij} - (A^{COC}_{g y(j)} - A^{RC}_g - A^{CC}_{y(j)}))^2
						itemCluster := c.itemClusters[j]
						diff := tmp1[i][j] - (c.coClusterMeans[k][itemCluster] - c.userClusterMeans[k] - c.itemClusterMeans[itemCluster])
						cost += diff * diff
					}
				}
				if cost < leastCost {
					bestCluster = k
					leastCost = cost
				}
			}
			c.userClusters[i] = bestCluster
		}

		// 更新用户簇后重新计算平均值
		clusterMean(c.userClusterMeans, c.userClusters, userRatings)
		coClusterMean(c.coClusterMeans, c.userClusters, c.itemClusters, userRatings)

		// 2. 更新物品聚类（修复了这里的关键错误）
		for j := range c.itemClusters {
			bestCluster, leastCost := 0, math.Inf(1) // 将初始值从-1改为0，确保至少有一个有效的簇索引
			count := 0
			for k := 0; k < nItemClusters; k++ {
				cost := 0.0

				for i := 0; i < len(userRatings); i++ {
					count++
					if !math.IsNaN(userRatings[i][j]) {
						// 正确公式: (A^{tmp1}_{ij} - (A^{COC}_{p(i) h} - A^{RC}_{p(i)} - A^{CC}_h))^2
						userCluster := c.userClusters[i] // 使用用户簇而不是物品簇
						diff := tmp1[i][j] - (c.coClusterMeans[userCluster][k] - c.userClusterMeans[userCluster] - c.itemClusterMeans[k])
						cost += diff * diff
					}
				}
				if cost < leastCost {
					bestCluster = k
					leastCost = cost
				}
			}
			c.itemClusters[j] = bestCluster
		}
		// 更新物品簇后重新计算平均值
		clusterMean(c.itemClusterMeans, c.itemClusters, itemRatings)
		coClusterMean(c.coClusterMeans, c.userClusters, c.itemClusters, userRatings)
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
		for j := range dst[i] {
			if count[i][j] > 0 { // 添加除零保护
				dst[i][j] /= count[i][j]
			}
		}
	}
}
