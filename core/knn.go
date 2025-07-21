package core

import (
	"math"
	"sort"
)

const (
	basic    = 0
	centered = 1
	zscore   = 2
	baseline = 3
)

type KNN struct {
	option     parameterReader
	tpe        int
	globalMean float64
	sims       [][]float64
	ratings    [][]float64
	means      []float64 // Centered KNN :user(item) means
	bias       []float64 // KNN BaseLine :bias
	trainSet   TrainSet

	// 参数
	userBased bool
	k         int
	minK      int
}
type CandidateSet struct {
	similarities []float64
	candidate    []int
}

func NewCandidateSet(sim []float64, candidates []int) *CandidateSet {
	neighbors := &CandidateSet{}
	neighbors.similarities = sim
	neighbors.candidate = candidates
	return neighbors
}
func (n *CandidateSet) Len() int {
	return len(n.candidate)
}

func (n *CandidateSet) Less(i, j int) bool {
	return n.similarities[n.candidate[i]] > n.similarities[n.candidate[j]]
}
func (n *CandidateSet) Swap(i, j int) {
	n.candidate[i], n.candidate[j] = n.candidate[j], n.candidate[i]
}

func NewKNN() *KNN {
	knn := new(KNN)
	knn.tpe = basic
	return knn
}
func NewKNNWithMean() *KNN {
	knn := new(KNN)
	knn.tpe = baseline
	return knn
}
func NewKNNBaseLine() *KNN {
	knn := new(KNN)
	knn.tpe = centered
	return knn
}

func (K *KNN) Predict(userID int, itemID int) float64 {
	innerUserID := K.trainSet.ConvertUserID(userID)
	innerItemID := K.trainSet.ConvertItemID(itemID)
	// 设置基于用户或者基于物品
	var leftID, rightID int
	if K.userBased {
		leftID, rightID = innerUserID, innerItemID
	} else {
		leftID, rightID = innerItemID, innerUserID
	}
	if leftID == newID || rightID == newID {
		return K.globalMean
	}
	// 获取用户（物品）有交互的 物品（用户）
	candidates := make([]int, 0)

	for otherID := range K.ratings {
		if !math.IsNaN(K.ratings[otherID][rightID]) && !math.IsNaN(K.sims[leftID][otherID]) {
			candidates = append(candidates, otherID)
		}
	}

	// 如果用户（物品） 的数量小于最小值 k。 则使用全局平均直作为预测结果
	if len(candidates) <= K.minK {
		return K.globalMean
	}

	// 排序 通过相似度排序
	candidateSet := NewCandidateSet(K.sims[leftID], candidates)
	sort.Sort(candidateSet)

	// 控制User邻居数量
	numNeighbors := K.k
	if numNeighbors > candidateSet.Len() {
		numNeighbors = candidateSet.Len()
	}
	// 预测分数 根据带权平均值
	weightSum := 0.0
	weightRating := 0.0
	for _, otherID := range candidateSet.candidate[0:numNeighbors] {
		weightSum += K.sims[leftID][otherID]
		// （以基于用户的角度）用户与候选人相似度 * 候选人对该物品的评分
		weightRating += K.sims[leftID][otherID] * K.ratings[otherID][rightID]

		if K.tpe == centered {
			weightRating -= K.sims[leftID][otherID] * K.means[otherID]
		} else if K.tpe == baseline {
			weightRating -= K.sims[leftID][otherID] * K.bias[otherID]
		}
	}
	prediction := weightRating / weightSum
	if K.tpe == centered {
		prediction += K.means[leftID]
	} else if K.tpe == baseline {
		prediction += K.bias[leftID]
	}
	return weightRating / weightSum

}

func (K *KNN) Fit(trainSet TrainSet, options Parameters) {
	// Setup options
	reader := newParameterReader(options)
	sim := reader.getSim("sim", MSD)
	K.userBased = reader.getBool("userBased", true)
	K.k = reader.getInt("k", 40)
	K.minK = reader.getInt("minK", 1)

	K.trainSet = trainSet
	// 设置全局平均值为新的用户（物品）
	K.globalMean = trainSet.GlobalMean()
	// 获取用户（物品） 评分
	if K.userBased {
		K.ratings = trainSet.UserRatings()
		K.sims = newNanMatrix(trainSet.userCount, trainSet.userCount)
	} else {
		K.ratings = trainSet.ItemRatings()
		K.sims = newNanMatrix(K.trainSet.itemCount, trainSet.itemCount)
	}
	// 获取 user（item）的平均值
	if K.tpe == centered {
		K.means = make([]float64, len(K.ratings))
		for i := range K.means {
			sum, count := 0.0, 0.0
			for j := range K.ratings[i] {
				if !math.IsNaN(K.ratings[i][j]) {
					sum += K.ratings[i][j]
					count++
				}
			}
			K.means[i] = sum / count
		}
	} else if K.tpe == baseline {
		baseLine := NewBaseLine()
		baseLine.Fit(trainSet, options)
		if K.userBased {
			K.bias = baseLine.userBias
		} else {
			K.bias = baseLine.itemBias
		}
	}
	// 两两相似度矩阵
	for leftID, leftRatings := range K.ratings {
		for rightID, rightRatings := range K.ratings {
			if leftID != rightID {
				if math.IsNaN(K.sims[leftID][rightID]) {
					ret := sim(leftRatings, rightRatings)
					if !math.IsNaN(ret) {
						K.sims[leftID][rightID] = ret
						K.sims[rightID][leftID] = ret
					}
				}
			}
		}
	}

}
