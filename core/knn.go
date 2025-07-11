package core

import (
	"math"
	"sort"
)

type KNN struct {
	option  Option
	mean    float64
	sims    map[int]map[int]float64
	ratings map[int]map[int]float64
}

func (K *KNN) Predict(userId int, itemId int) float64 {
	// 设置基于用户或者基于物品的预测
	var leftID, rightID int
	if K.option.userBased {
		leftID = userId
		rightID = itemId
	} else {
		leftID = itemId
		rightID = userId
	}
	// 获取用户（物品）有交互的 物品（用户）
	candidates := make([]int, 0)

	for otherID := range K.ratings {
		if _, exist := K.ratings[otherID][rightID]; exist && !math.IsNaN(K.sims[leftID][otherID]) {
			candidates = append(candidates, otherID)
		}
	}

	// 如果用户（物品） 的数量小于最小值 k。 则使用全局平均直作为预测结果
	if len(candidates) < K.option.minK {
		return K.mean
	}

	// 排序 通过相似度排序
	candidateSet := NewCandidateSet(K.sims[leftID], candidates)
	sort.Sort(candidateSet)

	// 获取邻居数量
	numNeighbors := K.option.k
	if numNeighbors > candidateSet.Len() {
		numNeighbors = candidateSet.Len()
	}
	// 预测分数 根据带权平均值
	weightSum := 0.0
	weightRating := 0.0
	for _, otherID := range candidateSet.candidate[0:numNeighbors] {
		weightSum += K.sims[leftID][otherID]
		// （以基于用户的角度）用户与候选人相似度 * 候选人对该物品的评分
		weightRating += K.sims[leftID][otherID] * K.ratings[otherID][itemId]
	}
	return weightRating / weightSum

}

func (K *KNN) Fit(trainSet TrainSet, options ...OptionSetter) {
	// 设置选项
	K.option = Option{
		sim:       MSD,
		userBased: true,
		k:         40, // the (max) number of neighbors to take into account for aggregation
		minK:      1,  // The minimum number of neighbors to take into account for aggregation.
		// If there are not enough neighbors, the prediction is set the global
		// mean of all interactionRatings
	}
	for _, setter := range options {
		setter(&K.option)
	}
	// 设置全局平均值为新的用户（物品）
	K.mean = trainSet.GlobalMean()
	// 获取用户（物品） 评分
	if K.option.userBased {
		K.ratings = trainSet.userRatings
	} else {
		K.ratings = trainSet.itemRatings
	}
	// 两两相似度
	K.sims = make(map[int]map[int]float64)

	for leftID, leftRatings := range K.ratings {
		for rightID, rightRatings := range K.ratings {
			if leftID != rightID {

				if _, exist := K.sims[leftID]; !exist {
					K.sims[leftID] = make(map[int]float64)
				}
				if _, exist := K.sims[rightID]; !exist {
					K.sims[rightID] = make(map[int]float64)
				}
				// 设置相似度值
				if _, exist := K.sims[leftID][rightID]; !exist {
					ret := K.option.sim(leftRatings, rightRatings)
					if !math.IsNaN(ret) {
						K.sims[leftID][rightID] = ret
						K.sims[rightID][leftID] = ret
					}
				}
			}
		}
	}

}

type CandidateSet struct {
	similarities map[int]float64
	candidate    []int
}

func NewCandidateSet(sim map[int]float64, candidates []int) *CandidateSet {
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
	return new(KNN)
}
