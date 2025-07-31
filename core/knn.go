package core

import (
	"math"
	"runtime"
	"sort"
	"sync"
)

const (
	basic    = "basic"
	centered = "centered"
	zScore   = "zscore"
	baseline = "baseline"
)

type KNN struct {
	Base
	KNNType      string
	GlobalMean   float64
	Sims         [][]float64
	LeftRatings  [][]IDRating
	RightRatings [][]IDRating
	Means        []float64 // Centered KNN :user(item) Means
	StdDevs      []float64 // KNN with Z Score: user (item) standard deviation
	Bias         []float64 // KNN BaseLine :Bias
}
type CandidateSet struct {
	similarities []float64
	candidates   []IDRating // 改为 candidates
}

func NewCandidateSet(sim []float64, candidates []IDRating) *CandidateSet {
	neighbors := &CandidateSet{}
	neighbors.similarities = sim
	neighbors.candidates = candidates
	return neighbors
}
func (n *CandidateSet) Len() int {
	return len(n.candidates)
}

func (n *CandidateSet) Less(i, j int) bool {
	return n.similarities[n.candidates[i].ID] > n.similarities[n.candidates[j].ID]
}
func (n *CandidateSet) Swap(i, j int) {
	n.candidates[i], n.candidates[j] = n.candidates[j], n.candidates[i]
}

func NewKNN(params Parameters) *KNN {
	knn := new(KNN)
	knn.KNNType = knn.Params.GetString("type", basic)
	knn.Params = params
	return knn
}
func NewKNNWithMean(params Parameters) *KNN {
	knn := new(KNN)
	knn.KNNType = knn.Params.GetString("type", centered)
	knn.Params = params
	return knn
}
func NewKNNWithZScore(params Parameters) *KNN {
	knn := new(KNN)
	knn.KNNType = knn.Params.GetString("type", zScore)
	knn.Params = params
	return knn
}
func NewKNNBaseLine(params Parameters) *KNN {
	knn := new(KNN)
	knn.KNNType = knn.Params.GetString("type", baseline)
	knn.Params = params
	return knn
}

func (K *KNN) Predict(userID int, itemID int) float64 {
	innerUserID := K.Data.ConvertUserID(userID)
	innerItemID := K.Data.ConvertItemID(itemID)
	// 获取参数
	userBased := K.Params.GetBool("userBased", true)
	k := K.Params.GetInt("k", 40)
	minK := K.Params.GetInt("mink", 1)
	// 基于用户 or 物品 ？
	var leftID, rightID int
	if userBased {
		leftID, rightID = innerUserID, innerItemID
	} else {
		leftID, rightID = innerItemID, innerUserID
	}
	if leftID == newID || rightID == newID {
		return K.GlobalMean
	}
	// 获取用户（物品）有交互的 物品（用户）
	candidates := make([]IDRating, 0)

	for _, ir := range K.RightRatings[rightID] {
		if !math.IsNaN(K.Sims[leftID][ir.ID]) {
			candidates = append(candidates, ir)
		}
	}

	// 如果用户（物品） 的数量小于最小值 k。 则使用全局平均直作为预测结果
	if len(candidates) <= minK {
		return K.GlobalMean
	}

	// 排序 通过相似度排序
	candidateSet := NewCandidateSet(K.Sims[leftID], candidates)
	sort.Sort(candidateSet)

	// 控制User邻居数量
	numNeighbors := k
	if numNeighbors > candidateSet.Len() {
		numNeighbors = candidateSet.Len()
	}
	// 预测分数 根据带权平均值
	weightSum := 0.0
	weightRating := 0.0
	for _, or := range candidateSet.candidates[0:numNeighbors] {
		weightSum += K.Sims[leftID][or.ID]
		// （以基于用户的角度）用户与候选人相似度 * 候选人对该物品的评分
		rating := or.Rating
		if K.KNNType == centered {
			rating -= K.Means[or.ID]
		} else if K.KNNType == zScore {
			rating = (rating - K.Means[or.ID]) / K.StdDevs[or.ID]
		} else if K.KNNType == baseline {
			rating -= K.Bias[or.ID]
		}
		weightRating += K.Sims[leftID][or.ID] * rating
	}
	prediction := weightRating / weightSum
	if K.KNNType == centered {
		prediction += K.Means[leftID]
	} else if K.KNNType == baseline {
		prediction += K.Bias[leftID]
	} else if K.KNNType == zScore {
		prediction *= K.StdDevs[leftID]
		prediction += K.Means[leftID]
	}
	return prediction
}

func (K *KNN) Fit(trainSet TrainSet) {
	// Setup parameters
	sim := K.Params.GetSim("sim", MSD)
	userBased := K.Params.GetBool("userBased", true)
	//  nJobs
	nJobs := K.Params.GetInt("nJobs", runtime.NumCPU())
	//nJobs := runtime.NumCPU()
	K.Data = trainSet
	// 设置全局平均值为新的用户（物品）
	K.GlobalMean = trainSet.GlobalMean
	// 获取用户（物品） 评分
	if userBased {
		K.LeftRatings = trainSet.UserRatings()
		K.RightRatings = trainSet.ItemRatings()
		K.Sims = newNanMatrix(trainSet.UserCount, trainSet.UserCount)
	} else {
		K.LeftRatings = trainSet.ItemRatings()
		K.RightRatings = trainSet.UserRatings()
		K.Sims = newNanMatrix(trainSet.ItemCount, trainSet.ItemCount)
	}
	// 获取 user（item）的平均值
	if K.KNNType == centered || K.KNNType == zScore {
		K.Means = means(K.LeftRatings)
	}
	if K.KNNType == zScore {
		K.StdDevs = make([]float64, len(K.LeftRatings))
		for i := range K.Means {
			sum, count := 0.0, 0.0
			for _, ir := range K.LeftRatings[i] {
				sum += (ir.Rating - K.Means[i]) * (ir.Rating - K.Means[i])
				count++
			}
			K.StdDevs[i] = math.Sqrt(sum/count) + 1e-5
		}
	}

	if K.KNNType == baseline {
		baseLine := NewBaseLine(K.Params)
		baseLine.Fit(trainSet)
		if userBased {
			K.Bias = baseLine.userBias
		} else {
			K.Bias = baseLine.itemBias
		}
	}
	// 计算全部矩阵 比 只计算上三角矩阵块约0.2s， 利用缓存友好性
	// 计算用户的两两相似性
	sortedLeftRatings := sorts(K.LeftRatings)
	length := len(sortedLeftRatings)
	var wg sync.WaitGroup
	wg.Add(nJobs)
	for j := 0; j < nJobs; j++ {
		go func(jobID int) {
			begin := length * jobID / nJobs
			end := length * (jobID + 1) / nJobs

			for iID := begin; iID < end; iID++ {
				iRatings := sortedLeftRatings[iID]
				for jID, jRatings := range sortedLeftRatings {
					if iID != jID {
						if math.IsNaN(K.Sims[iID][jID]) {
							ret := sim(iRatings, jRatings)
							if !math.IsNaN(ret) {
								K.Sims[iID][jID] = ret
								K.Sims[jID][iID] = ret
							}
						}
					}
				}
			}
			wg.Done()
		}(j)
	}
	wg.Wait()
}
