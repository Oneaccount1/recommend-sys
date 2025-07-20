package core

import "math"

type SlopeOne struct {
	globalMean  float64
	userRatings [][]float64
	userMean    []float64
	dev         [][]float64 // average difference between the ratings of i and those of j  存储两两物品间的平均评分差值
	trainSet    TrainSet
}

func (s *SlopeOne) Predict(userId int, itemId int) float64 {
	innerUserID := s.trainSet.ConvertUserID(userId)
	innerItemID := s.trainSet.ConvertItemID(itemId)
	prediction := 0.0
	if innerUserID != newID {
		prediction = s.userMean[innerUserID]
	} else {
		prediction = s.globalMean
	}

	if innerItemID != newID {
		sum, count := 0.0, 0.0
		for j := range s.userRatings[innerUserID] {
			if !math.IsNaN(s.userRatings[innerUserID][j]) {
				sum += s.dev[innerItemID][j]
				count++
			}
		}
		if count > 0 {
			prediction += sum / count
		}
	}

	return prediction

}

func (s *SlopeOne) Fit(trainSet TrainSet, options Options) {
	s.trainSet = trainSet
	s.globalMean = trainSet.GlobalMean()
	s.userRatings = trainSet.UserRatings()
	s.userMean = mean(s.userRatings)
	s.dev = newZeroMatrix(trainSet.ItemCount(), trainSet.ItemCount())
	ratings := trainSet.itemRatings

	// 计算两两物品间的差异值, 先遍历两个物品
	for i := 0; i < len(ratings); i++ {
		for j := 0; j < i; j++ {
			count, sum := 0.0, 0.0
			// 继续遍历对这两个物品都有评分的用户集合
			// 行为物品，列为用户
			for k := 0; k < len(ratings[i]); k++ {
				if !math.IsNaN(ratings[i][k]) && math.IsNaN(ratings[j][k]) {
					sum += ratings[i][k] - ratings[j][k]
					count++
				}
			}
			// 如果交集不为空
			if count > 0 {
				s.dev[i][j] = sum / count
				s.dev[j][i] = -s.dev[i][j]
			}
		}
	}

}

func NewSlopeOne() *SlopeOne {
	return new(SlopeOne)
}
