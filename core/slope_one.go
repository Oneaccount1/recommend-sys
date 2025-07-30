package core

import (
	"runtime"
	"sync"
)

type SlopeOne struct {
	Base
	globalMean  float64
	userRatings [][]IDRating
	userMeans   []float64
	dev         [][]float64
}

func NewSlopeOne(params Parameters) *SlopeOne {
	so := new(SlopeOne)
	so.Params = params
	return new(SlopeOne)
}

func (s *SlopeOne) Predict(userId int, itemId int) float64 {
	innerUserID := s.Data.ConvertUserID(userId)
	innerItemID := s.Data.ConvertItemID(itemId)
	prediction := 0.0

	if innerUserID != newID {
		prediction = s.userMeans[innerUserID]
	} else {
		prediction = s.globalMean
	}

	if innerItemID != newID && innerUserID != newID {
		sum, count := 0.0, 0.0
		for _, ir := range s.userRatings[innerUserID] {
			sum += s.dev[innerItemID][ir.ID]
			count++
		}
		if count > 0 {
			prediction += sum / count
		}
	}

	return prediction
}

func (s *SlopeOne) Fit(trainSet TrainSet) {
	nJobs := runtime.NumCPU()
	s.Data = trainSet
	s.globalMean = trainSet.GlobalMean
	s.userRatings = trainSet.UserRatings()
	s.userMeans = means(s.userRatings)
	s.dev = newZeroMatrix(trainSet.ItemCount, trainSet.ItemCount)
	itemRatings := trainSet.ItemRatings()
	sorts(itemRatings)
	// 计算物品偏差矩阵
	// dev[i][j] 代表i、j物品之间的差值，
	// dev[i][j] = Σ(rating_i - rating_j) / count

	// 并行计算
	length := len(itemRatings)
	var wg sync.WaitGroup
	wg.Add(nJobs)

	for j := 0; j < nJobs; j++ {
		go func(jobID int) {
			defer wg.Done()
			begin := length * jobID / nJobs
			end := length * (jobID + 1) / nJobs
			for i := begin; i < end; i++ {
				for j := 0; j < i; j++ {
					count, sum, ptr := 0.0, 0.0, 0
					for k := 0; k < len(itemRatings[i]) && ptr < len(itemRatings[j]); k++ {
						ur := itemRatings[i][k]
						for ptr < len(itemRatings[j]) && itemRatings[j][ptr].ID < ur.ID {
							ptr++
						}
						if ptr < len(itemRatings[j]) && itemRatings[j][ptr].ID == ur.ID {
							count++
							sum += ur.Rating - itemRatings[j][ptr].Rating
						}
					}
					if count > 0 {
						s.dev[i][j] = sum / count
						s.dev[j][i] = -s.dev[i][j]
					}
				}
			}

		}(j)
	}
	wg.Wait()
}
