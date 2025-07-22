package core

import (
	"gonum.org/v1/gonum/stat"
	"math/rand"
)

type Random struct {
	mean   float64
	stdDev float64
	low    float64
	high   float64
}

func (r *Random) Predict(userID, itemID int) float64 {
	ret := r.mean + r.stdDev*rand.NormFloat64()
	if ret < r.low {
		ret = r.low
	} else if ret > r.high {
		ret = r.high
	}
	return ret
}

func (r *Random) Fit(trainData TrainSet, params Parameters) {
	ratings := trainData.Ratings
	r.mean = stat.Mean(ratings, nil)
	r.stdDev = stat.StdDev(ratings, nil)
	r.low, r.high = trainData.RatingRange()
}

func NewRandom() *Random {
	return new(Random)
}
