package algo

import (
	"gonum.org/v1/gonum/stat"
	"math/rand"
	"recommend-sys/data"
)

type Random struct {
	mean   float64
	stdDev float64
}

func (r *Random) Predict(userID, itemID int) float64 {
	return r.mean + r.stdDev*rand.NormFloat64()
}

func (r *Random) Fit(trainData data.Set, option ...OptionsEditor) {
	ratings := trainData.AllRatings()
	r.mean = stat.Mean(ratings, nil)
	r.stdDev = stat.StdDev(ratings, nil)
}

func NewRandomRecommend() *Random {
	return &Random{}
}
