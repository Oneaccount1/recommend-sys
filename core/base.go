package core

import (
	"gonum.org/v1/gonum/stat"
	"math/rand"
)

type Estimator interface {
	SetParams(params Parameters)
	Predict(userId, itemId int) float64
	Fit(trainSet TrainSet)
}

type Parameters map[string]interface{}

func (parameters Parameters) Copy() Parameters {
	newParams := make(Parameters)
	for k, v := range parameters {
		newParams[k] = v
	}
	return newParams
}

func (parameters Parameters) GetInt(name string, _default int) int {
	if val, exist := parameters[name]; exist {
		return val.(int)
	}
	return _default
}

func (parameters Parameters) GetBool(name string, _default bool) bool {
	if val, exist := parameters[name]; exist {
		return val.(bool)
	}
	return _default
}

func (parameters Parameters) GetFloat64(name string, _default float64) float64 {
	if val, exist := parameters[name]; exist {
		return val.(float64)
	}
	return _default
}

func (parameters Parameters) GetSim(name string, _default Sim) Sim {
	if val, exist := parameters[name]; exist {
		return val.(Sim)
	}
	return _default
}

func (parameters Parameters) GetString(name string, _default string) string {
	if val, exist := parameters[name]; exist {
		return val.(string)
	}
	return _default
}

type Base struct {
	Params Parameters
	Data   TrainSet
}

func (base *Base) SetParams(params Parameters) {
	base.Params = params
}

func (base *Base) Predict(userId, itemId int) float64 {
	panic("Predict() not implemented")
}

func (base *Base) Fit(trainSet TrainSet) {
	panic("Fit() not implemented")
}

type Random struct {
	Base
	Mean   float64 // mu
	StdDev float64 // sigma
	Low    float64 // The lower bound of rating scores
	High   float64 // The upper bound of rating scores
}

func NewRandom(params Parameters) *Random {
	random := new(Random)
	random.Params = params
	return random
}

func (random *Random) Predict(userId int, itemId int) float64 {
	ret := rand.NormFloat64()*random.StdDev + random.Mean
	// Crop prediction
	if ret < random.Low {
		ret = random.Low
	} else if ret > random.High {
		ret = random.High
	}
	return ret
}

func (random *Random) Fit(trainSet TrainSet) {
	ratings := trainSet.Ratings
	random.Mean = trainSet.GlobalMean
	random.StdDev = stat.StdDev(ratings, nil)
	random.Low, random.High = trainSet.RatingRange()
}

type BaseLine struct {
	Base
	userBias   []float64 // b_u
	itemBias   []float64 // b_i
	globalBias float64   // mu
	trainSet   TrainSet
}

func NewBaseLine(params Parameters) *BaseLine {
	baseLine := new(BaseLine)
	baseLine.Params = params
	return baseLine
}

func (baseLine *BaseLine) Predict(userId, itemId int) float64 {
	// Convert to inner Id
	innerUserId := baseLine.trainSet.ConvertUserID(userId)
	innerItemId := baseLine.trainSet.ConvertItemID(itemId)
	ret := baseLine.globalBias
	if innerUserId != newID {
		ret += baseLine.userBias[innerUserId]
	}
	if innerItemId != newID {
		ret += baseLine.itemBias[innerItemId]
	}
	return ret
}
func (baseLine *BaseLine) Fit(trainSet TrainSet) {
	// Setup parameters
	reg := baseLine.Params.GetFloat64("reg", 0.02)
	lr := baseLine.Params.GetFloat64("lr", 0.005)
	nEpochs := baseLine.Params.GetInt("nEpochs", 20)
	// Initialize parameters
	baseLine.trainSet = trainSet
	baseLine.userBias = make([]float64, trainSet.UserCount)
	baseLine.itemBias = make([]float64, trainSet.ItemCount)
	// Stochastic Gradient Descent
	for epoch := 0; epoch < nEpochs; epoch++ {
		for i := 0; i < trainSet.Length(); i++ {
			userId, itemId, rating := trainSet.Users[i], trainSet.Items[i], trainSet.Ratings[i]
			innerUserId := trainSet.ConvertUserID(userId)
			innerItemId := trainSet.ConvertItemID(itemId)
			userBias := baseLine.userBias[innerUserId]
			itemBias := baseLine.itemBias[innerItemId]
			// Compute gradient
			diff := baseLine.Predict(userId, itemId) - rating
			gradGlobalBias := diff
			gradUserBias := diff + reg*userBias
			gradItemBias := diff + reg*itemBias
			// Update parameters
			baseLine.globalBias -= lr * gradGlobalBias
			baseLine.userBias[innerUserId] -= lr * gradUserBias
			baseLine.itemBias[innerItemId] -= lr * gradItemBias
		}
	}
}
