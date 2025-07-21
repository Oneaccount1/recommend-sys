package core

import (
	"gonum.org/v1/gonum/floats"
)

// The famous SVD algorithm, as popularized by Simon Funk during the
// Netflix Prize. The prediction \hat{r}_{ui} is set as:
//
//	\hat{r}_{ui} = μ + b_u + b_i + q_i^Tp_u
//
// If user u is unknown, then the bias b_u and the factors p_u are
// assumed to be zero. The same applies for item i with b_i and q_i.
type SVD struct {
	userFactor [][]float64
	itemFactor [][]float64
	userBias   []float64
	itemBias   []float64
	globalBias float64
	trainSet   TrainSet
}

func (s *SVD) Predict(userID, itemID int) float64 {
	innerUserID := s.trainSet.ConvertUserID(userID)
	innerItemID := s.trainSet.ConvertItemID(itemID)
	ret := s.globalBias
	// +b_u
	if innerUserID != newID {
		ret += s.userBias[innerUserID]
	}
	// +b_i
	if innerItemID != newID {
		ret += s.itemBias[innerItemID]
	}
	// +q_i^Tp_u
	if !(innerItemID == newID || innerUserID == newID) {
		userFactor := s.userFactor[innerUserID]
		itemFactor := s.itemFactor[innerItemID]
		ret += floats.Dot(userFactor, itemFactor)
	}
	return ret
}

// Fit a SVD model.
// Parameters:
//
//	 reg 		- The regularization parameter of the cost function that is
//				  optimized. Default is 0.02.
//	 lr 		- The learning rate of SGD. Default is 0.005.
//	 nFactors	- The number of latent factors. Default is 100.
//	 nEpochs	- The number of iteration of the SGD procedure. Default is 20.
//	 initMean	- The means of initial random latent factors. Default is 0.
//	 initStdDev	- The standard deviation of initial random latent factors. Default is 0.1.
func (s *SVD) Fit(trainData TrainSet, params Parameters) {
	// Setup options
	reader := parameterReader{params}
	nFactors := reader.getInt("nFactors", 100)
	nEpochs := reader.getInt("nEpochs", 20)
	lr := reader.getFloat64("lr", 0.005)
	reg := reader.getFloat64("reg", 0.02)
	//biased := options.GetBool("biased", true)
	initMean := reader.getFloat64("initMean", 0)
	initStdDev := reader.getFloat64("initStdDev", 0.1)
	// 初始化参数
	s.trainSet = trainData
	s.userFactor = make([][]float64, s.trainSet.userCount)
	s.itemFactor = make([][]float64, s.trainSet.itemCount)

	s.itemBias = make([]float64, s.trainSet.itemCount)
	s.userBias = make([]float64, s.trainSet.userCount)

	for i := range s.userFactor {
		s.userFactor[i] = newNormalVector(nFactors, initMean, initStdDev)
	}
	for i := range s.itemFactor {
		s.itemFactor[i] = newNormalVector(nFactors, initMean, initStdDev)
	}

	users, items, ratings := trainData.Interactions()
	// 创建缓存
	a := make([]float64, nFactors)
	b := make([]float64, nFactors)

	// 随机梯度下降
	for epoch := 0; epoch < nEpochs; epoch++ {
		for i := 0; i < trainData.Length(); i++ {
			userID, itemID, rating := users[i], items[i], ratings[i]
			innerUserID := trainData.ConvertUserID(userID)
			innerItemID := trainData.ConvertItemID(itemID)
			userBias := s.userBias[innerUserID]
			itemBias := s.itemBias[innerItemID]
			userFactor := s.userFactor[innerUserID]
			itemFactor := s.itemFactor[innerItemID]
			// 计算差值
			diff := s.Predict(userID, itemID) - rating

			// 计算各个参数的梯度
			gradGlobalBias := diff
			s.globalBias -= lr * gradGlobalBias

			gradUserBias := diff + reg*userBias
			s.userBias[innerUserID] -= lr * gradUserBias

			gradItemBias := diff + reg*itemBias
			s.itemBias[innerItemID] -= lr * gradItemBias
			// update user latent factor
			copy(a, itemFactor)
			mulConst(diff, a)
			copy(b, userFactor)
			mulConst(reg, b)
			floats.Add(a, b)
			mulConst(lr, a)
			floats.Sub(s.userFactor[innerUserID], a)

			copy(a, userFactor)
			mulConst(diff, a)
			copy(b, itemFactor)
			mulConst(reg, b)
			floats.Add(a, b)
			mulConst(lr, a)
			floats.Sub(s.itemFactor[innerItemID], a)
		}
	}

}

func NewSVD() *SVD {
	return &SVD{}
}
