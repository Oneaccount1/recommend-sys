package core

import (
	"gonum.org/v1/gonum/floats"
)

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
	if innerUserID != noBody {
		ret += s.userBias[innerUserID]
	}
	// +b_i
	if innerItemID != noBody {
		ret += s.itemBias[innerItemID]
	}
	// +q_i^Tp_u
	if !(innerItemID == noBody || innerUserID == noBody) {
		userFactor := s.userFactor[innerUserID]
		itemFactor := s.itemFactor[innerItemID]
		ret += floats.Dot(userFactor, itemFactor)
	}
	return ret
}

func (s *SVD) Fit(trainData TrainSet, options ...OptionSetter) {
	option := Option{
		nFactors:   100,
		nEpochs:    20,
		lr:         0.005,
		reg:        0.02,
		biased:     true,
		initMean:   0,
		initStdDev: 0.1,
	}

	for _, editor := range options {
		editor(&option)
	}

	// 初始化参数
	s.trainSet = trainData
	s.userFactor = make([][]float64, s.trainSet.userCount)
	s.itemFactor = make([][]float64, s.trainSet.itemCount)

	s.itemBias = make([]float64, s.trainSet.itemCount)
	s.userBias = make([]float64, s.trainSet.userCount)

	for i := range s.userFactor {
		s.userFactor[i] = NewNormalVector(option.nFactors, option.initMean, option.initStdDev)
	}
	for i := range s.itemFactor {
		s.itemFactor[i] = NewNormalVector(option.nFactors, option.initMean, option.initStdDev)
	}

	users, items, ratings := trainData.Interactions()
	// 创建缓存
	a := make([]float64, option.nFactors)
	b := make([]float64, option.nFactors)

	// 随机梯度下降
	for epoch := 0; epoch < option.nEpochs; epoch++ {
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
			s.globalBias -= option.lr * gradGlobalBias

			gradUserBias := diff + option.reg*userBias
			s.userBias[innerUserID] -= option.lr * gradUserBias

			gradItemBias := diff + option.reg*itemBias
			s.itemBias[innerItemID] -= option.lr * gradItemBias
			// update user latent factor
			copy(a, itemFactor)
			mulConst(diff, a)
			copy(b, userFactor)
			mulConst(option.reg, b)
			floats.Add(a, b)
			mulConst(option.lr, a)
			floats.Sub(s.userFactor[innerUserID], a)

			copy(a, userFactor)
			mulConst(diff, a)
			copy(b, itemFactor)
			mulConst(option.reg, b)
			floats.Add(a, b)
			mulConst(option.lr, a)
			floats.Sub(s.itemFactor[innerItemID], a)
		}
	}

}

func NewSVD() *SVD {
	return &SVD{}
}
