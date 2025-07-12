package core

import (
	"gonum.org/v1/gonum/floats"
)

type SVD struct {
	userFactor map[int][]float64
	itemFactor map[int][]float64
	userBias   map[int]float64
	itemBias   map[int]float64
	globalBias float64
}

func (s *SVD) Predict(userID, itemID int) float64 {
	userFactor, _ := s.userFactor[userID]
	itemFactor, _ := s.itemFactor[itemID]
	dot := .0
	if len(userFactor) == len(itemFactor) {
		dot = floats.Dot(userFactor, itemFactor)
	}

	userBias, _ := s.userBias[userID]
	itemBias, _ := s.itemBias[itemID]
	return s.globalBias + userBias + itemBias + dot
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
	s.userFactor = make(map[int][]float64)
	s.itemFactor = make(map[int][]float64)

	s.itemBias = make(map[int]float64)
	s.userBias = make(map[int]float64)

	for userID := range trainData.Users() {
		s.userBias[userID] = 0
		s.userFactor[userID] = NewNormalVector(option.nFactors, option.initMean, option.initStdDev)
	}

	for itemID := range trainData.Items() {
		s.itemBias[itemID] = 0
		s.itemFactor[itemID] = NewNormalVector(option.nFactors, option.initMean, option.initStdDev)
	}

	users, items, ratings := trainData.Interactions()
	// 创建缓存
	a := make([]float64, option.nFactors)
	b := make([]float64, option.nFactors)

	// 随机梯度下降
	for epoch := 0; epoch < option.nEpochs; epoch++ {
		for i := 0; i < trainData.Length(); i++ {
			userID := users[i]
			itemID := items[i]
			rating := ratings[i]
			userBias := s.userBias[userID]
			itemBias := s.itemBias[itemID]
			userFactor := s.userFactor[userID]
			itemFactor := s.itemFactor[itemID]
			// 计算差值
			diff := s.Predict(userID, itemID) - rating

			// 计算各个参数的梯度
			gradGlobalBias := diff
			s.globalBias -= option.lr * gradGlobalBias

			gradUserBias := diff + option.reg*userBias
			s.userBias[userID] -= option.lr * gradUserBias

			gradItemBias := diff + option.reg*itemBias
			s.itemBias[itemID] -= option.lr * gradItemBias
			// update user latent factor
			copy(a, itemFactor)
			mulConst(diff, a)
			copy(b, userFactor)
			mulConst(option.reg, b)
			floats.Add(a, b)
			mulConst(option.lr, a)
			floats.Sub(s.userFactor[userID], a)

			copy(a, userFactor)
			mulConst(diff, a)
			copy(b, itemFactor)
			mulConst(option.reg, b)
			floats.Add(a, b)
			mulConst(option.lr, a)
			floats.Sub(s.itemFactor[itemID], a)
		}
	}

}

func NewSVD() *SVD {
	return &SVD{}
}
