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
		nFactors: 100,
		nEpochs:  20,
		lr:       0.005,
		reg:      0.02,
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
		s.userFactor[userID] = make([]float64, option.nFactors)
	}

	for itemID := range trainData.Items() {
		s.itemBias[itemID] = 0
		s.itemFactor[itemID] = make([]float64, option.nFactors)
	}

	users, items, ratings := trainData.Interactions()

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
			diff := rating - s.globalBias - userBias - itemBias - floats.Dot(userFactor, itemFactor)

			// 计算各个参数的梯度
			gradGlobalBias := -2 * diff
			s.globalBias -= option.lr * gradGlobalBias

			gradUserBias := -2*diff + 2*option.reg*userBias
			s.userBias[userID] -= option.lr * gradUserBias

			gradItemBias := -2*diff + 2*option.reg*itemBias
			s.itemBias[itemID] -= option.lr * gradItemBias

			//gradUserFactor := Copy(buffer, userFactor)
			gradUserFactor := MulConst(-2*diff, itemFactor)
			floats.Add(gradUserFactor, MulConst(2*option.reg, userFactor))

			//floats.Sub(MulConst(-2*diff, itemFactor), MulConst(2*option.regularization, userFactor))
			floats.Sub(s.userFactor[userID], MulConst(option.lr, gradUserFactor))

			gradItemFactor := MulConst(-2*diff, userFactor)
			floats.Add(gradItemFactor, MulConst(2*option.reg, itemFactor))
			//gradItemFactor := Copy(buffer, itemFactor)
			//floats.Sub(MulConst(-2*diff, userFactor), MulConst(2*option.regularization, itemFactor))
			floats.Sub(s.itemFactor[itemID], MulConst(option.lr, gradItemFactor))
		}
	}

}

func NewSVD() *SVD {
	return &SVD{}
}
