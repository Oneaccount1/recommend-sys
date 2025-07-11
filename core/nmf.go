//  基于NMF的协同过滤算法， 与SVD非常类似

package core

import "gonum.org/v1/gonum/floats"

type NMF struct {
	userFactor map[int][]float64 // p_u
	itemFactor map[int][]float64 // q_i
}

func (N *NMF) Predict(userId int, itemId int) float64 {
	userFactor, _ := N.userFactor[userId]
	itemFactor, _ := N.itemFactor[itemId]
	if len(itemFactor) == len(userFactor) {
		return floats.Dot(userFactor, itemFactor)
	}
	return 0
}

func (N *NMF) Fit(trainSet TrainSet, options ...OptionSetter) {
	option := Option{
		nFactors: 15,
		nEpochs:  50,
		initLow:  0,
		initHigh: 1,
		reg:      0.06,
		lr:       0.005,
	}

	for _, editor := range options {
		editor(&option)
	}
	// 初始化参数
	N.userFactor = make(map[int][]float64)
	N.itemFactor = make(map[int][]float64)
	for userID := range trainSet.Users() {
		N.userFactor[userID] = NewNormalVector(option.nFactors, option.initMean, option.initStdDev)
	}
	for itemID := range trainSet.Items() {
		N.itemFactor[itemID] = NewNormalVector(option.nFactors, option.initMean, option.initStdDev)
	}

}

func NewNMF() *NMF {
	return new(NMF)
}
