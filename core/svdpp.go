package core

import (
	"fmt"
	"gonum.org/v1/gonum/floats"
	"math"
)

type SVDPP struct {
	userHistory map[int][]int
	userFactor  map[int][]float64
	itemFactor  map[int][]float64
	implFactor  map[int][]float64 //y_i
	cacheFactor map[int][]float64
	userBias    map[int]float64
	itemBias    map[int]float64
	globalBias  float64
}

func (pp *SVDPP) Predict(userID, itemID int) float64 {
	ret := .0
	userFactor, _ := pp.userFactor[userID]
	itemFactor, _ := pp.itemFactor[itemID]
	if len(itemFactor) > 0 {
		tmp := make([]float64, len(itemFactor))
		// + y_i
		history, historyAvailable := pp.userHistory[userID]
		cacheFactor, cacheAvailable := pp.cacheFactor[userID]

		if cacheAvailable {

			floats.Add(tmp, cacheFactor)
		} else if historyAvailable {

			pp.cacheFactor[userID] = make([]float64, len(userFactor))

			for _, itemID := range history {
				floats.Add(pp.cacheFactor[userID], pp.implFactor[itemID])
			}

			DivConst(math.Sqrt(float64(len(history))), pp.cacheFactor[userID])

			floats.Add(tmp, pp.cacheFactor[userID])
		}
		// + p_u
		if len(userFactor) > 0 {
			floats.Add(tmp, userFactor)
		}

		// dot q_i
		ret = floats.Dot(itemFactor, tmp)
	}
	// + b_u + b_i + mu
	userBias := pp.userBias[userID]
	itemBias := pp.itemBias[itemID]
	ret += userBias + itemBias + pp.globalBias
	return ret
}

func (pp *SVDPP) Fit(trainData TrainSet, options ...OptionSetter) {
	// 设置参数
	option := Option{
		nFactors:   20,
		nEpochs:    20,
		lr:         0.007,
		reg:        0.02,
		initMean:   0,
		initStdDev: 0.1,
	}
	for _, editor := range options {
		editor(&option)
	}

	// 初始化参数
	pp.userBias = make(map[int]float64)
	pp.itemBias = make(map[int]float64)
	pp.userFactor = make(map[int][]float64)
	pp.itemFactor = make(map[int][]float64)
	pp.implFactor = make(map[int][]float64)
	pp.cacheFactor = make(map[int][]float64)

	for userID := range trainData.Users() {
		pp.userBias[userID] = 0
		pp.userFactor[userID] = NewNormalVector(option.nFactors, option.initMean, option.initStdDev)
	}
	for itemID := range trainData.Items() {
		pp.itemBias[itemID] = 0
		pp.itemFactor[itemID] = NewNormalVector(option.nFactors, option.initMean, option.initStdDev)
		pp.implFactor[itemID] = NewNormalVector(option.nFactors, option.initMean, option.initStdDev)
	}
	// 创建用户历史物品
	pp.userHistory = make(map[int][]int)
	users, items, ratings := trainData.Interactions()
	for i := 0; i < len(users); i++ {
		userID := users[i]
		itemID := items[i]

		if _, exists := pp.userHistory[userID]; !exists {
			pp.userHistory[userID] = make([]int, 0)
		}
		// 插入物品
		pp.userHistory[userID] = append(pp.userHistory[userID], itemID)
	}
	// 随即梯度下降算法
	buffer := make([]float64, option.nFactors)
	// 系数常数已经保存在学习率和正则化系数中
	for epoch := 0; epoch < option.nEpochs; epoch++ {
		fmt.Printf("第 %d 轮\n", epoch)
		for i := 0; i < trainData.Length(); i++ {
			userID := users[i]
			itemID := items[i]
			rating := ratings[i]
			userBias := pp.userBias[userID]
			itemBias := pp.itemBias[itemID]
			userFactor := pp.userFactor[userID]
			itemFactor := pp.itemFactor[itemID]
			// 计算差值
			diff := pp.Predict(userID, itemID) - rating
			// 更新全局偏置
			gradGlobalBias := diff
			pp.globalBias -= option.lr * gradGlobalBias

			// 更新 User 偏置
			gradUserBias := diff + option.reg*userBias
			pp.userBias[userID] -= option.lr * gradUserBias

			// item  偏置
			gradItemBias := diff + option.reg*itemBias
			pp.itemBias[itemID] -= option.lr * gradItemBias

			// user 潜在因子
			gradUserFactor := Copy(buffer, itemFactor)
			floats.Add(MulConst(diff, gradUserFactor), MulConst(option.reg, userFactor))
			floats.Sub(pp.userFactor[userID], MulConst(option.lr, gradUserFactor))

			// item 潜在因子
			gradItemFactor := Copy(buffer, userFactor)
			floats.Add(MulConst(diff, gradItemFactor), MulConst(option.reg, itemFactor))
			floats.Sub(pp.itemFactor[itemID], MulConst(option.lr, gradItemFactor))

			// 隐因子
			set := pp.userHistory[userID]
			for _, itemID := range set {
				implFactor := pp.implFactor[itemID]

				gradImplFactor := Copy(buffer, itemFactor)
				MulConst(diff, gradImplFactor)
				DivConst(math.Sqrt(float64(len(set))), gradImplFactor)
				floats.Add(gradImplFactor, MulConst(option.reg, implFactor))
				floats.Sub(pp.implFactor[itemID], MulConst(option.lr, gradImplFactor))
			}

		}
	}

}

func NewSVDPPRecommend() *SVDPP {
	return new(SVDPP)
}
