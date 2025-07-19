// The SVD++ algorithm, an extension of SVD taking into account implicit
// interactionRatings. The prediction \hat{r}_{ui} is set as:
//
// \hat{r}_{ui} = \mu + b_u + b_i + q_i^T\left(p_u + |I_u|^{-\frac{1}{2}} \sum_{j \in I_u}y_j\right)
//
// Where the y_j terms are a new set of item factors that capture implicit
// interactionRatings. Here, an implicit rating describes the fact that a user u
// userHistory an item j, regardless of the rating value. If user u is unknown,
// then the bias b_u and the factors p_u are assumed to be zero. The same
// applies for item i with b_i, q_i and y_i.

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
	//cacheFactor map[int][]float64
	userBias   map[int]float64
	itemBias   map[int]float64
	globalBias float64
}

func (pp *SVDPP) EnsembleImplFactors(userID int) ([]float64, bool) {
	history, exist := pp.userHistory[userID]
	emImpFactor := make([]float64, 0)
	// 用户交互物品历史没有存在
	if !exist {
		return emImpFactor, false
	}
	// 用户交互物品历史存在
	for _, itemID := range history {
		if len(emImpFactor) == 0 {
			// 初始化，分配内存
			emImpFactor = make([]float64, len(pp.implFactor[itemID]))
		}
		floats.Add(emImpFactor, pp.implFactor[itemID])
	}
	divConst(math.Sqrt(float64(len(history))), emImpFactor)
	return emImpFactor, true
}

func (pp *SVDPP) InternalPredict(userID, itemID int) (float64, []float64) {
	ret := .0
	userFactor := pp.userFactor[userID]
	itemFactor := pp.itemFactor[itemID]
	emImpFactor, _ := pp.EnsembleImplFactors(userID)
	if len(itemFactor) > 0 {
		tmp := make([]float64, len(itemFactor))
		if len(userFactor) > 0 {
			floats.Add(tmp, userFactor)
		}
		if len(emImpFactor) > 0 {
			floats.Add(tmp, emImpFactor)
		}
		ret = floats.Dot(tmp, itemFactor)
	}
	userBias := pp.userBias[userID]
	itemBias := pp.itemBias[itemID]
	ret += userBias + itemBias + pp.globalBias
	return ret, emImpFactor
}
func (pp *SVDPP) Predict(userID, itemID int) float64 {
	predict, _ := pp.InternalPredict(userID, itemID)
	return predict
}

func (pp *SVDPP) Fit(trainData TrainSet, options Options) {
	// Setup options
	nFactors := options.GetInt("nFactors", 20)
	nEpochs := options.GetInt("nEpochs", 20)
	lr := options.GetFloat64("lr", 0.007)
	reg := options.GetFloat64("reg", 0.02)
	initMean := options.GetFloat64("initMean", 0)
	initStdDev := options.GetFloat64("initStdDev", 0.1)

	// 初始化参数
	pp.userBias = make(map[int]float64)
	pp.itemBias = make(map[int]float64)
	pp.userFactor = make(map[int][]float64)
	pp.itemFactor = make(map[int][]float64)
	pp.implFactor = make(map[int][]float64)
	//pp.cacheFactor = make(map[int][]float64)

	for userID := range trainData.Users() {
		pp.userBias[userID] = 0
		pp.userFactor[userID] = newNormalVector(nFactors, initMean, initStdDev)
	}
	for itemID := range trainData.Items() {
		pp.itemBias[itemID] = 0
		pp.itemFactor[itemID] = newNormalVector(nFactors, initMean, initStdDev)
		pp.implFactor[itemID] = newNormalVector(nFactors, initMean, initStdDev)
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

	// 创建缓存
	a := make([]float64, nFactors)
	b := make([]float64, nFactors)

	// 随即梯度下降算法
	// 系数常数已经保存在学习率和正则化系数中
	for epoch := 0; epoch < nEpochs; epoch++ {
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
			pred, emImpFactor := pp.InternalPredict(userID, itemID)
			diff := pred - rating
			// 更新全局偏置
			gradGlobalBias := diff
			pp.globalBias -= lr * gradGlobalBias

			// 更新 User 偏置
			gradUserBias := diff + reg*userBias
			pp.userBias[userID] -= lr * gradUserBias

			// item  偏置
			gradItemBias := diff + reg*itemBias
			pp.itemBias[itemID] -= lr * gradItemBias

			// user 潜在因子
			copy(a, itemFactor)
			mulConst(diff, a)
			copy(b, userFactor)
			mulConst(reg, b)
			floats.Add(a, b)
			mulConst(lr, a)
			floats.Sub(pp.userFactor[userID], a)

			// item 潜在因子
			copy(a, userFactor)
			if len(emImpFactor) > 0 {
				floats.Add(a, emImpFactor)
			}
			mulConst(diff, a)
			copy(b, itemFactor)
			mulConst(reg, b)
			floats.Add(a, b)
			mulConst(lr, a)
			floats.Sub(pp.itemFactor[itemID], a)

			// 隐因子
			set := pp.userHistory[userID]
			for _, itemID := range set {
				implFactor := pp.implFactor[itemID]

				copy(a, itemFactor)
				mulConst(diff, a)
				divConst(math.Sqrt(float64(len(set))), a)

				copy(b, implFactor)
				mulConst(reg, b)

				floats.Add(a, b)
				mulConst(lr, a)

				floats.Sub(pp.implFactor[itemID], a)
			}

		}
	}

}

func NewSVDPP() *SVDPP {
	return new(SVDPP)
}
