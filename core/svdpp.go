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
	userHistory [][]float64
	userFactor  [][]float64
	itemFactor  [][]float64
	implFactor  [][]float64 //y_i
	//cacheFactor map[int][]float64
	userBias   []float64
	itemBias   []float64
	globalBias float64

	trainSet TrainSet
}

func (pp *SVDPP) EnsembleImplFactors(innerUserID int) []float64 {
	history := pp.userHistory[innerUserID]
	emImpFactor := make([]float64, 0)

	// 用户交互物品历史存在
	for itemID := range history {
		if len(emImpFactor) == 0 {
			// 初始化，分配内存
			emImpFactor = make([]float64, len(pp.implFactor[itemID]))
		}
		floats.Add(emImpFactor, pp.implFactor[itemID])
	}
	divConst(math.Sqrt(float64(len(history))), emImpFactor)
	return emImpFactor
}

func (pp *SVDPP) InternalPredict(userID, itemID int) (float64, []float64) {

	// convert to inner id

	innerUserID := pp.trainSet.ConvertUserID(userID)
	innerItemID := pp.trainSet.ConvertItemID(itemID)
	ret := pp.globalBias
	if innerUserID != newID {
		ret += pp.userBias[innerUserID]
	}

	if innerItemID != newID {
		ret += pp.itemBias[innerItemID]
	}
	if innerItemID != newID && innerUserID != newID {
		userFactor := pp.userFactor[innerUserID]
		itemFactor := pp.itemFactor[innerItemID]
		emImpFactor := pp.EnsembleImplFactors(innerUserID)
		tmp := make([]float64, len(itemFactor))
		floats.Add(tmp, userFactor)
		floats.Add(tmp, emImpFactor)
		ret += floats.Dot(tmp, itemFactor)
		return ret, emImpFactor
	}
	return ret, []float64{}

}
func (pp *SVDPP) Predict(userID, itemID int) float64 {
	predict, _ := pp.InternalPredict(userID, itemID)
	return predict
}

func (pp *SVDPP) Fit(trainData TrainSet, params Parameters) {
	// Setup options
	reader := newParameterReader(params)
	nFactors := reader.getInt("nFactors", 20)
	nEpochs := reader.getInt("nEpochs", 20)
	lr := reader.getFloat64("lr", 0.007)
	reg := reader.getFloat64("reg", 0.02)
	initMean := reader.getFloat64("initMean", 0)
	initStdDev := reader.getFloat64("initStdDev", 0.1)

	// 初始化参数
	pp.trainSet = trainData
	pp.userBias = make([]float64, trainData.UserCount())
	pp.itemBias = make([]float64, trainData.ItemCount())
	pp.userFactor = make([][]float64, trainData.UserCount())
	pp.itemFactor = make([][]float64, trainData.ItemCount())
	pp.implFactor = make([][]float64, trainData.ItemCount())
	//pp.cacheFactor = make(map[int][]float64)

	for innerUserID := range pp.userBias {
		pp.userFactor[innerUserID] = newNormalVector(nFactors, initMean, initStdDev)
	}
	for innerItemID := range pp.itemBias {
		pp.itemFactor[innerItemID] = newNormalVector(nFactors, initMean, initStdDev)
		pp.implFactor[innerItemID] = newNormalVector(nFactors, initMean, initStdDev)
	}
	// 创建用户历史物品
	pp.userHistory = trainData.UserRatings()

	// 创建缓存
	a := make([]float64, nFactors)
	b := make([]float64, nFactors)

	// 随即梯度下降算法
	// 系数常数已经保存在学习率和正则化系数中
	for epoch := 0; epoch < nEpochs; epoch++ {
		fmt.Printf("第 %d 轮\n", epoch)
		for i := 0; i < trainData.Length(); i++ {
			userID, itemID, rating := trainData.Index(i)

			innerUserID := trainData.ConvertUserID(userID)
			innerItemID := trainData.ConvertItemID(itemID)

			userBias := pp.userBias[innerUserID]
			itemBias := pp.itemBias[innerItemID]
			userFactor := pp.userFactor[innerUserID]
			itemFactor := pp.itemFactor[innerItemID]
			// 计算差值
			pred, emImpFactor := pp.InternalPredict(userID, itemID)
			diff := pred - rating
			// 更新全局偏置
			gradGlobalBias := diff
			pp.globalBias -= lr * gradGlobalBias

			// 更新 User 偏置
			gradUserBias := diff + reg*userBias
			pp.userBias[innerUserID] -= lr * gradUserBias

			// item  偏置
			gradItemBias := diff + reg*itemBias
			pp.itemBias[innerItemID] -= lr * gradItemBias

			// user 潜在因子
			copy(a, itemFactor)
			mulConst(diff, a)
			copy(b, userFactor)
			mulConst(reg, b)
			floats.Add(a, b)
			mulConst(lr, a)
			floats.Sub(pp.userFactor[innerUserID], a)

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
			floats.Sub(pp.itemFactor[innerItemID], a)

			// 隐因子
			set := pp.userHistory[innerUserID]
			for itemID := range set {
				if !math.IsNaN(pp.userHistory[innerUserID][itemID]) {
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

}

func NewSVDPP() *SVDPP {
	return new(SVDPP)
}
